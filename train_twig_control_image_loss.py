from __future__ import annotations

import argparse
import glob
import io
import inspect
import os
import time
from dataclasses import asdict
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms
from tqdm import tqdm

import pyarrow.dataset as ds
from PIL import Image
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor

from config_io import build_latent_controller_config, load_json_config, resolve_config_path
from models.latent_control.controller import LatentController


def _find_parquets(glob_str: str) -> List[str]:
    paths = sorted(glob.glob(glob_str))
    if paths:
        return paths
    # 常见路径兜底（用户可能忘了写前导 /）
    if not glob_str.startswith("/"):
        paths = sorted(glob.glob("/" + glob_str))
        if paths:
            return paths
    # 如果用户写的是 /root/autodl-fs/data/*.parquet，但真实挂载在 /autodl-fs/data
    if glob_str.startswith("/root/autodl-fs/"):
        alt = glob_str.replace("/root/autodl-fs/", "/autodl-fs/")
        paths = sorted(glob.glob(alt))
        if paths:
            return paths
    return []


def _decode_parquet_image(cell) -> Image.Image:
    """
    parquet 里 image 列通常是 struct{bytes, path}，兼容：
    - dict: {'bytes': b'...', 'path': '...'}
    - None / bytes / path(str)
    """
    if cell is None:
        raise ValueError("image cell is None")

    if isinstance(cell, dict):
        b = cell.get("bytes", None)
        p = cell.get("path", None)
        if b is not None and len(b) > 0:
            return Image.open(io.BytesIO(b)).convert("RGB")
        if p:
            return Image.open(p).convert("RGB")
        raise ValueError(f"image dict has no bytes/path: keys={list(cell.keys())}")

    if isinstance(cell, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(cell))).convert("RGB")

    if isinstance(cell, str):
        return Image.open(cell).convert("RGB")

    # pyarrow scalar
    if hasattr(cell, "as_py"):
        return _decode_parquet_image(cell.as_py())

    raise TypeError(f"Unsupported image cell type: {type(cell)}")


class ParquetImageCaptionIterable(IterableDataset):
    def __init__(
        self,
        parquet_glob: str,
        image_key: str = "image",
        caption_key: str = "caption_composition",
        batch_rows: int = 256,
    ):
        super().__init__()
        self.files = _find_parquets(parquet_glob)
        if not self.files:
            raise FileNotFoundError(f"No parquet files found for glob: {parquet_glob}")
        self.image_key = image_key
        self.caption_key = caption_key
        self.batch_rows = int(max(1, batch_rows))

    def __iter__(self):
        dataset = ds.dataset(self.files, format="parquet")
        cols = [self.image_key, self.caption_key]
        scanner = dataset.scanner(columns=cols, batch_size=self.batch_rows)
        for rb in scanner.to_batches():
            data = rb.to_pydict()
            imgs = data.get(self.image_key)
            caps = data.get(self.caption_key)
            if imgs is None or caps is None:
                raise KeyError(f"Missing columns in parquet batch. keys={list(data.keys())}")
            for img_cell, cap in zip(imgs, caps):
                if cap is None:
                    continue
                yield img_cell, str(cap)


def _build_prompt(processor: VLChatProcessor, caption: str) -> str:
    conversation = [
        {"role": "<|User|>", "content": caption},
        {"role": "<|Assistant|>", "content": ""},
    ]
    sft = processor.apply_sft_template_for_multi_turn_prompts(
        conversations=conversation,
        sft_format=processor.sft_format,
        system_prompt="",
    )
    return sft + processor.image_start_tag


def _expand_cfg_batch(x: torch.Tensor) -> torch.Tensor:
    """
    [B,...] -> [2B,...] with [cond0, uncond0, cond1, uncond1, ...]
    """
    b = x.shape[0]
    x2 = x.unsqueeze(1).expand(b, 2, *x.shape[1:]).reshape(b * 2, *x.shape[1:])
    return x2


@torch.no_grad()
def _freeze_all_params(m: torch.nn.Module):
    for p in m.parameters():
        p.requires_grad_(False)


def _set_trainable_control(controller: LatentController):
    # 显式开启（以防上游误冻结）
    for p in controller.parameters():
        p.requires_grad_(True)


def _soft_codebook_quant(
    probs: torch.Tensor,  # [B, V]
    codebook_weight: torch.Tensor,  # [V, C] where C=8
    topk: int,
) -> torch.Tensor:
    """
    用 top-k 概率做 soft quant embedding：z = sum_k p_k * e[idx_k]
    返回 [B, C]
    """
    k = int(max(1, min(topk, probs.shape[-1])))
    p, idx = torch.topk(probs, k=k, dim=-1)
    p = p / (p.sum(dim=-1, keepdim=True) + 1e-8)
    emb = codebook_weight[idx]  # [B,k,C]
    return (p.unsqueeze(-1) * emb).sum(dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_glob", type=str, default="/autodl-fs/data/*.parquet")
    ap.add_argument("--image_key", type=str, default="image")
    ap.add_argument("--caption_key", type=str, default="caption_composition")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--out_dir", type=str, default="/root/TwiGpipline/checkpoints_control_image_loss")
    ap.add_argument("--device", type=str, default="cuda")

    # soft generation knobs
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--cfg_weight", type=float, default=5.0)
    ap.add_argument("--topk_decode", type=int, default=32)
    ap.add_argument("--tbptt_window", type=int, default=64, help="只对最后 W 个生成步反传，降低显存")
    ap.add_argument("--image_tokens", type=int, default=0, help="覆盖配置里的 image_token_num（用于debug，0表示不覆盖）")
    ap.add_argument("--trigger_every", type=int, default=32, help="训练时每多少步触发一次注入（覆盖配置）")

    # loss knobs
    ap.add_argument("--l1_weight", type=float, default=1.0)
    ap.add_argument("--mse_weight", type=float, default=0.1)

    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # -----------------------------
    # 配置加载
    # -----------------------------
    default_cfg_path = os.path.join(os.path.dirname(__file__), "twig_config.json")
    cfg_path = resolve_config_path(default_cfg_path)
    cfg = load_json_config(cfg_path)

    model_path = cfg.get("model_path", "deepseek-ai/Janus-Pro-7B")
    img_size = int(cfg.get("img_size", 384))
    patch_size = int(cfg.get("patch_size", 16))
    image_token_num = int(cfg.get("image_token_num", 576))
    if int(args.image_tokens) > 0:
        image_token_num = int(args.image_tokens)

    # 期望 token_num == (img_size/patch)^2
    side = img_size // patch_size
    expected = side * side
    if expected != image_token_num:
        print(f"[WARN] image_token_num={image_token_num} but (img_size/patch)^2={expected}; using {image_token_num} tokens anyway.")

    # -----------------------------
    # 模型加载（冻结 Janus-Pro）
    # -----------------------------
    processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer

    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model = model.to(device=device, dtype=torch.float16)
    model.eval()
    _freeze_all_params(model)

    # -----------------------------
    # Control 模块（可训练）
    # -----------------------------
    latent_cfg = build_latent_controller_config(cfg)
    # 训练时希望每步都可注入（否则 32 步一次太稀疏）
    latent_cfg.enabled = True
    latent_cfg.trigger.check_every = int(max(1, args.trigger_every))
    latent_cfg.max_triggers_per_image = max(int(image_token_num), int(latent_cfg.max_triggers_per_image))
    d_model = int(model.language_model.get_input_embeddings().weight.shape[1])
    controller = LatentController(d_model=d_model, tokenizer=tokenizer, cfg=latent_cfg).to(device)
    # 训练：把 control 模块升到 FP32，避免 GradScaler 报 “unscale FP16 gradients”
    controller = controller.to(dtype=torch.float32)
    controller.train()
    _set_trainable_control(controller)

    # -----------------------------
    # 数据
    # -----------------------------
    tfm = transforms.Compose(
        [
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),  # [0,1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1,1]
        ]
    )

    ds_iter = ParquetImageCaptionIterable(
        parquet_glob=args.data_glob,
        image_key=args.image_key,
        caption_key=args.caption_key,
        batch_rows=256,
    )

    def collate(batch):
        imgs = []
        caps = []
        for img_cell, cap in batch:
            try:
                pil = _decode_parquet_image(img_cell)
                imgs.append(tfm(pil))
                caps.append(cap)
            except Exception:
                # 丢掉坏样本
                continue
        if not imgs:
            return None
        return torch.stack(imgs, dim=0), caps

    loader = DataLoader(
        ds_iter,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate,
    )

    # -----------------------------
    # Optimizer / AMP
    # -----------------------------
    # 关掉 foreach 可以显著降低 optimizer.step 的峰值显存（尤其是卡已经很满时）
    adamw_sig = inspect.signature(torch.optim.AdamW).parameters
    opt_kwargs = dict(lr=args.lr, weight_decay=args.weight_decay, foreach=False)
    if "fused" in adamw_sig and device.type == "cuda":
        # fused AdamW 在部分版本上更省显存/更快；不可用时自动忽略
        opt_kwargs["fused"] = True
    opt = torch.optim.AdamW(controller.parameters(), **opt_kwargs)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    # codebook weight for soft decode: [V, 8]
    # 注意：VQ quantizer 可能使用 l2_norm，weight 用其内部逻辑的 embedding.weight 即可
    codebook_weight = model.gen_vision_model.quantize.embedding.weight  # [V,8]

    # -----------------------------
    # Train loop
    # -----------------------------
    step = 0
    t0 = time.time()
    pbar = tqdm(total=args.max_steps, desc="train", dynamic_ncols=True)

    it = iter(loader)
    while step < args.max_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            continue

        if batch is None:
            continue

        gt_img, caps = batch
        gt_img = gt_img.to(device=device, dtype=torch.float16, non_blocking=True)
        bsz = gt_img.shape[0]

        # 组 prompt ids -> CFG batch
        prompts = [_build_prompt(processor, c) for c in caps]
        # 训练脚本里用最简单方案：每个 batch 内的 prompt 长度可能不同，做 padding 到 max_len
        id_seqs = [tokenizer.encode(p) for p in prompts]
        max_len = max(len(x) for x in id_seqs)
        tokens = torch.full((bsz * 2, max_len), fill_value=processor.pad_id, device=device, dtype=torch.long)
        for i, ids in enumerate(id_seqs):
            ids_t = torch.tensor(ids, device=device, dtype=torch.long)
            tokens[2 * i, : ids_t.numel()] = ids_t  # cond
            tokens[2 * i + 1, : ids_t.numel()] = ids_t  # uncond (后面再抹掉中间)
            if ids_t.numel() > 2:
                tokens[2 * i + 1, 1 : ids_t.numel() - 1] = processor.pad_id

        # TBPTT：只对最后 W 步开梯度（显存关键）
        window = int(max(1, min(args.tbptt_window, image_token_num)))
        warm_steps = int(max(0, image_token_num - window))

        # prompt embedding
        prompt_embeds = model.language_model.get_input_embeddings()(tokens)  # [2B,T,D]
        prompt_vec = prompt_embeds[0::2].mean(dim=1)  # [B,D]

        controller.reset(batch_size=bsz, device=device)
        past_key_values = None
        inputs_embeds = prompt_embeds

        soft_quant_seq: List[torch.Tensor] = []

        # ---------- warmup: no_grad ----------
        if warm_steps > 0:
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
                for t in range(warm_steps):
                    out = model.language_model.model(
                        inputs_embeds=inputs_embeds,
                        use_cache=True,
                        past_key_values=past_key_values,
                    )
                    past_key_values = out.past_key_values
                    h_last = out.last_hidden_state[:, -1, :]  # [2B,D]

                    logits2 = model.gen_head(h_last)  # [2B,V]
                    cond, uncond = logits2[0::2], logits2[1::2]
                    logits = uncond + float(args.cfg_weight) * (cond - uncond)  # [B,V]
                    probs = torch.softmax(logits / float(args.temperature), dim=-1)  # [B,V]

                    z = _soft_codebook_quant(probs, codebook_weight=codebook_weight, topk=args.topk_decode)  # [B,8]
                    soft_quant_seq.append(z.detach())

                    think_text = f"prompt: {caps[0][:120]}" if len(caps) == 1 else "prompt batch"
                    past_key_values, _ = controller.maybe_inject(
                        model=model,
                        past_key_values=past_key_values,
                        step_idx=t,
                        prompt_vec=prompt_vec,
                        h_img_last_cond=h_last[0::2],
                        next_token_probs=probs,
                        prompt_text_for_think=think_text,
                    )

                    soft_gen = torch.matmul(probs, model.gen_embed.weight)  # [B, n_embed]
                    img_embed = model.gen_aligner(soft_gen)  # [B, D]
                    inputs_embeds = _expand_cfg_batch(img_embed).unsqueeze(1)  # [2B,1,D]

        # 进入可训练阶段：把 warmup 得到的 KV cache 从图里切断（避免把整段挂图）
        if past_key_values is not None:
            past_key_values = tuple(
                tuple(x.detach() if torch.is_tensor(x) else x for x in layer) for layer in past_key_values
            )

        # ---------- train window: grad enabled ----------
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda"), dtype=torch.float16):
            for t in range(warm_steps, image_token_num):
                out = model.language_model.model(
                    inputs_embeds=inputs_embeds,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                past_key_values = out.past_key_values
                h_last = out.last_hidden_state[:, -1, :]  # [2B,D]

                logits2 = model.gen_head(h_last)  # [2B,V]
                cond, uncond = logits2[0::2], logits2[1::2]
                logits = uncond + float(args.cfg_weight) * (cond - uncond)  # [B,V]
                probs = torch.softmax(logits / float(args.temperature), dim=-1)  # [B,V]

                z = _soft_codebook_quant(probs, codebook_weight=codebook_weight, topk=args.topk_decode)  # [B,8]
                soft_quant_seq.append(z)

                think_text = f"prompt: {caps[0][:120]}" if len(caps) == 1 else "prompt batch"
                past_key_values, _ = controller.maybe_inject(
                    model=model,
                    past_key_values=past_key_values,
                    step_idx=t,
                    prompt_vec=prompt_vec,
                    h_img_last_cond=h_last[0::2],
                    next_token_probs=probs,
                    prompt_text_for_think=think_text,
                )

                soft_gen = torch.matmul(probs, model.gen_embed.weight)  # [B, n_embed]
                img_embed = model.gen_aligner(soft_gen)  # [B, D]
                inputs_embeds = _expand_cfg_batch(img_embed).unsqueeze(1)  # [2B,1,D]

            z_all = torch.stack(soft_quant_seq, dim=1)  # [B,T,8]
            if z_all.shape[1] != expected:
                side2 = int((z_all.shape[1]) ** 0.5)
                if side2 * side2 != z_all.shape[1]:
                    raise RuntimeError(f"Cannot reshape {z_all.shape[1]} tokens into square grid.")
                hh = ww = side2
            else:
                hh = ww = side
            z_grid = z_all.view(bsz, hh, ww, 8).permute(0, 3, 1, 2).contiguous()  # [B,8,H,W]

            pred = model.gen_vision_model.decode(z_grid)  # [B,3,img_size,img_size]

            # debug / 非标准 token 数时：预测分辨率可能 != 目标分辨率
            if pred.shape[-2:] != gt_img.shape[-2:]:
                gt_for_loss = F.interpolate(
                    gt_img.to(dtype=pred.dtype),
                    size=pred.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                gt_for_loss = gt_img

            l1 = F.l1_loss(pred, gt_for_loss)
            mse = F.mse_loss(pred, gt_for_loss)
            loss = float(args.l1_weight) * l1 + float(args.mse_weight) * mse

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        step += 1
        pbar.update(1)

        if step % args.log_every == 0:
            dt = time.time() - t0
            pbar.set_postfix(
                loss=float(loss.detach().cpu()),
                l1=float(l1.detach().cpu()),
                mse=float(mse.detach().cpu()),
                sps=step / max(1e-6, dt),
            )

        if step % args.save_every == 0 or step == args.max_steps:
            ckpt = {
                "step": step,
                "cfg_path": cfg_path,
                "latent_cfg": asdict(latent_cfg),
                "controller": controller.state_dict(),
                "opt": opt.state_dict(),
                "args": vars(args),
            }
            out = os.path.join(args.out_dir, f"ckpt_step{step}.pt")
            torch.save(ckpt, out)

    pbar.close()


if __name__ == "__main__":
    main()


