from __future__ import annotations

import argparse
import io
import os
import sys
import time
import warnings
from dataclasses import asdict
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms
from torchvision import models as tv_models
from tqdm import tqdm

import pyarrow.dataset as ds
from PIL import Image
from transformers import AutoModelForCausalLM

from janus.models import MultiModalityCausalLM, VLChatProcessor

# 允许从任意工作目录启动（尤其是 torchrun 从 repo root 启动）
_THIS_DIR = os.path.dirname(__file__)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from config_io import build_latent_controller_config, load_json_config, resolve_config_path  # noqa: E402
from models.latent_control.controller import LatentController  # noqa: E402


def _print(msg: str):
    print(msg, flush=True)

def _is_main_process() -> bool:
    try:
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return torch.distributed.get_rank() == 0
    except Exception:
        return True
    return True


def _autocast_ctx(device: torch.device):
    """
    统一 AMP autocast 写法，避免 torch.cuda.amp 的 FutureWarning。
    """
    if device.type == "cuda":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    # cpu 上不开 autocast
    return torch.amp.autocast("cpu", enabled=False)


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
        parquet_files: List[str],
        image_key: str = "image",
        caption_key: str = "caption_composition",
        batch_rows: int = 256,
    ):
        super().__init__()
        self.files = list(parquet_files)
        if not self.files:
            raise FileNotFoundError("No parquet files provided.")
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


def _expand_cfg_batch(x: torch.Tensor) -> torch.Tensor:
    b = x.shape[0]
    x2 = x.unsqueeze(1).expand(b, 2, *x.shape[1:]).reshape(b * 2, *x.shape[1:])
    return x2


def _twig_stage_part_name(part_template: str, stage_idx: int) -> str:
    # 支持 {i}/{idx}/{stage} 占位（与 TwiG.py 一致）
    return str(part_template).format(i=stage_idx + 1, idx=stage_idx, stage=stage_idx + 1)


def _twig_truncate_ids(ids: List[int], max_tokens: int) -> List[int]:
    if int(max_tokens) <= 0:
        return []
    return ids[: int(max_tokens)]


def _twig_find_subsequence(haystack: List[int], needle: List[int]) -> int:
    if not needle:
        return -1
    n = len(needle)
    for j in range(0, len(haystack) - n + 1):
        if haystack[j : j + n] == needle:
            return j
    return -1


@torch.no_grad()
def _twig_vec_to_token_ids(model: MultiModalityCausalLM, vec: torch.Tensor, k: int) -> List[int]:
    """
    与 TwiG.py 的 _vec_to_token_ids 一致：把 [D] 向量投影到 LM embedding 空间，取 top-k 最相似 token ids。
    仅用于“understanding tokens -> translator -> token ids”的启发式映射（不可导）。
    """
    if int(k) <= 0:
        return []
    emb_w = model.language_model.get_input_embeddings().weight.detach()  # [V,D]
    v = F.normalize(vec.to(emb_w.dtype), dim=-1)  # [D]
    w = F.normalize(emb_w, dim=-1)  # [V,D]
    sims = torch.matmul(w, v)  # [V]
    topk = torch.topk(sims, k=min(int(k), int(sims.numel())), dim=0).indices
    return topk.to(torch.long).tolist()


@torch.no_grad()
def _twig_understanding_prompt_ids(
    model: MultiModalityCausalLM,
    processor: VLChatProcessor,
    tokenizer,
    base_prompt: str,
    stages: int,
    stage_idx: int,
    pos: str,
    stage_images: Optional[List[Image.Image]],
    understanding_max_tokens: int,
    max_new_tokens: int = 300,
) -> List[int]:
    """
    与 TwiG.py 的“understanding”一致：让 LM 输出一个“可直接用于生成的优化 prompt”，但我们不 decode 成文本，
    而是直接取新生成的 token ids（truncate 到 understanding_max_tokens）。
    """
    conversation = [
        {
            "role": "User",
            "content": (
                f"You are a professional artist. We are drawing << {base_prompt} >> in {stages} stages; "
                f"we have finished {stage_idx} stages.\n"
                f"Task: write an optimized, generation-ready prompt for Stage {stage_idx + 1} focusing on the {pos}.\n"
                f"Rules: output ONLY the optimized prompt. No explanation, no bullet points."
            ),
        },
        {"role": "Assistant", "content": ""},
    ]

    if stage_images:
        prepare_inputs = processor(
            conversations=conversation, images=stage_images, force_batchify=True
        ).to(model.device)
        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs).to(torch.float16)
        attention_mask = prepare_inputs.attention_mask
        prompt_len = int(prepare_inputs.input_ids.shape[1]) if getattr(prepare_inputs, "input_ids", None) is not None else 0
    else:
        input_ids_text = torch.LongTensor(tokenizer.encode(conversation[0]["content"])).unsqueeze(0).to(model.device)
        inputs_embeds = model.language_model.get_input_embeddings()(input_ids_text).to(torch.float16)
        attention_mask = None
        prompt_len = int(input_ids_text.shape[1])

    outputs_text = model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=int(max_new_tokens),
        do_sample=True,
    )
    understanding_ids = outputs_text[0][prompt_len:].detach().cpu().tolist()
    return _twig_truncate_ids(understanding_ids, understanding_max_tokens)


def _twig_build_gen_prefix_ids(processor: VLChatProcessor, tokenizer, gen_prompt_ids: List[int]) -> List[int]:
    """
    与 TwiG.py 一致：用 placeholder 找到 SFT template 的插入点，把“gen_prompt_ids”插进去，
    并在末尾追加 image_start_tag。
    """
    placeholder = "<<<PROMPT_PLACEHOLDER>>>"
    gen_conv = [
        {"role": "<|User|>", "content": placeholder},
        {"role": "<|Assistant|>", "content": ""},
    ]
    sft_with_ph = processor.apply_sft_template_for_multi_turn_prompts(
        conversations=gen_conv,
        sft_format=processor.sft_format,
        system_prompt="",
    )
    sft_ids = tokenizer.encode(sft_with_ph)
    ph_ids = tokenizer.encode(placeholder)
    k = _twig_find_subsequence(sft_ids, ph_ids)
    if k < 0:
        prefix_ids = sft_ids
        suffix_ids: List[int] = []
    else:
        prefix_ids = sft_ids[:k]
        suffix_ids = sft_ids[k + len(ph_ids) :]
    img_tag_ids = tokenizer.encode(processor.image_start_tag)
    return prefix_ids + list(gen_prompt_ids) + suffix_ids + img_tag_ids


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


class VGGPerceptualLoss(torch.nn.Module):
    """
    VGG-based perceptual loss (frozen VGG16 features).
    使用 relu2_2 / relu3_3 / relu4_3 三层特征做 L2(MSE)。
    """

    # VGG16 features 索引：
    # relu2_2 -> 8, relu3_3 -> 15, relu4_3 -> 22
    _LAYER_IDXS = (8, 15, 22)

    def __init__(self, device: torch.device):
        super().__init__()
        self.device = device
        try:
            vgg = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT).features
        except Exception:
            vgg = tv_models.vgg16(pretrained=True).features  # 兼容旧 torchvision
        self.vgg = vgg.to(device).eval()
        for p in self.vgg.parameters():
            p.requires_grad_(False)

        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        # 输入 pred/gt 是 [-1,1]，VGG 期望 [0,1] + ImageNet normalize
        x01 = (x + 1.0) * 0.5
        x01 = x01.clamp(0.0, 1.0)
        return (x01 - self.mean) / self.std

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred_n = self._norm(pred)
        tgt_n = self._norm(target)

        loss = pred.new_zeros(())
        x_p, x_t = pred_n, tgt_n
        max_idx = max(self._LAYER_IDXS)
        for i, layer in enumerate(self.vgg):
            x_p = layer(x_p)
            x_t = layer(x_t)
            if i in self._LAYER_IDXS:
                loss = loss + F.mse_loss(x_p, x_t)
            if i >= max_idx:
                break
        return loss


def _kv_detach(past_key_values):
    if past_key_values is None:
        return None
    return tuple(tuple(x.detach() if torch.is_tensor(x) else x for x in layer) for layer in past_key_values)


def _build_cfg_prompt_embeds(
    model: MultiModalityCausalLM,
    processor: VLChatProcessor,
    tokenizer,
    gen_prompt_ids: List[int],
    bsz: int,
    device: torch.device,
):
    ids = _twig_build_gen_prefix_ids(processor, tokenizer, gen_prompt_ids)
    input_ids = torch.tensor(ids, device=device, dtype=torch.long)
    tokens = input_ids.unsqueeze(0).expand(bsz * 2, -1).clone()
    if tokens.shape[1] > 2:
        tokens[1::2, 1:-1] = processor.pad_id
    prompt_embeds = model.language_model.get_input_embeddings()(tokens).to(torch.float16)  # [2B,T,D]
    prompt_vec = prompt_embeds[0::2].mean(dim=1).to(prompt_embeds.dtype)  # [B,D]
    return prompt_embeds, prompt_vec


def _decode_prev_stage_image(
    model: MultiModalityCausalLM,
    soft_quant_all: List[torch.Tensor],
    stage_idx: int,
    img_size: int,
    patch_size: int,
    bsz: int,
) -> Optional[List[Image.Image]]:
    # TwiG.py 固定按 1/3,2/3,full 增长（写死 3）
    full_side = img_size // patch_size
    h = (full_side // 3) * stage_idx
    w = full_side
    if h <= 0:
        return None
    if len(soft_quant_all) < h * w:
        return None

    z_prev = torch.stack(soft_quant_all[: h * w], dim=1)  # [B,hw,8]
    z_grid = z_prev.view(bsz, h, w, 8).permute(0, 3, 1, 2).contiguous()  # [B,8,h,w]
    dec = model.gen_vision_model.decode(z_grid).float().clamp(-1, 1)  # [B,3,ph,pw]
    arr = ((dec[0].permute(1, 2, 0).cpu().numpy() + 1) / 2 * 255.0).clip(0, 255).astype("uint8")
    return [Image.fromarray(arr)]


def _run_twig_soft_generate(
    *,
    model: MultiModalityCausalLM,
    processor: VLChatProcessor,
    tokenizer,
    controller: LatentController,
    base_prompt: str,
    bsz: int,
    device: torch.device,
    codebook_weight: torch.Tensor,
    img_size: int,
    patch_size: int,
    image_token_num: int,
    stages: int,
    part_template: str,
    use_understanding: bool,
    understanding_max_tokens: int,
    cfg_weight: float,
    temperature: float,
    topk_decode: int,
    tbptt_window: int,
) -> tuple[List[torch.Tensor], int]:

    stages = max(1, int(stages))
    per = image_token_num // stages
    rem = image_token_num - per * stages
    stage_token_counts = [per + (1 if i < rem else 0) for i in range(stages)]
    per_stage_window = int(max(1, min(tbptt_window, max(stage_token_counts))))

    soft_quant_all: List[torch.Tensor] = []
    prev_stage_img_embeds_cfg: Optional[torch.Tensor] = None  # [2B,L,D]

    for stage_idx in range(stages):
        n_tokens = int(stage_token_counts[stage_idx])
        if n_tokens <= 0:
            continue
        controller.reset(batch_size=bsz, device=device)

        stage_images = None
        if prev_stage_img_embeds_cfg is not None:
            with torch.no_grad(), _autocast_ctx(device):
                stage_images = _decode_prev_stage_image(
                    model=model,
                    soft_quant_all=soft_quant_all,
                    stage_idx=stage_idx,
                    img_size=img_size,
                    patch_size=patch_size,
                    bsz=bsz,
                )

        optimized_prompt_ids: List[int] = []
        if use_understanding:
            pos = _twig_stage_part_name(part_template, stage_idx)
            optimized_prompt_ids = _twig_understanding_prompt_ids(
                model=model,
                processor=processor,
                tokenizer=tokenizer,
                base_prompt=base_prompt,
                stages=stages,
                stage_idx=stage_idx,
                pos=pos,
                stage_images=stage_images,
                understanding_max_tokens=understanding_max_tokens,
            )

        understanding_head = optimized_prompt_ids
        prompt_embeds, prompt_vec = _build_cfg_prompt_embeds(
            model, processor, tokenizer, optimized_prompt_ids, bsz, device
        )
        think_prompt_text = f"Given [{understanding_head}], we have already generated "

        if understanding_head:
            with torch.no_grad():
                uh_ids = torch.tensor(understanding_head, device=device, dtype=torch.long).unsqueeze(0)
                uh_emb = model.language_model.get_input_embeddings()(uh_ids)
                uh_out = model.language_model.model(inputs_embeds=uh_emb.to(torch.float16), use_cache=False)
                uh_vec = uh_out.last_hidden_state.mean(dim=1)
            uh_vec = uh_vec.expand(bsz, -1).to(prompt_vec.dtype)
            m_zeros = torch.zeros_like(prompt_vec)
            # 保证 dtype 一致（避免 LayerNorm 的 Half/Float 不匹配）
            with _autocast_ctx(device):
                translated_vec = controller.translator(z_vec=uh_vec, m_vec=m_zeros, p_vec=prompt_vec)
            translated_ids = _twig_vec_to_token_ids(model, translated_vec.mean(dim=0), k=len(understanding_head))
            think_prompt_text = f"Given [{translated_ids}], we have already generated "
            prompt_embeds, prompt_vec = _build_cfg_prompt_embeds(
                model, processor, tokenizer, translated_ids, bsz, device
            )

        prefix_embeds = (
            torch.cat([prompt_embeds, prev_stage_img_embeds_cfg.detach()], dim=1)
            if prev_stage_img_embeds_cfg is not None
            else prompt_embeds
        )

        window = int(max(1, min(per_stage_window, n_tokens)))
        warm_steps = int(max(0, n_tokens - window))

        past_key_values = None
        inputs_embeds = prefix_embeds
        stage_token_ids: List[int] = []
        stage_img_embeds_cfg_steps: List[torch.Tensor] = []

        def _one_step(probs: torch.Tensor) -> torch.Tensor:
            z = _soft_codebook_quant(probs, codebook_weight=codebook_weight, topk=topk_decode)  # [B,8]
            soft_gen = torch.matmul(probs, model.gen_embed.weight)  # [B,n_embed]
            img_embed = model.gen_aligner(soft_gen)  # [B,D]
            img_embed_cfg = _expand_cfg_batch(img_embed)  # [2B,D]
            stage_img_embeds_cfg_steps.append(img_embed_cfg.detach().unsqueeze(1))  # [2B,1,D]
            return z, img_embed_cfg

        # warmup（不建图）
        if warm_steps > 0:
            with torch.no_grad(), _autocast_ctx(device):
                for i in range(warm_steps):
                    out = model.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values)
                    past_key_values = out.past_key_values
                    h_last = out.last_hidden_state[:, -1, :]
                    logits2 = model.gen_head(h_last)
                    cond, uncond = logits2[0::2], logits2[1::2]
                    logits = uncond + float(cfg_weight) * (cond - uncond)
                    probs = torch.softmax(logits / float(temperature), dim=-1)

                    z, img_embed_cfg = _one_step(probs)
                    soft_quant_all.append(z.detach())
                    stage_token_ids.append(int(torch.argmax(probs[0]).detach().cpu().item()))
                    step_think = think_prompt_text + f"[b0:{stage_token_ids}]. Please ONLY output the optimized prompt."
                    past_key_values, _ = controller.maybe_inject(
                        model=model,
                        past_key_values=past_key_values,
                        step_idx=i,
                        prompt_vec=prompt_vec,
                        h_img_last_cond=h_last[0::2],
                        next_token_probs=probs,
                        prompt_text_for_think=step_think,
                    )
                    inputs_embeds = img_embed_cfg.unsqueeze(1)

        past_key_values = _kv_detach(past_key_values)

        # train window（建图）
        with _autocast_ctx(device):
            for i in range(warm_steps, n_tokens):
                out = model.language_model.model(inputs_embeds=inputs_embeds, use_cache=True, past_key_values=past_key_values)
                past_key_values = out.past_key_values
                h_last = out.last_hidden_state[:, -1, :]
                logits2 = model.gen_head(h_last)
                cond, uncond = logits2[0::2], logits2[1::2]
                logits = uncond + float(cfg_weight) * (cond - uncond)
                probs = torch.softmax(logits / float(temperature), dim=-1)

                z, img_embed_cfg = _one_step(probs)
                soft_quant_all.append(z)
                stage_token_ids.append(int(torch.argmax(probs[0]).detach().cpu().item()))
                step_think = think_prompt_text + f"[b0:{stage_token_ids}]. Please ONLY output the optimized prompt."
                past_key_values, _ = controller.maybe_inject(
                    model=model,
                    past_key_values=past_key_values,
                    step_idx=int(i - warm_steps),
                    prompt_vec=prompt_vec,
                    h_img_last_cond=h_last[0::2],
                    next_token_probs=probs.detach(),
                    prompt_text_for_think=step_think,
                )
                inputs_embeds = img_embed_cfg.unsqueeze(1)

        prev_stage_img_embeds_cfg = torch.cat(stage_img_embeds_cfg_steps, dim=1) if stage_img_embeds_cfg_steps else None

    return soft_quant_all, per_stage_window


def _resolve_train_files(cfg: dict) -> List[str]:
    """
    从 cfg['train_data'] 里按模板/范围拼接文件列表。
    """
    td = cfg.get("train_data", None)
    if not isinstance(td, dict):
        raise ValueError("twig_config.json 中缺少 train_data 配置。")

    root_dir = str(td.get("root_dir", "")).strip()
    sources = td.get("sources", None)
    if not root_dir or not isinstance(sources, list) or not sources:
        raise ValueError("twig_config.json 的 train_data 必须包含 root_dir 和 sources 列表。")

    out: List[str] = []
    missing: List[str] = []
    for s in sources:
        if not isinstance(s, dict):
            continue
        subdir = str(s.get("subdir", "")).strip()
        template = str(s.get("template", "")).strip()
        start = int(s.get("start", 0))
        end = int(s.get("end", -1))
        if not subdir or not template or end < start:
            continue
        for idx in range(start, end + 1):
            p = os.path.join(root_dir, subdir, template.format(idx=idx))
            if os.path.exists(p):
                out.append(p)
            else:
                missing.append(p)

    if not out:
        raise FileNotFoundError("train_data 没有解析出任何 parquet 文件（请检查 root_dir/模板/范围）。")
    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(f"train_data 中有文件不存在（前20个）：\n{preview}\n... total missing={len(missing)}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_key", type=str, default="image")
    ap.add_argument("--caption_key", type=str, default="caption_composition")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--max_steps", type=int, default=2000)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="为空则默认写到 repo_root/checkpoints_control_image_loss（避免 /root 路径问题）",
    )
    ap.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="设备选择：默认 cuda；无 GPU 时自动回退到 cpu",
    )
    ap.add_argument(
        "--visible_gpus",
        type=str,
        default="",
        help="可选：在脚本内部设置 CUDA_VISIBLE_DEVICES（例如 '0,1,2,3'）。建议直接在命令行环境变量里设置。",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tbptt_window", type=int, default=64, help="只对最后 W 个生成步反传，降低显存")

    # loss knobs
    ap.add_argument("--l1_weight", type=float, default=1.0, help="lambda_pix：像素级 L1 权重")
    ap.add_argument("--mse_weight", type=float, default=0.1, help="lambda_perc：感知损失(Perceptual)权重")

    args = ap.parse_args()

    # 压掉第三方库的 FutureWarning（避免训练日志刷屏）
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", message="Special tokens have been added*")
    try:
        from transformers.utils import logging as hf_logging  # type: ignore

        hf_logging.set_verbosity_error()
    except Exception:
        pass

    # 允许通过参数限制可见 GPU（满足“只用 0-3 号卡，不要多用”）
    # 注意：最好在启动前设置环境变量 CUDA_VISIBLE_DEVICES，这里仅做兜底。
    if args.visible_gpus:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.visible_gpus)

    # 单卡：直接选 device（默认 cuda）
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 默认输出目录：repo_root/checkpoints_control_image_loss
    repo_root = os.path.abspath(os.path.join(_THIS_DIR, ".."))
    out_dir = args.out_dir.strip()
    if not out_dir:
        out_dir = os.path.join(repo_root, "checkpoints_control_image_loss")
    os.makedirs(out_dir, exist_ok=True)

    # -----------------------------
    # 配置加载
    # -----------------------------
    _print("[setup] loading config...")
    default_cfg_path = os.path.join(os.path.dirname(__file__), "twig_config.json")
    cfg_path = resolve_config_path(default_cfg_path)
    cfg = load_json_config(cfg_path)

    train_files = _resolve_train_files(cfg)
    _print(f"[setup] train parquet files: {len(train_files)}")

    model_path = cfg.get("model_path", "deepseek-ai/Janus-Pro-7B")
    img_size = int(cfg.get("img_size", 384))
    patch_size = int(cfg.get("patch_size", 16))
    image_token_num = int(cfg.get("image_token_num", 576))
    stages = int(cfg.get("stages", 1))
    channels = int(cfg.get("channels", 8))
    part_template = str(cfg.get("part_template", "{i}-part"))
    stage_prompt_cfg = cfg.get("stage_prompt", {}) if isinstance(cfg.get("stage_prompt", {}), dict) else {}
    use_understanding_in_stage_text = bool(stage_prompt_cfg.get("use_understanding", True))
    understanding_max_tokens = int(stage_prompt_cfg.get("understanding_max_tokens", 128))
    generation_cfg = cfg.get("generation", {}) if isinstance(cfg.get("generation", {}), dict) else {}
    cfg_weight = float(generation_cfg.get("cfg_weight", 5.0))
    temperature = float(generation_cfg.get("temperature", 1.0))
    # 训练时 soft quant 的 top-k（不影响 TwiG.py 的 multinomial 采样，仅用于可导近似）
    topk_decode = int(generation_cfg.get("topk_decode", 32))

    # 期望 token_num == (img_size/patch)^2
    side = img_size // patch_size
    expected = side * side
    if expected != image_token_num:
        raise ValueError(f"image_token_num={image_token_num} must equal (img_size/patch)^2={expected}")
    # -----------------------------
    # 模型加载（冻结 Janus-Pro）
    # -----------------------------
    _print(f"[setup] model_path = {model_path}")
    _print("[setup] loading VLChatProcessor...")
    processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = processor.tokenizer

    _print("[setup] loading model weights...")
    model: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    model = model.to(device=device, dtype=torch.float16)
    model.eval()
    _freeze_all_params(model)
    _print("[setup] model loaded + frozen.")

    # -----------------------------
    # Control 模块（可训练）
    # -----------------------------
    _print("[setup] building LatentController (trainable control module)...")
    latent_cfg = build_latent_controller_config(cfg)
    latent_cfg.enabled = True
    # 训练时：保证 trigger.check_every 不超过 TBPTT window，否则 loss 可能不依赖 controller -> 无梯度
    tbptt_window = int(max(1, min(args.tbptt_window, image_token_num)))
    orig_check_every = int(getattr(latent_cfg.trigger, "check_every", 1))
    upper = tbptt_window - 1 if tbptt_window > 1 else 1
    eff_check_every = int(max(1, min(orig_check_every, upper)))
    if eff_check_every != orig_check_every:
        _print(
            f"[WARN] latent_cfg.trigger.check_every={orig_check_every} too large for tbptt_window={tbptt_window}; "
            f"clamped to {eff_check_every} to keep loss differentiable."
        )
    latent_cfg.trigger.check_every = eff_check_every
    latent_cfg.max_triggers_per_image = max(int(image_token_num), int(latent_cfg.max_triggers_per_image))
    d_model = int(model.language_model.get_input_embeddings().weight.shape[1])
    controller = LatentController(d_model=d_model, tokenizer=tokenizer, cfg=latent_cfg).to(device)
    # 训练：把 control 模块升到 FP32，避免 GradScaler 报 “unscale FP16 gradients”
    controller = controller.to(dtype=torch.float32)
    controller.train()
    _set_trainable_control(controller)
    _print("[setup] controller ready.")

    # -----------------------------
    # Perceptual loss (frozen VGG)
    # -----------------------------
    perceptual_loss_fn = VGGPerceptualLoss(device=device)

    # -----------------------------
    # 数据
    # -----------------------------
    _print(f"[setup] building dataloader from {len(train_files)} files")
    tfm = transforms.Compose(
        [
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),  # [0,1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1,1]
        ]
    )

    ds_iter = ParquetImageCaptionIterable(
        parquet_files=train_files,
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
    _print("[setup] dataloader ready. start training...")

    # -----------------------------
    # Optimizer / AMP
    # -----------------------------
    opt = torch.optim.AdamW(controller.parameters(), lr=args.lr, weight_decay=args.weight_decay, foreach=False)
    # 新接口：torch.amp.GradScaler('cuda')，避免 FutureWarning
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # codebook weight for soft decode: [V, 8]
    # 注意：VQ quantizer 可能使用 l2_norm，weight 用其内部逻辑的 embedding.weight 即可
    codebook_weight = model.gen_vision_model.quantize.embedding.weight  # [V,8]

    # -----------------------------
    # Train loop
    # -----------------------------
    step = 0
    t0 = time.time()
    pbar = tqdm(total=args.max_steps, desc="train", dynamic_ncols=True, leave=True)

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

        base_prompt = str(caps[0])
        soft_quant_all, per_stage_window = _run_twig_soft_generate(
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            controller=controller,
            base_prompt=base_prompt,
            bsz=bsz,
            device=device,
            codebook_weight=codebook_weight,
            img_size=img_size,
            patch_size=patch_size,
            image_token_num=image_token_num,
            stages=stages,
            part_template=part_template,
            use_understanding=use_understanding_in_stage_text,
            understanding_max_tokens=understanding_max_tokens,
            cfg_weight=cfg_weight,
            temperature=temperature,
            topk_decode=topk_decode,
            tbptt_window=tbptt_window,
        )

        with _autocast_ctx(device):
            z_all = torch.stack(soft_quant_all, dim=1)  # [B,T,8]
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

            pix = F.l1_loss(pred, gt_for_loss)
            perc = perceptual_loss_fn(pred, gt_for_loss)
            loss = float(args.l1_weight) * pix + float(args.mse_weight) * perc

        # 每个 step 单独打印一行 loss（DDP/torchrun 时只 rank0 打印）
        if _is_main_process():
            tqdm.write(
                f"[step {step+1}/{args.max_steps}] "
                f"loss={float(loss.detach().cpu()):.6f} "
                f"pix={float(pix.detach().cpu()):.6f} "
                f"perc={float(perc.detach().cpu()):.6f}"
            )

        if not loss.requires_grad:
            raise RuntimeError(
                "loss.requires_grad=False; controller injection likely never happened in the TBPTT window. "
                f"Try increasing --tbptt_window (currently {tbptt_window}) or decreasing latent_control.trigger.check_every in twig_config.json."
            )

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()

        step += 1
        pbar.update(1)

        # 每一步都更新/输出 loss（tqdm 的 postfix）
        dt = time.time() - t0
        pbar.set_postfix(
            loss=float(loss.detach().cpu()),
            pix=float(pix.detach().cpu()),
            perc=float(perc.detach().cpu()),
            sps=step / max(1e-6, dt),
        )

        if step % args.save_every == 0 or step == args.max_steps:
            ctrl_state = controller.state_dict()
            ckpt = {
                "step": step,
                "cfg_path": cfg_path,
                "latent_cfg": asdict(latent_cfg),
                "controller": ctrl_state,
                "opt": opt.state_dict(),
                "args": vars(args),
            }
            out = os.path.join(out_dir, f"ckpt_step{step}.pt")
            torch.save(ckpt, out)

    pbar.close()


if __name__ == "__main__":
    main()


