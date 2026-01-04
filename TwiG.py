import os
import PIL.Image
import torch
import numpy as np
import logging
import torch.nn.functional as F
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
from models.latent_control.controller import LatentController, LatentControllerConfig
from config_io import build_latent_controller_config, load_json_config, resolve_config_path

logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)

# -----------------------------
# 配置加载（从 twig_config.json 读入，避免硬编码不生效）
# -----------------------------
_default_cfg_path = os.path.join(os.path.dirname(__file__), "twig_config.json")
_cfg_path = resolve_config_path(_default_cfg_path)
_cfg = load_json_config(_cfg_path)

# -----------------------------
# 模型与处理器加载
# -----------------------------
model_path = _cfg.get("model_path", "deepseek-ai/Janus-Pro-7B")
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.float16).cuda().eval()

# -----------------------------
# 基础配置
# -----------------------------
# base_prompt = "A cozy wooden cabin beside a calm lake at sunrise, snow-covered pines and mountains glowing warmly"
# base_prompt = "Capture a close-up shot of a vibrant sunflower in full bloom, with a honeybee perched on its petals, its delicate wings catching the sunlight."
base_prompt = _cfg.get(
    "base_prompt",
    "A cozy wooden cabin beside a calm lake at sunrise, snow-covered pines and mountains glowing warmly",
)
parallel_size = int(_cfg.get("parallel_size", 1))  # gen_number
stages = int(_cfg.get("stages", 3))  # stage_number
image_token_num = int(_cfg.get("image_token_num", 576))  # image_token_number
img_size = int(_cfg.get("img_size", 384))  # image_size
patch_size = int(_cfg.get("patch_size", 16))  # patch_size
channels = int(_cfg.get("channels", 8))  # channels
part_template = str(_cfg.get("part_template", "{i}-part"))  # e.g. "1-part", "2-part", ...
stage_prompt_cfg = _cfg.get("stage_prompt", {}) if isinstance(_cfg.get("stage_prompt", {}), dict) else {}
# 这里的 use_understanding 指的是：是否把“understanding 步骤输出的优化 prompt tokens”
# 直接用于 generation 前缀 + Translator 控制前缀（不再 decode 成文本）。
use_understanding_in_stage_text = bool(stage_prompt_cfg.get("use_understanding", True))
understanding_max_tokens = int(stage_prompt_cfg.get("understanding_max_tokens", 128))
generation_cfg = _cfg.get("generation", {}) if isinstance(_cfg.get("generation", {}), dict) else {}
cfg_weight = float(generation_cfg.get("cfg_weight", 5.0))
temperature = float(generation_cfg.get("temperature", 1.0))
os.makedirs("generated_samples", exist_ok=True)

# -----------------------------
# Latent control（跑题检测 -> think -> 安全注入）
# -----------------------------
latent_cfg = build_latent_controller_config(_cfg)
enable_latent_control = bool(latent_cfg.enabled)

# =============================
# 多阶段生成函数（先询问再生成）
# =============================
@torch.inference_mode()
def generate_multi_stage_with_image_feedback(
    model: MultiModalityCausalLM,
    processor: VLChatProcessor,
    base_prompt: str,
    parallel_size: int = 1,
    stages: int = 3,
    image_token_num: int = 576,
    cfg_weight: float = 5.0,
    temperature: float = 1.0,
    img_size: int = 384,
    patch_size: int = 16,
    channels: int = 8,
    part_template: str = "{i}-part",
):
    prev_stage_tokens = None
    per_stage_tokens = image_token_num // stages
    print("Per stage gen ",per_stage_tokens,"tokens");
    all_tokens = []

    def _stage_part_name(stage_idx: int) -> str:
        # 支持 {i}/{idx}/{stage} 占位
        return str(part_template).format(i=stage_idx + 1, idx=stage_idx, stage=stage_idx + 1)

    def _truncate_ids(ids, max_tokens: int):
        if max_tokens <= 0:
            return []
        return ids[:max_tokens]

    def _vec_to_token_ids(vec: torch.Tensor, k: int) -> list:
        """
        Map a single vector [D] to k nearest token ids in the LM embedding space (cosine similarity).
        Used here as a heuristic “Translator pass” over _understanding_head: ids -> latent -> Translator -> ids.
        """
        if k <= 0:
            return []
        emb_w = model.language_model.get_input_embeddings().weight.detach()  # [V,D]
        v = F.normalize(vec.to(emb_w.dtype), dim=-1)                         # [D]
        w = F.normalize(emb_w, dim=-1)                                      # [V,D]
        sims = torch.matmul(w, v)                                           # [V]
        topk = torch.topk(sims, k=min(k, sims.numel()), dim=0).indices
        return topk.to(torch.long).tolist()

    def _find_subsequence(haystack, needle):
        """Return start index of needle in haystack, or -1."""
        if not needle:
            return -1
        n = len(needle)
        for j in range(0, len(haystack) - n + 1):
            if haystack[j : j + n] == needle:
                return j
        return -1

    for stage_idx in range(stages):
        pos = _stage_part_name(stage_idx)

        print(f"\n[Stage {stage_idx+1}/{stages}] Processing '{pos}'")

        stage_image_paths = []
        if prev_stage_tokens is not None:
            dec_prev_stage = model.gen_vision_model.decode_code(
                torch.cat(all_tokens, dim=1),
                shape=[parallel_size, channels, (img_size // patch_size) // 3 * stage_idx, img_size // patch_size]
            )
            dec_prev_stage = dec_prev_stage.float().cpu().numpy().transpose(0, 2, 3, 1)
            dec_prev_stage = np.clip((dec_prev_stage + 1) / 2 * 255, 0, 255).astype(np.uint8)
            for i in range(parallel_size):
                path = f"generated_samples/stage{stage_idx}_img_{i}.jpg"
                PIL.Image.fromarray(dec_prev_stage[i]).save(path)
                stage_image_paths.append(path)

        # 构造短 prompt
        conversation = [{"role": "User",
                         # 注意：这里不再让模型“解释/给建议”，而是让它输出一个可直接用于生成的“优化 prompt”
                         # 后续我们不会 decode 成文本，而是直接用它的 token ids / hidden 去驱动 Translator -> control prefix
                         "content": f"You are a professional artist. We are drawing << {base_prompt} >> in {stages} stages; we have finished {stage_idx} stages.\n"
                                    f"Task: write an optimized, generation-ready prompt for Stage {stage_idx + 1} focusing on the {pos}.\n"
                                    f"Rules: output ONLY the optimized prompt. No explanation, no bullet points."
                            }
                        ]
        if stage_image_paths:
            conversation[0]["images"] = stage_image_paths
        conversation.append({"role": "Assistant", "content": ""})

        # print(f"[Stage {stage_idx + 1} prompt]:",conversation)

        if stage_image_paths:
            pil_images = load_pil_images(conversation)
            prepare_inputs = processor(conversations=conversation, images=pil_images, force_batchify=True).to(model.device)
            inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs).to(torch.float16)
            attention_mask = prepare_inputs.attention_mask
        else:
            input_ids_text = torch.LongTensor(tokenizer.encode(conversation[0]['content'])).unsqueeze(0).cuda()
            inputs_embeds = model.language_model.get_input_embeddings()(input_ids_text).to(torch.float16)
            attention_mask = None

        outputs_text = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=300,
            do_sample=True
        )
        # 只取“新生成”的部分 token ids，避免把 prompt 一起带进来
        if stage_image_paths and hasattr(prepare_inputs, "input_ids") and prepare_inputs.input_ids is not None:
            prompt_len = int(prepare_inputs.input_ids.shape[1])
        else:
            prompt_len = int(input_ids_text.shape[1])
        understanding_ids = outputs_text[0][prompt_len:].detach().cpu().tolist()
        # 这里不 decode 成文本；把它当作“优化 prompt tokens”
        optimized_prompt_ids = _truncate_ids(understanding_ids, understanding_max_tokens) if use_understanding_in_stage_text else []

        _understanding_head = optimized_prompt_ids if optimized_prompt_ids else []
        think_prompt_text = ""  # will be set after controller init (optionally via Translator)

        _placeholder = "<<<PROMPT_PLACEHOLDER>>>"
        _gen_conv = [
            {"role": "<|User|>", "content": _placeholder},
            {"role": "<|Assistant|>", "content": ""},
        ]
        _sft_with_ph = processor.apply_sft_template_for_multi_turn_prompts(
            conversations=_gen_conv,
            sft_format=processor.sft_format,
            system_prompt="",
        )
        _sft_ids = tokenizer.encode(_sft_with_ph)
        _ph_ids = tokenizer.encode(_placeholder)
        _k = _find_subsequence(_sft_ids, _ph_ids)
        if _k < 0:
            prefix_ids = _sft_ids
            suffix_ids = []
        else:
            prefix_ids = _sft_ids[:_k]
            suffix_ids = _sft_ids[_k + len(_ph_ids) :]

        img_tag_ids = tokenizer.encode(processor.image_start_tag)
        input_ids = torch.tensor(prefix_ids + optimized_prompt_ids + suffix_ids + img_tag_ids, device=model.device, dtype=torch.long)

        # import pdb; pdb.set_trace()

        tokens_emb = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.long, device=model.device)
        for i in range(parallel_size*2):
            tokens_emb[i] = input_ids
            if i % 2 != 0:
                tokens_emb[i, 1:-1] = processor.pad_id
        prompt_embeds = model.language_model.get_input_embeddings()(tokens_emb).to(torch.float16)
        # prompt pooled vector（仅用 conditional 分支）
        prompt_vec = prompt_embeds[0::2].mean(dim=1)  # [B, D]

        # controller：每个 stage 复位一次 state（防止跨 stage 污染）
        controller = None
        if enable_latent_control:
            d_model = prompt_embeds.shape[-1]
            controller = LatentController(d_model=d_model, tokenizer=tokenizer, cfg=latent_cfg).to(model.device)
            controller.reset(batch_size=parallel_size, device=model.device)

        # 对 _understanding_head 做一次 Translator（latent->latent），再映射回 token ids（用于 think prompt）
        # 同时：把 translated_ids 也用于“正常生成”的前缀 prompt（替换 optimized_prompt_ids）
        gen_prompt_ids = optimized_prompt_ids
        if controller is not None and _understanding_head:
            uh_ids = torch.tensor(_understanding_head, device=model.device, dtype=torch.long).unsqueeze(0)  # [1,T]
            uh_emb = model.language_model.get_input_embeddings()(uh_ids)  # [1,T,D]
            uh_out = model.language_model.model(inputs_embeds=uh_emb.to(torch.float16), use_cache=False)
            uh_vec = uh_out.last_hidden_state.mean(dim=1)  # [1,D]
            uh_vec = uh_vec.expand(parallel_size, -1).to(prompt_vec.dtype)  # [B,D]
            m_zeros = torch.zeros_like(prompt_vec)
            translated_vec = controller.translator(z_vec=uh_vec, m_vec=m_zeros, p_vec=prompt_vec)  # [B,D]
            translated_ids = _vec_to_token_ids(translated_vec.mean(dim=0), k=len(_understanding_head))
            think_prompt_text = f"Given [{translated_ids}], we have already generated "
            gen_prompt_ids = translated_ids
        else:
            think_prompt_text = f"Given [{_understanding_head}], we have already generated "

        # 如果有 translated ids，则用它重新构造 generation prefix / prompt embeddings（确保“正常生成”真的用上）
        if gen_prompt_ids is not optimized_prompt_ids:
            input_ids = torch.tensor(prefix_ids + gen_prompt_ids + suffix_ids + img_tag_ids, device=model.device, dtype=torch.long)
            tokens_emb = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.long, device=model.device)
            for i in range(parallel_size*2):
                tokens_emb[i] = input_ids
                if i % 2 != 0:
                    tokens_emb[i, 1:-1] = processor.pad_id
            prompt_embeds = model.language_model.get_input_embeddings()(tokens_emb)
            prompt_vec = prompt_embeds[0::2].mean(dim=1)  # [B, D]

        if prev_stage_tokens is not None:
            prev_embeds = model.prepare_gen_img_embeds(prev_stage_tokens.view(-1)).to(torch.float16)
            prev_embeds_exp = prev_embeds.unsqueeze(0).expand(prompt_embeds.size(0), -1, -1)
            inputs_embeds = torch.cat([prompt_embeds, prev_embeds_exp], dim=1)
        else:
            inputs_embeds = prompt_embeds

        # import pdb; pdb.set_trace()
        # ----------------------------
        # 生成 token
        # ----------------------------
        stage_tokens = torch.zeros((parallel_size, per_stage_tokens), dtype=torch.long).cuda()
        past_key_values = None
        for i in range(per_stage_tokens):
            # print("gen ",i)
            outputs = model.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=past_key_values,
            )
            past_key_values = outputs.past_key_values
            h = outputs.last_hidden_state[:, -1, :]
            logits = model.gen_head(h)
            cond, uncond = logits[0::2], logits[1::2]
            logits = uncond + cfg_weight * (cond - uncond)
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            stage_tokens[:, i] = next_token.squeeze(-1)

            if controller is not None:
                h_img_last_cond = h[0::2]  # [B,D]
                _tail_n = i + 1
                _tail = stage_tokens[:, : (i + 1)].detach().cpu().tolist()
                _tail_str = "; ".join([f"b{bi}:{toks}" for bi, toks in enumerate(_tail)])
                step_think_prompt_text = think_prompt_text + f"[{_tail_str}]. Please ONLY output the optimized prompt."
                past_key_values, did = controller.maybe_inject(
                    model=model,
                    past_key_values=past_key_values,
                    step_idx=i,
                    prompt_vec=prompt_vec,
                    h_img_last_cond=h_img_last_cond,
                    next_token_probs=probs,
                    prompt_text_for_think=step_think_prompt_text,
                )
                if did:
                    print(f"[LatentControl] injected at stage={stage_idx+1}, step={i+1}")

            next_token_exp = torch.cat([next_token.unsqueeze(1), next_token.unsqueeze(1)], dim=1).view(-1)
            img_embeds = model.prepare_gen_img_embeds(next_token_exp).to(torch.float16)
            inputs_embeds = img_embeds.unsqueeze(1)

        prev_stage_tokens = stage_tokens
        all_tokens.append(stage_tokens)

    total_tokens = torch.cat(all_tokens, dim=1)
    full_patch_side = img_size // patch_size
    dec_final = model.gen_vision_model.decode_code(
        total_tokens,
        shape=[parallel_size, channels, full_patch_side, full_patch_side]
    )
    dec_final = dec_final.float().cpu().numpy().transpose(0, 2, 3, 1)
    dec_final = np.clip((dec_final + 1) / 2 * 255, 0, 255).astype(np.uint8)

    for i in range(parallel_size):
        PIL.Image.fromarray(dec_final[i]).save(f"generated_samples/stage3_img_{i}.jpg")
    print(f"✅ Finished generating {parallel_size} final images.")

generate_multi_stage_with_image_feedback(
    vl_gpt,
    vl_chat_processor,
    base_prompt,
    parallel_size=parallel_size,
    stages=stages,
    image_token_num=image_token_num,
    cfg_weight=cfg_weight,
    temperature=temperature,
    img_size=img_size,
    patch_size=patch_size,
    channels=channels,
    part_template=part_template,
)
