import os
import PIL.Image
import torch
import numpy as np
import logging
from janus.models import MultiModalityCausalLM, VLChatProcessor
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images
# =========================
# 屏蔽 transformers 配置日志，避免 AlignerConfig 序列化报错
# =========================
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)

# =========================
# 模型路径
# =========================
model_path = "deepseek-ai/Janus-Pro-7B"

# =========================
# 加载处理器和模型
# =========================
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.float16).cuda().eval()

# =========================
# Prompt
# =========================
conversation = [
    {
        "role": "<|User|>",
        "content": "A stunning princess from Kabul in red and white traditional clothing, blue eyes, brown hair",
    },
    {"role": "<|Assistant|>", "content": ""},
]

sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(
    conversations=conversation,
    sft_format=vl_chat_processor.sft_format,
    system_prompt="",
)
prompt = sft_format + vl_chat_processor.image_start_tag

# =========================
# 三阶段递归生成函数
# =========================
@torch.inference_mode()
def generate_recursive(
    mmgpt: MultiModalityCausalLM,
    vl_chat_processor: VLChatProcessor,
    prompt: str,
    parallel_size: int = 8,
    stages: int = 3,
    image_token_num: int = 576,
    cfg_weight: float = 5.0,
    temperature: float = 1.0,
    img_size: int = 384,
    patch_size: int = 16,
):
    # ---- prompt embedding ----
    input_ids = torch.LongTensor(vl_chat_processor.tokenizer.encode(prompt)).cuda()
    tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.long).cuda()
    for i in range(parallel_size*2):
        tokens[i] = input_ids
        if i % 2 != 0:
            tokens[i, 1:-1] = vl_chat_processor.pad_id
    prompt_embeds = mmgpt.language_model.get_input_embeddings()(tokens)

    per_stage = image_token_num // stages
    memory_embeds = []
    all_tokens = []

    inputs_embeds = prompt_embeds
    outputs = None

    for stage in range(stages):
        print(f"[Stage {stage+1}/{stages}] Generating {per_stage} tokens")
        stage_tokens = torch.zeros((parallel_size, per_stage), dtype=torch.long).cuda()
        stage_hidden = []

        for i in range(per_stage):
            outputs = mmgpt.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=outputs.past_key_values if outputs else None,
            )
            h = outputs.last_hidden_state[:, -1, :]
            stage_hidden.append(h)

            logits = mmgpt.gen_head(h)
            cond, uncond = logits[0::2], logits[1::2]
            logits = uncond + cfg_weight * (cond - uncond)
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            stage_tokens[:, i] = next_token.squeeze(-1)

            # update inputs_embeds for next token
            next_token_exp = torch.cat([next_token.unsqueeze(1), next_token.unsqueeze(1)], dim=1).view(-1)
            img_embeds = mmgpt.prepare_gen_img_embeds(next_token_exp)
            inputs_embeds = img_embeds.unsqueeze(1)

        all_tokens.append(stage_tokens)

        # ---- memory embedding for next stage ----
        stage_hidden = torch.stack(stage_hidden, dim=1)[0::2]  # only conditional
        u = stage_hidden.mean(dim=1, keepdim=True)
        u = torch.cat([u, u], dim=0)
        memory_embeds.append(u)

        # inject memory into next stage
        inputs_embeds = torch.cat([prompt_embeds] + memory_embeds, dim=1)
        outputs = None  # reset KV cache

    # ---- decode all tokens ----
    all_tokens = torch.cat(all_tokens, dim=1)
    dec = mmgpt.gen_vision_model.decode_code(
        all_tokens,
        shape=[parallel_size, 8, img_size // patch_size, img_size // patch_size]
    )
    dec = dec.float().cpu().numpy().transpose(0, 2, 3, 1)
    dec = np.clip((dec + 1) / 2 * 255, 0, 255).astype(np.uint8)

    os.makedirs("generated_samples", exist_ok=True)
    for i in range(parallel_size):
        PIL.Image.fromarray(dec[i]).save(f"generated_samples/img_{i}.jpg")

    print(f"✅ Finished generating {parallel_size} images.")

# =========================
# 运行
# =========================
generate_recursive(
    vl_gpt,
    vl_chat_processor,
    prompt,
    parallel_size=8,
    stages=3,
    image_token_num=576,
)
