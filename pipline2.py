import os
import PIL.Image
import torch
import numpy as np
import logging
from janus.models import MultiModalityCausalLM, VLChatProcessor
from transformers import AutoModelForCausalLM
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
model_path = "deepseek-ai/Janus-Pro-7B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer
vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.float16).cuda().eval()
base_prompt = "A stunning princess from Kabul in red and white traditional clothing, blue eyes, brown hair"
positions = ["top part", "middle part", "bottom part"]

@torch.inference_mode()
def generate_multi_stage_image_with_token_prompt(
    model: MultiModalityCausalLM,
    processor: VLChatProcessor,
    base_prompt: str,
    positions: list,
    parallel_size: int = 4,
    stages: int = 3,
    image_token_num: int = 576,
    cfg_weight: float = 5.0,
    temperature: float = 1.0,
    img_size: int = 384,
    patch_size: int = 16,
):
    per_stage_tokens = image_token_num // stages
    all_tokens = []
    prev_stage_tokens = None
    prev_stage_prompt = None
    os.makedirs("generated_samples", exist_ok=True)
    channels = 8  # VQ-VAE latent channels

    for stage_idx, pos in enumerate(positions):
        print(f"[Stage {stage_idx+1}/{stages}] Generating {per_stage_tokens} tokens for '{pos}'")

        # -------------------------------
        # 融合上一阶段 token embedding为占位符/描述
        # -------------------------------
        stage_text = f"{base_prompt}, {pos}"
        if prev_stage_tokens is not None:
            stage_text = f"[Previous stage tokens considered] | {stage_text}"

        # -------------------------------
        # 调用 SFT 模板处理
        # -------------------------------
        sft_format = processor.apply_sft_template_for_multi_turn_prompts(
            conversations=[{"role": "<|User|>", "content": stage_text},
                           {"role": "<|Assistant|>", "content": ""}],
            sft_format=processor.sft_format,
            system_prompt=""
        )
        prompt = sft_format + processor.image_start_tag

        # -------------------------------
        # 生成 prompt embedding
        # -------------------------------
        input_ids = torch.LongTensor(tokenizer.encode(prompt)).cuda()
        tokens = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.long).cuda()
        for i in range(parallel_size*2):
            tokens[i] = input_ids
            if i % 2 != 0:
                tokens[i, 1:-1] = processor.pad_id
        prompt_embeds = model.language_model.get_input_embeddings()(tokens)

        # -------------------------------
        # 拼接上一阶段 token embedding
        # -------------------------------
        if prev_stage_tokens is not None:
            prev_embeds = model.prepare_gen_img_embeds(prev_stage_tokens.view(-1))
            prev_embeds_exp = prev_embeds.unsqueeze(0).expand(prompt_embeds.size(0), -1, -1)
            inputs_embeds = torch.cat([prompt_embeds, prev_embeds_exp], dim=1)
        else:
            inputs_embeds = prompt_embeds

        # -------------------------------
        # 生成阶段 token
        # -------------------------------
        stage_tokens = torch.zeros((parallel_size, per_stage_tokens), dtype=torch.long).cuda()
        outputs = None

        for i in range(per_stage_tokens):
            outputs = model.language_model.model(
                inputs_embeds=inputs_embeds,
                use_cache=True,
                past_key_values=outputs.past_key_values if outputs else None,
            )
            h = outputs.last_hidden_state[:, -1, :]
            logits = model.gen_head(h)
            cond, uncond = logits[0::2], logits[1::2]
            logits = uncond + cfg_weight * (cond - uncond)
            probs = torch.softmax(logits / temperature, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            stage_tokens[:, i] = next_token.squeeze(-1)

            # 更新 inputs_embeds
            next_token_exp = torch.cat([next_token.unsqueeze(1), next_token.unsqueeze(1)], dim=1).view(-1)
            img_embeds = model.prepare_gen_img_embeds(next_token_exp)
            inputs_embeds = img_embeds.unsqueeze(1)

        # -------------------------------
        # 生成阶段理解提示（方法 A：embedding + attention_mask）
        # -------------------------------
        stage_goal = f"Generate the {pos} part of the full princess image"
        understand_text = f"This is stage {stage_idx+1}. The goal is: {stage_goal}. Provide suggestions for this stage."

        input_ids_text = torch.LongTensor(tokenizer.encode(understand_text)).unsqueeze(0).cuda()
        prompt_embeds_text = model.language_model.get_input_embeddings()(input_ids_text)

        if prev_stage_tokens is not None:
            prev_embeds_for_understanding = model.prepare_gen_img_embeds(prev_stage_tokens.view(-1))
            prev_embeds_for_understanding = prev_embeds_for_understanding.unsqueeze(0)
            inputs_embeds_understand = torch.cat([prompt_embeds_text, prev_embeds_for_understanding], dim=1)
        else:
            inputs_embeds_understand = prompt_embeds_text

        # -------------------------------
        # attention_mask
        # -------------------------------
        attention_mask = torch.ones(inputs_embeds_understand.size()[:-1], dtype=torch.long).cuda()

        # -------------------------------
        # 调用 generate
        # -------------------------------
        try:
            outputs_text = model.language_model.generate(
                inputs_embeds=inputs_embeds_understand,
                attention_mask=attention_mask,
                pad_token_id=tokenizer.eos_token_id,
                max_new_tokens=50,
                do_sample=True
            )
            understanding_text = tokenizer.decode(outputs_text[0], skip_special_tokens=True)
        except:
            understanding_text = "Unable to generate understanding suggestions."
        print(f"[Stage {stage_idx+1} understanding]: {understanding_text}\n")

        # -------------------------------
        # 保存阶段 token
        # -------------------------------
        all_tokens.append(stage_tokens)
        prev_stage_tokens = stage_tokens
        prev_stage_prompt = stage_text

    # -------------------------------
    # 最终 decode 全部 token
    # -------------------------------
    total_tokens = torch.cat(all_tokens, dim=1)
    full_patch_side = img_size // patch_size
    dec_final = model.gen_vision_model.decode_code(
        total_tokens,
        shape=[parallel_size, channels, full_patch_side, full_patch_side]
    )
    dec_final = dec_final.float().cpu().numpy().transpose(0, 2, 3, 1)
    dec_final = np.clip((dec_final + 1) / 2 * 255, 0, 255).astype(np.uint8)

    for i in range(parallel_size):
        PIL.Image.fromarray(dec_final[i]).save(f"generated_samples/final_img_{i}.jpg")

    print(f"✅ Finished generating {parallel_size} final images.")


# -------------------------------
# 运行
# -------------------------------
generate_multi_stage_image_with_token_prompt(
    vl_gpt,
    vl_chat_processor,
    base_prompt,
    positions,
    parallel_size=4,
    stages=3,
    image_token_num=576,
)
