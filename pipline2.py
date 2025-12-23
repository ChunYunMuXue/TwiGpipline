import os
import PIL.Image
import torch
import numpy as np
import logging
from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
from janus.utils.io import load_pil_images

logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)

model_path = "deepseek-ai/Janus-Pro-7B"
vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
tokenizer = vl_chat_processor.tokenizer

vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True
)
vl_gpt = vl_gpt.to(torch.float16).cuda().eval()

base_prompt = "some universal students are sitting and studying in the bright library"
positions = ["top part", "middle part", "bottom part","null"]

parallel_size = 4
stages = 3
image_token_num = 576
img_size = 384
patch_size = 16
channels = 8
os.makedirs("generated_samples", exist_ok=True)


@torch.inference_mode()
def generate_multi_stage_with_image_feedback(
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
    prev_stage_tokens = None
    per_stage_tokens = image_token_num // stages
    all_tokens = []

    for stage_idx, pos in enumerate(positions):
        if(stage_idx == 3):
            break 
        print(f"\n[Stage {stage_idx+1}/{stages}] Generating {per_stage_tokens} tokens for '{pos}'")
        stage_text = f"{base_prompt}, {pos}"
        if prev_stage_tokens is not None:
            stage_text = f"[Previous stage considered] | {stage_text}"
        sft_format = processor.apply_sft_template_for_multi_turn_prompts(
            conversations=[{"role": "<|User|>", "content": stage_text},
                           {"role": "<|Assistant|>", "content": ""}],
            sft_format=processor.sft_format,
            system_prompt=""
        )
        prompt = sft_format + processor.image_start_tag
        input_ids = torch.LongTensor(tokenizer.encode(prompt)).cuda()
        tokens_emb = torch.zeros((parallel_size*2, len(input_ids)), dtype=torch.long).cuda()
        for i in range(parallel_size*2):
            tokens_emb[i] = input_ids
            if i % 2 != 0:
                tokens_emb[i, 1:-1] = processor.pad_id
        prompt_embeds = model.language_model.get_input_embeddings()(tokens_emb)

        # -----------------------------------------------------------
        ## embedding last token
        # -----------------------------------------------------------

        if prev_stage_tokens is not None:
            prev_embeds = model.prepare_gen_img_embeds(prev_stage_tokens.view(-1))
            prev_embeds_exp = prev_embeds.unsqueeze(0).expand(prompt_embeds.size(0), -1, -1)
            inputs_embeds = torch.cat([prompt_embeds, prev_embeds_exp], dim=1)
        else:
            inputs_embeds = prompt_embeds

        # -----------------------------------------------------------
        ## gen picture tokens
        # -----------------------------------------------------------
            
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

            next_token_exp = torch.cat([next_token.unsqueeze(1), next_token.unsqueeze(1)], dim=1).view(-1)
            img_embeds = model.prepare_gen_img_embeds(next_token_exp)
            inputs_embeds = img_embeds.unsqueeze(1)

        prev_stage_tokens = stage_tokens
        all_tokens.append(stage_tokens)

        # -----------------------------------------------------------
        ## decode,save picture
        # -----------------------------------------------------------

        dec_stage = model.gen_vision_model.decode_code(
            torch.cat(all_tokens, dim=1),
            shape=[parallel_size, channels,(img_size // patch_size) // 3 * (stage_idx + 1), (img_size // patch_size)]
        )
        dec_stage = dec_stage.float().cpu().numpy().transpose(0, 2, 3, 1)
        dec_stage = np.clip((dec_stage + 1) / 2 * 255, 0, 255).astype(np.uint8)

        stage_image_paths = []
        for i in range(parallel_size):
            path = f"generated_samples/stage{stage_idx+1}_img_{i}.jpg"
            PIL.Image.fromarray(dec_stage[i]).save(path)
            stage_image_paths.append(path)
        print(f"[Stage {stage_idx+1}] Saved {parallel_size} images.")

        # -----------------------------------------------------------
        ## !! understanding the picture
        # -----------------------------------------------------------

        # conversation = [
        #     {"role": "User", "content": f"This is stage {stage_idx+1}. Provide suggestions to improve the {pos} part of the image.", 
        #      "images": stage_image_paths},
        #     {"role": "Assistant", "content": ""}
        # ]
        conversation = [
            {
                "role": "User",
                "content": (
                    f"we have done the stage {stage_idx+1}, focus on the {pos}. "
                    f"Give concise guidance for the next stage {stage_idx + 2} to improve colors, shapes, and details of {positions[stage_idx + 1]}."
                    f"please,use the brief lanuage."
                ),
                "images": stage_image_paths
            },
            {"role": "Assistant", "content": ""}
        ]

        print("##",stage_idx," : ",conversation)

        pil_images = load_pil_images(conversation)
        prepare_inputs = processor(conversations=conversation, images=pil_images, force_batchify=True).to(model.device)
        inputs_embeds = model.prepare_inputs_embeds(**prepare_inputs)

        outputs_text = model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=200,
            do_sample=True
        )
        understanding_text = tokenizer.decode(outputs_text[0].cpu().tolist(), skip_special_tokens=True)
        print(f"[Stage {stage_idx+1} understanding]: {understanding_text}\n")

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

generate_multi_stage_with_image_feedback(
    vl_gpt,
    vl_chat_processor,
    base_prompt,
    positions,
    parallel_size=1,
    stages=3,
    image_token_num=576,
)
