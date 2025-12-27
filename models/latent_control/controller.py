from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from .condenser import AttentionCondenser, CondenserConfig
from .trigger import TriggerConfig, TriggerState, cosine_sim, entropy_from_probs, should_trigger
from .translator import Translator, TranslatorConfig
from .shaper import ControlTokenShaper, ShaperConfig


@dataclass
class LatentControllerConfig:
    enabled: bool = True

    # buffer/window
    img_hidden_window: int = 32  # 保存最近多少步 image-token hidden states 给 Condenser 用

    # trigger
    trigger: TriggerConfig = field(default_factory=TriggerConfig)
    max_triggers_per_image: int = 3

    # condenser/translator/shaper configs
    condenser: CondenserConfig = field(default_factory=CondenserConfig)
    translator: TranslatorConfig = field(default_factory=TranslatorConfig)
    shaper: ShaperConfig = field(default_factory=ShaperConfig)

    # think context
    think_prompt_max_tokens: int = 96


class LatentController(nn.Module):
    """
    把 Condenser / Trigger / Translator / Shaper 串起来，并负责把 control tokens
    “插入”到 KV cache（无回滚：仅额外 forward 一次 control prefix）。
    """

    def __init__(self, d_model: int, tokenizer, cfg: LatentControllerConfig):
        super().__init__()
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.d_model = d_model

        self.condenser = AttentionCondenser(d_model, cfg.condenser)
        self.translator = Translator(d_model, cfg.translator)
        self.shaper = ControlTokenShaper(d_model, cfg.shaper)

        # 运行态 state（每张图/每个 stage reset）
        self._trigger_state: Optional[TriggerState] = None
        self._img_h_buf: Optional[torch.Tensor] = None  # [B, W, D]
        self._img_h_ptr: int = 0
        self._triggers_used: int = 0

    def reset(self, batch_size: int, device: torch.device):
        w = int(self.cfg.img_hidden_window)
        self._img_h_buf = torch.zeros((batch_size, w, self.d_model), device=device)
        self._img_h_ptr = 0
        self._trigger_state = TriggerState(batch_size, self.cfg.trigger.window, device=device)
        self._triggers_used = 0

    def _push_img_hidden(self, h_img_last: torch.Tensor) -> torch.Tensor:
        """
        h_img_last: [B, D]
        返回用于 Condenser 的序列: [B, S, D] (S<=W)
        """
        assert self._img_h_buf is not None
        b, w, d = self._img_h_buf.shape
        self._img_h_buf[:, self._img_h_ptr] = h_img_last
        self._img_h_ptr = (self._img_h_ptr + 1) % w

        # 以时间顺序展开 buffer（最近的在最后）
        # idx: [ptr, ptr+1, ..., w-1, 0, 1, ..., ptr-1]
        idx = torch.arange(w, device=h_img_last.device)
        idx = (idx + self._img_h_ptr) % w
        seq = self._img_h_buf[:, idx]
        return seq

    @torch.inference_mode()
    def _think_latent(
        self,
        model,  # MultiModalityCausalLM
        prompt_text: str,
        m_tokens: torch.Tensor,  # [B, M, D]
    ) -> torch.Tensor:
        """
        使用 language model 做一次短 forward，拿 pooled hidden 作为 z_vec。
        不输出可读 CoT，只保留 hidden。
        """
        # tokenize prompt summary
        ids = self.tokenizer.encode(prompt_text)
        if len(ids) > self.cfg.think_prompt_max_tokens:
            ids = ids[-self.cfg.think_prompt_max_tokens :]
        input_ids = torch.tensor(ids, device=m_tokens.device, dtype=torch.long).unsqueeze(0)
        # embed text -> [1, T, D] then expand to [B, T, D]
        text_emb = model.language_model.get_input_embeddings()(input_ids).expand(
            m_tokens.shape[0], -1, -1
        )
        inputs_embeds = torch.cat([text_emb, m_tokens], dim=1)  # [B, T+M, D]
        out = model.language_model.model(inputs_embeds=inputs_embeds, use_cache=False)
        h = out.last_hidden_state  # [B, T+M, D]
        z_vec = h.mean(dim=1)
        return z_vec

    @torch.inference_mode()
    def maybe_inject(
        self,
        model,  # MultiModalityCausalLM
        past_key_values,
        step_idx: int,
        prompt_vec: torch.Tensor,  # [B, D]
        h_img_last_cond: torch.Tensor,  # [B, D]
        next_token_probs: torch.Tensor,  # [B, V] (通常是 CFG 后的 probs)
        prompt_text_for_think: str,
    ):
        """
        在生成循环中调用。
        - 更新视觉记忆 (Condenser)
        - 计算 trigger
        - 触发则：think -> translate -> shape -> prefix injection (update KV)
        返回: (new_past_key_values, did_inject: bool)
        """
        if not self.cfg.enabled:
            return past_key_values, False

        if self._img_h_buf is None or self._trigger_state is None:
            self.reset(batch_size=h_img_last_cond.shape[0], device=h_img_last_cond.device)

        img_seq = self._push_img_hidden(h_img_last_cond)  # [B,W,D]

        # 预算 & 频率控制
        if self._triggers_used >= self.cfg.max_triggers_per_image:
            return past_key_values, False
        if (step_idx + 1) % self.cfg.trigger.check_every != 0:
            return past_key_values, False

        # Condenser
        m_tokens, m_vec = self.condenser(img_seq)

        # Trigger features
        s_t = cosine_sim(m_vec, prompt_vec)  # [B]
        u_t = entropy_from_probs(next_token_probs)  # [B]
        delta_s, var_s = self._trigger_state.update(s_t)
        trig = should_trigger(self.cfg.trigger, s_t=s_t, delta_s=delta_s, var_s=var_s, u_t=u_t)

        if not bool(trig.any()):
            return past_key_values, False

        # Think -> Translator -> Shaper
        z_vec = self._think_latent(model=model, prompt_text=prompt_text_for_think, m_tokens=m_tokens)
        c_vec = self.translator(z_vec=z_vec, m_vec=m_vec, p_vec=prompt_vec)  # [B,D]
        ctrl_tokens = self.shaper.make_control_tokens_for_cfg(c_vec)  # [2B,K,D]

        # Prefix injection: 额外 forward 一次，把 control tokens 写进 KV
        inj_out = model.language_model.model(
            inputs_embeds=ctrl_tokens,
            use_cache=True,
            past_key_values=past_key_values,
        )
        self._triggers_used += 1
        return inj_out.past_key_values, True


