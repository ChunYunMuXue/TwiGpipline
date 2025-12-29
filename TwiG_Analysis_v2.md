# TwiG Pipeline 运行逻辑分析 - 完整版

## 概述

本文档详细分析了两个脚本的运行逻辑：
- `t2i_inference.py`: 基于Janus-Pro的基础文生图脚本
- `TwiG.py`: 改进的多阶段文生图脚本，包含latent control机制

## 1. t2i_inference.py 运行逻辑

### 主要组件
1. **模型加载**: 加载Janus-Pro-7B模型和VLChatProcessor
2. **对话格式化**: 将用户输入转换为模型期望的对话格式
3. **CFG生成**: 使用Classifier-Free Guidance生成图像tokens
4. **图像解码**: 将tokens解码为实际图像

### 详细流程

```python
# 1. 模型初始化
model_path = "deepseek-ai/Janus-Pro-7B"
vl_chat_processor = VLChatProcessor.from_pretrained(model_path)
vl_gpt = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)

# 2. 对话构造
conversation = [
    {"role": "<|User|>", "content": "用户输入的文本描述"},
    {"role": "<|Assistant|>", "content": ""}
]

# 3. Prompt格式化
sft_format = vl_chat_processor.apply_sft_template_for_multi_turn_prompts(...)
prompt = sft_format + vl_chat_processor.image_start_tag

# 4. 生成函数调用
generate(vl_gpt, vl_chat_processor, prompt)
```

### generate() 函数逻辑

1. **Tokenization**: 将prompt编码为tokens
2. **CFG Setup**: 创建条件和无条件tokens对
3. **Token生成循环**:
   - 使用语言模型预测下一个token
   - 应用CFG权重调整logits
   - 从概率分布采样
4. **图像解码**: 将生成的576个tokens解码为384x384图像

## 2. TwiG.py 运行逻辑

### 核心创新
1. **多阶段生成**: 将图像生成分为3个阶段
2. **智能引导**: 每个阶段前使用语言模型进行"理解"任务
3. **Latent Control**: 动态注入控制tokens防止"跑题"
4. **渐进式构建**: 基于前一阶段的结果继续生成

### 主要组件
- **基础配置**: 3阶段生成，576个tokens，384x384图像
- **Latent Controller**: 防止生成偏离prompt的机制
- **多阶段生成函数**: `generate_multi_stage_with_image_feedback()`

### 详细流程

#### 初始化阶段
```python
# 模型加载 (使用float16精度)
vl_gpt = vl_gpt.to(torch.float16).cuda().eval()

# 配置参数
stages = 3
image_token_num = 576  # 每个阶段192个tokens
positions = ["top part", "middle part", "bottom part", "null"]
```

#### 多阶段生成循环

对于每个阶段 (stage_idx: 0, 1, 2):

1. **阶段理解** (Stage Understanding):
   ```python
   # 构造对话让模型理解当前任务
   conversation = [{
       "role": "User",
       "content": f"你是专业艺术家。我们正在分3阶段绘制 << {base_prompt} >> ..."
   }]
   ```

2. **语言模型推理**:
   ```python
   # 使用语言模型生成指导文本
   outputs_text = model.language_model.generate(
       inputs_embeds=inputs_embeds,
       max_new_tokens=300,
       do_sample=True
   )
   ```

3. **图像生成Prompt构造**:
   ```python
   # 基于理解结果构造生成prompt
   stage_text = f"{base_prompt}, {positions[stage_idx]}"
   ```

4. **CFG Token生成**:
   - 创建条件/无条件tokens对
   - 应用CFG权重
   - 生成当前阶段的tokens

5. **Latent Control注入**:
   ```python
   # 检查是否需要注入控制tokens
   past_key_values, did = controller.maybe_inject(...)
   ```

6. **Tokens累积**: 将当前阶段tokens添加到all_tokens列表

#### 最终解码
```python
# 将所有阶段的tokens拼接并解码为完整图像
total_tokens = torch.cat(all_tokens, dim=1)
dec_final = model.gen_vision_model.decode_code(
    total_tokens,
    shape=[parallel_size, 8, 24, 24]  # 384/16 = 24
)
```

## 3. Latent Control 机制详解

Latent Control 是 TwiG.py 的核心创新机制，用于防止图像生成过程中"跑题"（生成内容偏离原始prompt）。这是一个多组件的动态控制系统。

### 核心组件架构

```
┌─────────────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────────┐
│   Condenser     │ -> │   Trigger    │ -> │  Translator  │ -> │    Shaper   │
│  注意力凝聚器   │    │   触发器     │    │   翻译器     │    │   塑形器    │
└─────────────────┘    └──────────────┘    └──────────────┘    └─────────────┘
         ↓                     ↓                    ↓                    ↓
    视觉记忆提取           偏差检测            控制信号生成          Token注入
```

### 详细运行流程

#### 1. Condenser（注意力凝聚器）
**作用**：从图像生成过程中的hidden states提取视觉记忆
- **输入**：`h_img_seq` [B, S, D] - 最近32步image-token的hidden states
- **机制**：
  - 使用可学习参数 `latents` [M, D] 作为查询向量
  - 通过交叉注意力机制从图像序列中提取关键信息
  - 输出固定8个memory tokens [B, M, D] 和汇总向量 [B, D]
- **代码实现**：
  ```python
  # 可学习查询向量
  self.latents = nn.Parameter(torch.randn(cfg.memory_tokens, d_model) * 0.02)

  # 交叉注意力
  q = self.latents.unsqueeze(0).expand(b, -1, -1)
  m_tokens, _ = self.cross_attn(query=q, key=h_img_seq, value=h_img_seq, need_weights=False)

  # 残差连接和MLP
  m_tokens = m_tokens + self.mlp(m_tokens)
  m_vec = m_tokens.mean(dim=1)
  ```

#### 2. Trigger（触发器）
**作用**：检测生成过程是否偏离预期
- **触发条件**：`(Δs_t < -0.05 AND Var>0.002) OR (u_t>3.5 AND s_t<0.25)`
- **指标计算**：
  - `s_t`：当前生成与prompt的余弦相似度
  - `Δs_t`：相似度变化率（与前一步比较）
  - `Var(s)`：相似度在窗口内的方差
  - `u_t`：next token概率分布的熵（不确定性）
- **代码实现**：
  ```python
  # 相似度计算
  s_t = cosine_sim(m_vec, prompt_vec)  # [B]

  # 熵计算（不确定性度量）
  u_t = entropy_from_probs(next_token_probs)  # [B]

  # 滑动窗口统计
  delta_s, var_s = self._trigger_state.update(s_t)

  # 触发判断
  cond1 = (delta_s < -cfg.tau_drop) & (var_s > cfg.tau_var)  # 相似度快速下降且波动大
  cond2 = (u_t > cfg.tau_entropy) & (s_t < cfg.tau_sim)    # 不确定性高且相似度低
  trig = cond1 | cond2
  ```

#### 3. Translator（翻译器）
**作用**：将检测到的偏差转换为控制信号
- **输入**：
  - `z_vec`：think模式的latent thought向量（通过语言模型推理得到）
  - `m_vec`：视觉记忆向量（Condenser输出）
  - `p_vec`：原始prompt向量
- **机制**：
  - 三向量拼接后通过MLP生成控制向量
  - 使用门控机制控制控制信号强度
- **代码实现**：
  ```python
  # 输入拼接
  x = torch.cat([z_vec, m_vec, p_vec], dim=-1)  # [B, 3D]

  # MLP处理
  c = self.net(x)

  # 门控机制（控制强度）
  g = self.gate(c) * self.cfg.max_scale
  c_vec = c * g  # [B, D]
  ```

#### 4. Shaper（塑形器）
**作用**：将控制向量转换为可注入的control tokens
- **机制**：
  - 将控制向量映射为4个control tokens
  - 扩展为CFG兼容的格式 [2B, K, D]
- **代码实现**：
  ```python
  # 生成control tokens
  e = self.mlp(c_vec).view(b, self.cfg.control_tokens, d)  # [B, 4, D]

  # 扩展为CFG格式（条件+无条件）
  return expand_to_cfg_batch(e)  # [2B, 4, D]
  ```

### Think机制详解
**作用**：模拟"思考"过程，生成更智能的控制信号
- **过程**：
  1. 将prompt文本编码为token embeddings
  2. 与memory tokens拼接作为输入
  3. 语言模型forward一次获得latent thought
  4. pooling得到z_vec用于控制信号生成

**代码实现**：
```python
def _think_latent(self, model, prompt_text: str, m_tokens: torch.Tensor):
    # 1. Prompt编码
    ids = self.tokenizer.encode(prompt_text)
    if len(ids) > self.cfg.think_prompt_max_tokens:
        ids = ids[-self.cfg.think_prompt_max_tokens:]
    input_ids = torch.tensor(ids, device=m_tokens.device, dtype=torch.long).unsqueeze(0)

    # 2. Text embeddings
    text_emb = model.language_model.get_input_embeddings()(input_ids)

    # 3. 拼接输入：text + memory tokens
    inputs_embeds = torch.cat([text_emb, m_tokens], dim=1)  # [B, T+M, D]

    # 4. 语言模型推理
    out = model.language_model.model(inputs_embeds=inputs_embeds, use_cache=False)
    h = out.last_hidden_state  # [B, T+M, D]

    # 5. Pooling得到thought向量
    z_vec = h.mean(dim=1)  # [B, D]

    return z_vec
```

### KV Cache注入机制
**作用**：将控制信号无缝注入到生成过程中
- **机制**：
  1. 使用control tokens作为额外输入
  2. forward一次更新KV cache
  3. 后续生成继续使用更新后的cache
- **优势**：不回滚生成过程，无额外计算开销
- **代码实现**：
  ```python
  # 使用control tokens更新KV cache
  inj_out = model.language_model.model(
      inputs_embeds=ctrl_tokens,      # [2B, K, D]
      use_cache=True,
      past_key_values=past_key_values,  # 现有的KV cache
  )
  new_past_key_values = inj_out.past_key_values  # 更新后的cache
  ```

## 4. 发现的代码逻辑漏洞

### 严重问题

#### 1. 解码形状计算错误 (第79行)
```python
# 当前代码
shape=[parallel_size, channels, (img_size // patch_size) // 3 * stage_idx, img_size // patch_size]

# 问题分析
# 当stage_idx=1时: (384//16)//3*1 = 24//3*1 = 8*1 = 8
# 但实际应该是逐步增加的高度
# 正确的逻辑应该是:
# stage 1 (top): height = 8 (24//3)
# stage 2 (middle): height = 16 (24//3*2)
# stage 3 (bottom): height = 24 (24//3*3)
```

#### 2. 中间阶段图像保存错误 (第85行)
```python
# 当前代码
PIL.Image.fromarray(dec_prev_stage[i]).save(path)

# 问题: dec_prev_stage可能未正确初始化或形状不匹配
# 需要确保dec_prev_stage[i]是有效的uint8数组
```

#### 3. Latent Controller初始化错误 (第151行)
```python
# 当前代码
controller = LatentController(d_model=d_model, tokenizer=tokenizer, cfg=latent_cfg)

# 问题: tokenizer参数可能不正确，应该使用vl_chat_processor.tokenizer
```

#### 4. 控制逻辑中的参数问题 (第191行)
```python
# 当前代码
prompt_text_for_think=stage_text,

# 问题: stage_text可能包含格式化信息，不是纯净的prompt文本
```

### 中等问题

#### 5. 并行大小处理不一致
- `parallel_size = 1` 但代码中仍有并行处理的逻辑
- 部分代码假设parallel_size > 1，但实际使用时为1

#### 6. 内存管理
- 大量中间tensor没有及时释放
- 在循环中累积的past_key_values可能占用大量显存

## 5. 建议的修复方案

### 紧急修复

1. **修复解码形状计算**:
```python
# 正确的计算方式
stage_height = (img_size // patch_size) // stages * (stage_idx + 1)
shape=[parallel_size, channels, stage_height, img_size // patch_size]
```

2. **修复Latent Controller初始化**:
```python
controller = LatentController(
    d_model=d_model,
    tokenizer=vl_chat_processor.tokenizer,  # 使用正确的tokenizer
    cfg=latent_cfg
)
```

3. **添加图像有效性检查**:
```python
if dec_prev_stage is not None and len(dec_prev_stage) > i:
    PIL.Image.fromarray(dec_prev_stage[i]).save(path)
```

### 性能优化

1. **添加显存清理**:
```python
# 在每个阶段结束时
if stage_idx > 0:
    del dec_prev_stage
    torch.cuda.empty_cache()
```

2. **优化past_key_values管理**:
```python
# 限制past_key_values的缓存大小
if len(past_key_values) > max_cache_length:
    past_key_values = past_key_values[-max_cache_length:]
```

## 6. 总结

TwiG.py相比基础脚本的主要改进：
- ✅ 多阶段渐进式生成，提高生成质量
- ✅ 智能任务理解机制
- ✅ Latent control防止生成偏离
- ❌ 存在严重的解码逻辑错误
- ❌ 内存管理不够优化

Latent Control机制的核心创新在于：
1. **实时监控**：通过Condenser从生成过程中提取视觉记忆
2. **智能触发**：基于相似度和不确定性的双重触发机制
3. **思考干预**：使用语言模型生成控制信号
4. **无缝注入**：通过KV cache更新实现无损干预

建议优先修复解码形状计算错误，然后进行性能优化。
