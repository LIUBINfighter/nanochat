# 视觉层集成策略报告

**日期**: 2026-02-04  
**目标**: 为d8架构alphaTex模型规划视觉层集成路径  
**显存预算**: 8GB  
**核心原则**: 保留文本能力，渐进式视觉化

---

## 📋 当前架构优势分析

### d8架构特点（视觉化友好）

| 特性 | 当前实现 | 视觉化优势 |
|------|---------|-----------|
| **RoPE位置编码** | 旋转位置编码 | ✅ 天然支持1D序列扩展，易于适配2D图像位置 |
| **GQA注意力** | Group Query Attention | ✅ 减少视觉token的KV缓存，高效处理长视觉序列 |
| **Flash Attention** | FA3/SDPA | ✅ 处理大量视觉token时内存高效 |
| **8层深度** | 适中的transformer深度 | ✅ 深层适合复杂视觉-文本融合 |
| **256维嵌入** | n_embd=256 | ✅ 维度适中，视觉投影层参数量可控 |

### 关键发现
- 8层深度比宽度更适合alphaTex结构化数据（质量83% vs 77%）
- 深度模型更适合处理视觉-文本的多模态融合

---

## 🚀 视觉Token预留策略

### 1. 词表预留方案

在tokenizer中预留视觉占位符token：

```python
# nanochat/tokenizer.py 扩展 SPECIAL_TOKENS
SPECIAL_TOKENS = [
    # 现有token...
    "<|bos|>",
    "<|user_start|>",
    # 新增视觉占位符
    "<|vision_start|>",     # 视觉序列开始
    "<|vision_end|>",       # 视觉序列结束  
    "<|vision_pad|>",       # 视觉序列padding
]
```

**预留数量**: 3-5个特殊token位置
**当前词表**: 8192，可扩展至16384以容纳更多视觉相关token

---

## 🏗️ 架构分层设计

### 方案: 浅层视觉编码 + 深层融合

```
输入: 图像patch (如16x16=256个patch)
  ↓
Vision Encoder (可学习的投影层)
  ↓
[视觉token序列: 256个] + [文本token序列]
  ↓
浅层Transformer (前2-3层): 视觉专用处理
  ↓
深层Transformer (后5-6层): 视觉-文本融合 (d8架构)
  ↓
输出: 文本token (alphaTex)
```

### d8架构分工建议

| 层数 | 功能 | 说明 |
|------|------|------|
| 第1-2层 | 纯视觉处理 | 学习图像局部特征 |
| 第3-5层 | 视觉-文本融合 | 跨模态交互关键层 |
| 第6-8层 | 纯文本生成 | **保留当前训练的alphaTex能力** |

**关键**: 第6-8层不做修改，预训练权重直接复用！

---

## 📐 位置编码适配

### 方案1: 1D展平 (推荐初期)

```python
# 视觉位置: 0-255 (图像patch展平)
# 文本位置: 256-511 (接在视觉之后)

# 当前训练调整为:
--max-seq-len=1024  # 从512扩展到1024
# 视觉: 512 tokens
# 文本: 512 tokens
```

### 方案2: 2D-aware RoPE (进阶)

```python
class VisionRoPE:
    """给视觉token添加2D感知的位置编码"""
    def __init__(self, img_h, img_w, head_dim):
        self.rope_x = RoPE(img_w, head_dim//2)  # 水平
        self.rope_y = RoPE(img_h, head_dim//2)  # 垂直
    
    def forward(self, vision_tokens):
        # 将2D位置分解为x,y两个1D位置编码后拼接
        pass
```

**建议**: 初期用方案1，后期优化用方案2

---

## 💡 预训练期策略（当前阶段）

### 策略1: 视觉占位符预训练

在.atex文本前添加视觉占位符，让模型预分配注意力机制：

```python
# prepare_data.py 修改
VISION_PAD = "<|vision_pad|> " * 256  # 256个占位符

def convert_to_parquet():
    for doc in documents:
        text_with_vision = VISION_PAD + doc
        # 保存...
```

**目的**: 让模型"习惯"序列前半部分有256个特殊token

### 策略2: 序列长度扩展

**立即调整**: `--max-seq-len=1024`

```bash
python -m scripts.base_train \
    --depth=8 \
    --aspect-ratio=32 \
    --max-seq-len=1024 \      # 关键扩展
    --device-batch-size=2 \    # batch减半保持显存
    ...
```

**分布**:
- 位置0-511: 预留给视觉
- 位置512-1023: 文本生成（alphaTex）

### 策略3: GQA优化视觉token处理

当前配置优化:
```python
vision_attention_config = {
    "n_head": 4,
    "n_kv_head": 2,  # GQA: KV缓存减半
    # 256个视觉patch的KV缓存减少50%
}
```

**好处**: 显存压力小，适合8GB预算

---

## 🔧 下一阶段架构改造

### 多模态GPT架构

```python
class MultimodalGPT(nn.Module):
    def __init__(self, text_config, vision_config):
        # 1. 冻结的d8文本模型（当前训练成果）
        self.text_transformer = GPT(text_config)
        for param in self.text_transformer.parameters():
            param.requires_grad = False  # 冻结保留能力
        
        # 2. 视觉编码器（新增）
        self.vision_encoder = ViTEncoder(
            patch_size=16,
            embed_dim=256,  # 匹配文本维度
            num_layers=3,   # 浅层即可
        )
        
        # 3. 视觉-文本投影层（关键）
        self.vision_proj = nn.Linear(256, 256)
        
        # 4. 浅层融合层（新增）
        self.fusion_layers = nn.ModuleList([
            Block(fusion_config) for _ in range(3)
        ])
```

### 接口定义

```python
class VisionTextInterface:
    """明确定义的视觉-文本接口"""
    
    def __init__(self, text_model):
        self.text_layers = text_model.layers[3:8]  # 后5层（冻结）
        self.embedding_dim = 256
        self.vision_seq_len = 256  # 预留位置
        
    def forward(self, vision_features, text_tokens):
        """
        Args:
            vision_features: (B, N_v, 256) 视觉特征
            text_tokens: (B, N_t, 256) 文本token
        Returns:
            生成的alphaTex文本
        """
        combined = torch.cat([vision_features, text_tokens], dim=1)
        return self.text_layers(combined)
```

---

## 📊 分阶段实施计划

### 阶段1: 文本预训练期（当前）

**目标**: 训练纯文本alphaTex模型，预留视觉接口

**具体工作**:
1. ✅ 在tokenizer中加入 `<|vision_pad|>` 等特殊token
2. ✅ 将 `--max-seq-len` 改为1024
3. ✅ 前512位置用占位符填充训练
4. ✅ 完成d8_depth_v2到2万步训练

**交付物**:
- 冻结的文本生成能力（第6-8层）
- 预训练好的位置编码（适应1024长度）
- 预留的视觉token位置（512个）

### 阶段2: 视觉对齐期（下一阶段）

**目标**: 添加视觉编码能力，实现图像→alphaTex

**具体工作**:
1. 冻结d8后5层（第3-8层）
2. 添加ViT视觉编码器（3层浅层）
3. 添加视觉投影层
4. 训练前3层融合层

**预期显存**: 7-8GB（8GB预算内）

### 阶段3: 能力扩展期（未来）

**目标**: 完整多模态能力

**功能**:
- 图像编辑alphaTex
- 乐谱OCR识别
- 音乐教学对话

---

## ⚠️ 关键注意事项

### 1. 能力保持
- **绝对不能破坏**已训练的alphaTex生成能力
- 后5层必须冻结，只训练浅层融合层
- 定期测试文本-only生成质量

### 2. 显存管理
- 使用GQA减少视觉token的KV缓存
- Flash Attention处理长视觉序列
- 8GB预算内：视觉512 + 文本512 = 1024总长

### 3. 数据准备
- 准备图像-alphaTex配对数据
- 图像分辨率建议：16x32=512 patches 或 16x16=256 patches
- 数据格式：图像 + 对应的alphaTex标注

---

## 🎯 预期效果

| 阶段 | 能力 | 显存 | 模型大小 |
|------|------|------|---------|
| **当前d8** | 纯文本alphaTex | ~6GB | 20M |
| **+视觉编码** | 图像→alphaTex | ~7-8GB | +10M视觉层 |
| **完整多模态** | 图像编辑、OCR | ~8GB+ | +20M完整视觉 |

**关键优势**:
- ✅ 文本能力完全保留
- ✅ 视觉层独立迭代
- ✅ 显存可控
- ✅ 架构清晰

---

## 📁 相关文件

| 文件 | 说明 |
|------|------|
| `nanochat/gpt.py` | d8架构定义，RoPE/GQA/Flash Attention |
| `nanochat/tokenizer.py` | 需要扩展视觉占位符token |
| `scripts/base_train.py` | 需要支持1024序列长度 |
| `workspace/prepare_data.py` | 需要添加视觉占位符 |

---

## 💡 总结

d8架构天然适合多模态扩展：

1. **RoPE** → 易于扩展到2D视觉位置
2. **GQA** → 高效处理大量视觉token
3. **8层深度** → 完美分层：浅层视觉 + 深层融合 + 冻结文本
4. **256维** → 视觉投影参数量适中

**推荐路径**:
1. 当前：完成d8_v2训练，扩展到1024长度，预留视觉位置
2. 下一阶段：冻结后5层，添加3层视觉编码 + 3层融合层
3. 未来：完整多模态系统

这是**LLaVA、Qwen-VL等主流多模态LLM的标准做法**，你的d8架构完全适配！

---

**报告完成时间**: 2026-02-04  
**适用模型**: d8_depth (8层, 256维)  
**目标**: 图像→alphaTex音乐谱生成
