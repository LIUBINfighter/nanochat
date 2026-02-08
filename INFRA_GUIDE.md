# nanochat 模型基建技术指南

> **目标读者**: 新加入的基建维护同学  
> **前置知识**: 文本自回归Pretrain基础  
> **文档定位**: 从原理到实践的全链路技术串讲

---

## 一、文本自回归Pretrain基础知识

### 1.1 核心概念

**自回归语言模型 (Autoregressive LM)** 是GPT系列的核心范式：

```
给定序列: x₁, x₂, x₃, ..., xₜ
预测目标: P(xₜ₊₁ | x₁, x₂, ..., xₜ)

Loss = -log P(xₜ₊₁ | x₁ₜ)
```

**关键特点**:
- **单向**: 只能看到前面的token，不能看后文
- **因果掩码**: 使用下三角矩阵确保时序因果性
- **逐token预测**: 每次预测下一个token的概率分布

### 1.2 训练目标

**交叉熵损失**:
```python
loss = F.cross_entropy(logits, targets)
```

**Bits Per Byte (BPB)** - 压缩效率指标:
```
BPB = CrossEntropy / ln(2) / bytes_per_token
```
- BPB < 1.0: 比原始数据压缩率更高
- BPB越低，模型对数据分布建模越好

### 1.3 Scaling Laws (缩放定律)

**Chinchilla Paper (DeepMind, 2022)** 核心结论:

```
最优训练token数 = 20 × 参数量 (非嵌入参数)

即: tokens : params ≈ 20:1
```

**nanochat的实践**:
- 使用 `target-param-data-ratio=10.5` (计算最优，训练更快)
- Speedrun使用 `target-param-data-ratio=12` (轻微过训练以加速收敛)

---

## 二、nanochat项目架构总览

### 2.1 代码结构

```
nanochat/
├── nanochat/
│   ├── gpt.py              # 核心Transformer模型
│   ├── optim.py            # Muon + AdamW混合优化器
│   ├── dataloader.py       # 分布式数据加载
│   ├── engine.py           # 推理引擎(KV Cache)
│   └── core_eval.py        # CORE指标评估
├── scripts/
│   ├── base_train.py       # 预训练脚本
│   ├── base_eval.py        # 评估脚本
│   ├── chat_sft.py         # SFT微调
│   └── chat_web.py         # WebUI
└── runs/
    ├── speedrun.sh         # 主力训练脚本
    ├── scaling_laws.sh     # 缩放定律实验
    └── miniseries.sh       # 多尺度模型系列
```

### 2.2 训练Pipeline

```
[Tokenizer训练] → [Pretrain] → [SFT微调] → [Chat模型]
       ↓               ↓              ↓
   词表构建      自回归训练      对话能力
   BPE算法       10B+ tokens    指令遵循
```

---

## 三、主力模型架构详解

### 3.1 核心设计理念

**nanochat的设计哲学**:
1. **单一复杂度旋钮**: 只用 `--depth` (层数) 控制模型大小
2. **固定长宽比**: `aspect-ratio=64` → `dim = depth × 64`
3. **现代Transformer**: 融合Llama、Qwen等最新改进

### 3.2 GPTConfig 配置类

```python
@dataclass
class GPTConfig:
    sequence_len: int = 2048        # 最大序列长度
    vocab_size: int = 32768         # 词表大小 (2^15)
    n_layer: int = 12               # Transformer层数
    n_head: int = 6                 # Query头数
    n_kv_head: int = 6              # KV头数 (GQA比例)
    n_embd: int = 768               # 隐藏层维度
    window_pattern: str = "SSSL"    # 滑动窗口模式
```

### 3.3 模型架构创新点

| 特性 | 实现 | 来源/动机 |
|------|------|-----------|
| **RoPE** | `apply_rotary_emb()` | Llama系列，相对位置编码 |
| **QK Norm** | `norm(q), norm(k)` | 稳定训练，防止softmax饱和 |
| **relu²** | `F.relu(x).square()` | 替代GELU，计算更快 |
| **Untied Weights** | wte ≠ lm_head | 词嵌入和解码独立 |
| **Sliding Window** | SSSL模式 | 长文本效率，最后层全窗口 |
| **Value Embeddings** | ResFormer风格 | 交替层value残差 |
| **Layer-wise Scalars** | resid_lambdas, x0_lambdas | modded-nanogpt创新 |
| **RMSNorm** | `F.rms_norm()` | 无学习参数，更稳定 |

### 3.4 注意力机制详解

```python
class CausalSelfAttention(nn.Module):
    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        # 1. QKV投影
        q = self.c_q(x).view(B, T, n_head, head_dim)
        k = self.c_k(x).view(B, T, n_kv_head, head_dim)
        v = self.c_v(x).view(B, T, n_kv_head, head_dim)
        
        # 2. Value残差 (ResFormer)
        if ve is not None:
            gate = 2 * torch.sigmoid(self.ve_gate(x))
            v = v + gate * ve
        
        # 3. RoPE位置编码
        q, k = apply_rotary_emb(q, cos_sin), apply_rotary_emb(k, cos_sin)
        q, k = norm(q), norm(k)  # QK Norm
        
        # 4. Flash Attention 3
        y = flash_attn(q, k, v, causal=True, window_size=window_size)
        
        return self.c_proj(y)
```

---

## 四、基建参数配置

### 4.1 主力模型矩阵

| 模型代号 | 层数 | 维度 | 头数 | 参数量 | 显存需求 | 适用场景 |
|----------|------|------|------|--------|----------|----------|
| **d4** | 4 | 256 | 4 | ~20M | 5-6GB | 快速实验 |
| **d8** | 8 | 512 | 8 | ~70M | 7-8GB | alphaTex主力 |
| **d12** | 12 | 768 | 12 | ~150M | 12GB | 研究迭代 |
| **d16** | 16 | 1024 | 16 | ~300M | 20GB | 大规模实验 |
| **d20** | 20 | 1280 | 20 | ~500M | 40GB | 接近GPT-2 |
| **d24** | 24 | 1536 | 24 | ~800M | 80GB | GPT-2级 |

**注**: 本地alphaTex项目目前主力使用 **d8_depth** (8层深度优先)

### 4.2 d8_depth 详细配置 (alphaTex主力)

```bash
# ========== 模型架构 ==========
--depth=8                   # 8层Transformer
--aspect-ratio=32           # 维度 = 8 × 32 = 256
--head-dim=64               # 每个头64维 → 4个头
--max-seq-len=512           # 序列长度512 (音乐片段)
--window-pattern=L          # 全长度注意力

# ========== 训练配置 ==========
--device-batch-size=4       # 每设备batch
--total-batch-size=4096     # 全局batch = 4096 tokens
--num-iterations=20000      # 总训练步数

# ========== 优化器配置 ==========
--embedding-lr=0.22         # 嵌入层学习率
--matrix-lr=0.016           # 矩阵参数学习率(Muon)
--unembedding-lr=0.003      # 输出层学习率
--weight-decay=0.12         # 权重衰减

# ========== 学习率调度 ==========
--warmup-ratio=0.05         # 5%步数warmup
--warmdown-ratio=0.25       # 25%步数衰减
--final-lr-frac=0.05        # 最终LR=5%初始值
```

### 4.3 d6_wide 对比配置 (宽度优先)

```bash
# ========== 模型架构 ==========
--depth=6                   # 6层
--aspect-ratio=64           # 维度 = 6 × 64 = 384
--head-dim=64               # 6个头 (384/64)
--max-seq-len=512
--window-pattern=L

# ========== 训练配置 ==========
--device-batch-size=2       # 显存更大，减小batch
--total-batch-size=4096
--num-iterations=5000

# ========== 优化器配置 ==========
--embedding-lr=0.2          # 宽模型更敏感，降低LR
--matrix-lr=0.015
--unembedding-lr=0.003
--weight-decay=0.15         # 增加正则化
--warmup-ratio=0.08         # 更长warmup
```

---

## 五、优化器基建深度解析

### 5.1 Muon + AdamW 混合策略

**核心思想**:
- **Matrix参数** (Linear权重): 使用 **Muon** (Momentum Orthogonalization)
- **非Matrix参数** (嵌入、标量): 使用 **AdamW**

**参数分组**:
```python
param_groups = [
    # AdamW组
    {'kind': 'adamw', 'params': lm_head_params, 'lr': unembedding_lr},
    {'kind': 'adamw', 'params': embedding_params, 'lr': embedding_lr},
    {'kind': 'adamw', 'params': scalar_params, 'lr': scalar_lr},
    
    # Muon组 (按shape分组stack)
    {'kind': 'muon', 'params': matrix_params, 'lr': matrix_lr, 
     'momentum': 0.95, 'ns_steps': 5},
]
```

### 5.2 Muon优化器原理

**关键创新 - Polar Express (2025)**:

```python
# Newton-Schulz迭代 → Polar Express迭代
def polar_express(X, num_iters=5):
    X = X / (X.norm() * 1.02 + 1e-6)  # 归一化
    
    for a, b, c in polar_express_coeffs:
        if X是 tall矩阵:
            A = X.mT @ X
            B = b * A + c * (A @ A)
            X = a * X + X @ B
        else:  # wide矩阵
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
    
    return X  # 近似正交矩阵
```

**相比AdamW的优势**:
1. **正交化更新**: 保持梯度方向多样性
2. **谱范数控制**: 更稳定的参数更新
3. **bf16稳定**: 可在低精度下稳定运行

### 5.3 分布式优化

**DistMuonAdamW** 特点:
- **ZeRO-2风格**: 优化器状态分片到各GPU
- **3阶段异步通信**: Reduce → Compute → Gather
- **参数分桶**: 小参数all_reduce, 大参数reduce_scatter

```python
# 通信流程
Phase 1: 启动所有异步reduce操作
Phase 2: 等待reduce → 计算更新 → 启动gather
Phase 3: 等待gather → 复制回原始参数
```

---

## 六、训练调优实战

### 6.1 学习率缩放规则

**Batch Size缩放** (参考batch=2^19=524288):
```python
# AdamW: sqrt缩放
lr_scale = (current_batch / 524288) ** 0.5

# 示例: batch=4096时的缩放
scale = (4096 / 524288) ** 0.5 = 0.088
```

**模型维度缩放** (参考dim=768):
```python
# 参数按 1/√d 缩放
dmodel_lr_scale = (model_dim / 768) ** -0.5

# 示例: dim=256时的缩放
scale = (256 / 768) ** -0.5 = 1.732
```

**Weight Decay缩放** (参考depth=12):
```python
# decay ∝ 1/depth²
wd_scaled = base_wd * (12 / depth) ** 2

# 示例: depth=8时的缩放
wd = 0.12 * (12/8)**2 = 0.27 → 实际使用0.12 (经验调整)
```

### 6.2 显存优化策略

| 优化手段 | 效果 | 实现方式 |
|----------|------|----------|
| **Gradient Accumulation** | 线性节省 | `grad_accum_steps = total_batch / (device_batch × world_size)` |
| **BF16混合精度** | ~50%节省 | `torch.amp.autocast(dtype=torch.bfloat16)` |
| **Expandable Segments** | 防OOM | `PYTORCH_ALLOC_CONF=expandable_segments:True` |
| **Torch Compile** | 速度+显存优化 | `torch.compile(model, dynamic=False)` |
| **Sliding Window** | 注意力内存 ↓ | `window_pattern=SSSL` |

### 6.3 训练监控指标

```python
# 每步输出
step 01670/20000 (8.35%) | loss: 0.823456 | lrm: 1.00 | 
dt: 125.43ms | tok/sec: 32,658 | mfu: 42.15 | 
epoch: 3 | total time: 12.45m | eta: 135.2m

# 关键指标解释
- lrm: Learning Rate Multiplier (调度器倍率)
- dt: 每步耗时(ms)
- tok/sec: 吞吐率
- mfu: Model FLOPs Utilization (模型浮点利用率)
- eta: 预计剩余时间
```

**MFU计算**:
```python
flops_per_sec = num_flops_per_token × batch_size / dt
mfu = flops_per_sec / (gpu_peak_flops × num_gpus)

# H100 BF16峰值: 989 TFLOPS
# 目标MFU: 40-50% (优秀), 30-40% (良好)
```

---

## 七、实验矩阵对比

### 7.1 当前alphaTex实验矩阵

| 实验 | 深度 | 维度 | 参数量 | 训练步数 | 验证BPB | 质量评分 | 特点 |
|------|------|------|--------|----------|---------|----------|------|
| **d4_baseline** | 4 | 128 | ~5M | 1,000 | 1.42 | ~60% | 快速验证基线 |
| **d8_depth** | 8 | 256 | ~20M | 5,000 | 0.66 | 83.0% | **当前主力** |
| **d6_wide** | 6 | 384 | ~35M | 5,000 | 待测 | 待测 | 宽度优先对比 |
| **d8_depth_v2** | 8 | 256 | ~20M | 20,000 | 待测 | 待测 | 深度+更多数据 |

### 7.2 GPT-2 Speedrun对比

| 指标 | GPT-2 (2019) | nanochat d24 (2026) | 提升 |
|------|--------------|---------------------|------|
| **参数量** | 1.6B | ~800M | -50% |
| **训练成本** | ~$50,000 | ~$73 | **684x↓** |
| **训练时间** | 数周 | 3.04小时 | **>100x↓** |
| **CORE分数** | 0.256525 | 0.25851 | ✅ 超越 |
| **硬件** | 大量TPU | 8×H100 GPU | 单机完成 |

### 7.3 深度 vs 宽度 设计权衡

**深度优先 (d8_depth)**:
```
优势:
- 更强的长程依赖建模
- 更好的层次化特征学习
- 适合alphaTex嵌套结构

劣势:
- 梯度传播路径长
- 需要 careful initialization

适用: 结构化数据(alphaTex)、代码
```

**宽度优先 (d6_wide)**:
```
优势:
- 更强的表达能力
- 每层信息容量更大
- 并行度更高

劣势:
- 注意力计算量 ↑↑
- 显存占用更大

适用: 复杂模式、丰富特征
```

---

## 八、脚本速查手册

### 8.1 训练脚本

```bash
# ========== 主力训练 ==========
# d8_depth 从0训练
bash workspace/exp1_train_d8_depth.sh

# d8_depth 续训 (5000→20000)
bash workspace/resume_d8_depth_5k_to_20k.sh

# d6_wide 训练
bash workspace/exp2_train_d6_wide.sh

# ========== 通用模板 ==========
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- \
    --depth=8 \
    --aspect-ratio=32 \
    --head-dim=64 \
    --max-seq-len=512 \
    --device-batch-size=4 \
    --total-batch-size=4096 \
    --num-iterations=20000 \
    --run="d8_main" \
    --model-tag="d8_depth"
```

### 8.2 评估脚本

```bash
# 基础评估
python -m scripts.base_eval -- --device-batch-size=4

# 多模型对比评估
python workspace/enhanced_compare_with_note_completion.py

# CORE指标 (学术标准)
# 自动在base_train中运行
```

### 8.3 环境变量

```bash
# 数据目录
export NANOCHAT_BASE_DIR="./data/t1"

# 显存优化
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# 多线程
export OMP_NUM_THREADS=4

# WandB日志 (可选)
export WANDB_MODE=offline
export WANDB_DISABLED=true
```

---

## 九、常见问题排查

### 9.1 OOM (显存溢出)

```bash
# 解决步骤:
1. 减小 --device-batch-size (4 → 2 → 1)
2. 减小 --max-seq-len (512 → 384 → 256)
3. 使用更小的 depth (8 → 6 → 4)
4. 启用 gradient checkpointing (需要代码修改)
```

### 9.2 训练不收敛

```bash
# 检查清单:
□ 学习率是否过大? (embedding_lr > 0.5?)
□ warmup比例是否足够? (warmup-ratio < 0.05?)
□ weight_decay是否过大? (wd > 0.2?)
□ 数据是否正确加载? (检查dataloader输出)

# 调试命令
python -m scripts.base_train --num-iterations=10 --device-batch-size=1
```

### 9.3 断点续训

```bash
# 自动检测最新checkpoint
python -m scripts.base_train \
    --resume-from-step=5000 \
    --num-iterations=20000 \
    --model-tag="d8_depth"
```

---

## 十、进阶阅读

### 10.1 关键论文

1. **Attention Is All You Need** (Transformer基础)
2. **Llama 2/3 Papers** (现代架构改进)
3. **Chinchilla** (Scaling Laws)
4. **Muon Paper** (Keller Jordan's blog)
5. **Polar Express** (2025, 正交优化)

### 10.2 相关项目

- **nanoGPT**: 前身项目，仅预训练
- **modded-nanogpt**: 游戏化改进版本
- **llm.c**: Andrej的纯C实现

### 10.3 社区资源

- [nanochat Discussions](https://github.com/karpathy/nanochat/discussions)
- [#nanochat Discord](https://discord.com/channels/1020383067459821711/1427295580895314031)
- [DeepWiki](https://deepwiki.com/karpathy/nanochat)

---

## 附录: 快速启动清单

对于新加入的同学，请按以下顺序熟悉基建:

- [ ] 阅读本技术指南
- [ ] 跑通 `runs/runcpu.sh` (理解流程)
- [ ] 在GPU上跑 `exp1_train_d8_depth.sh`
- [ ] 阅读 `nanochat/gpt.py` 核心架构
- [ ] 阅读 `nanochat/optim.py` 优化器实现
- [ ] 跑通评估脚本，理解指标含义
- [ ] 尝试修改一个小参数并观察效果

---

**文档版本**: 2026-02-07  
**维护者**: nanochat团队  
**更新频率**: 随重要改动同步更新
