# alphaTex模型实验设计报告

**日期**: 2026-02-04  
**项目**: nanochat alphaTex音乐生成模型优化  
**显存限制**: 8GB

---

## 📋 实验目标

通过控制变量法，系统性地探索模型架构对alphaTex音乐生成能力的影响：
- **实验1**: 测试**深度**（层数）对长程依赖学习的影响
- **实验2**: 测试**宽度**（维度）对表达能力的影响
- **评估**: 对比当前模型(d4) + 实验1(d8) + 实验2(d6_wide)的生成质量

---

## 🧪 实验设计

### 基准模型 (d4_baseline)

| 参数 | 值 | 说明 |
|------|-----|------|
| depth | 4 | Transformer层数 |
| n_embd | 128 | 隐藏层维度 (4×32) |
| n_head | 2 | 注意力头数 |
| 参数量 | ~5M | 轻量级配置 |
| 显存占用 | ~3.4GB | RTX 4060 |
| checkpoint | `model_001000.pt` | 当前已训练1000步 |

**特点**: 快速迭代，适合验证数据pipeline

---

### 实验1: 深度优先 (d8_depth)

**假设**: 更深的网络能更好地学习alphaTex的结构化模式（音轨嵌套、小节序列）

| 参数 | 值 | 对比基准 |
|------|-----|---------|
| depth | 8 | **2倍↑** |
| n_embd | 256 | 2倍↑ (8×32) |
| n_head | 4 | 2倍↑ (256/64) |
| 参数量 | ~20M | **4倍↑** |
| 预计显存 | ~6GB | 8GB限制内 |
| 训练步数 | 5000 | 快速验证 |

**学习率调整**:
- embedding_lr: 0.25 (略降，防止深层不稳定)
- matrix_lr: 0.018 (略降)
- warmup: 5% (新增，帮助深层收敛)

**优势**:
- 更强的长程依赖建模能力
- 更好的层次化特征学习
- 适合学习alphaTex的嵌套结构

---

### 实验2: 宽度优先 (d6_wide)

**假设**: 更宽的维度能提供更强的表达能力，学习更丰富的音乐特征

| 参数 | 值 | 对比基准 |
|------|-----|---------|
| depth | 6 | 适度增加 |
| n_embd | 384 | **3倍↑** (6×64) |
| n_head | 6 | 3倍↑ (384/64) |
| 参数量 | ~35M | **7倍↑** |
| 预计显存 | ~7.5GB | 接近8GB上限 |
| device_batch | 2 | 减半以控制显存 |
| 训练步数 | 5000 | 快速验证 |

**学习率调整**:
- embedding_lr: 0.2 (显著降低，宽模型更敏感)
- matrix_lr: 0.015 (降低)
- weight_decay: 0.15 (增加正则化)
- warmup: 8% (更长预热)

**优势**:
- 更强的特征表达能力
- 每个位置的信息容量更大
- 可能学习更复杂的和弦模式

---

## 📊 模型对比矩阵

| 模型 | 深度 | 维度 | 参数量 | 显存 | 特点 |
|------|------|------|--------|------|------|
| **d4_baseline** | 4 | 128 | ~5M | 3.4GB | 轻量、快速 |
| **d8_depth** | 8 | 256 | ~20M | ~6GB | 深度优先 |
| **d6_wide** | 6 | 384 | ~35M | ~7.5GB | 宽度优先 |

---

## 🔧 文件索引

### 训练脚本

| 脚本 | 路径 | 用途 |
|------|------|------|
| 实验1训练 | `workspace/exp1_train_d8_depth.sh` | 深度优先模型训练 |
| 实验2训练 | `workspace/exp2_train_d6_wide.sh` | 宽度优先模型训练 |

### 评估脚本

| 脚本 | 路径 | 用途 |
|------|------|------|
| 联合评估 | `workspace/evaluate_all_models.py` | 三模型对比评估 |

### 输出位置

```
data/t1/base_checkpoints/
├── d4_5gb/                    # 基准模型 (已存在)
│   └── model_001000.pt
├── d8_depth/                  # 实验1输出
│   └── model_005000.pt
└── d6_wide/                   # 实验2输出
    └── model_005000.pt
```

---

## 🚀 运行方法

### 前置条件

确保当前模型训练已完成或有可用的checkpoint：
```bash
ls data/t1/base_checkpoints/d4_5gb/model_001000.pt
# 或更高step的checkpoint
```

### 1. 运行实验1 (深度优先)

```bash
cd /mnt/f/workspace/SimpleTab/train_model/nanochat
bash workspace/exp1_train_d8_depth.sh
```

**预计时间**: ~2小时  
**监控**: `watch -n 1 nvidia-smi`

---

### 2. 运行实验2 (宽度优先)

```bash
cd /mnt/f/workspace/SimpleTab/train_model/nanochat
bash workspace/exp2_train_d6_wide.sh
```

**预计时间**: ~2.5小时  
**注意**: 此实验显存占用接近上限，请关闭其他GPU程序

---

### 3. 顺序运行两个实验

如果想连续运行两个实验（推荐）：

```bash
cd /mnt/f/workspace/SimpleTab/train_model/nanochat

# 使用tmux保持会话
tmux new-session -d -s nanochat-exp

# 运行实验1，完成后自动运行实验2
tmux send-keys -t nanochat-exp "bash workspace/exp1_train_d8_depth.sh && bash workspace/exp2_train_d6_wide.sh" Enter

# 查看进度
tmux attach -t nanochat-exp
```

---

### 4. 联合评估

当所有模型训练完成后，运行对比评估：

```bash
cd /mnt/f/workspace/SimpleTab/train_model/nanochat
source .venv/bin/activate
python workspace/evaluate_all_models.py
```

**评估内容**:
- 9个alphaTex风格prompt的生成质量
- 语法正确性检查
- 模型排名
- 输出报告: `workspace/evaluation_results.json`

---

## 📈 评估指标

### 生成质量检查项

| 检查项 | 说明 |
|--------|------|
| has_title | 是否生成`\title`标签 |
| has_track | 是否生成`\track`标签 |
| has_braces | 是否有正确的大括号配对 |
| has_quotes | 是否有引号包围字符串 |
| valid_tokens | 是否包含无效字符 |
| no_gibberish | 是否有足够的字母内容 |

### 评分标准

- **质量评分**: 0-100%，基于6项检查通过率
- **平均质量**: 所有prompt的平均得分
- **模型排名**: 按平均质量排序

---

## 🎯 预期结果分析

### 情景1: d8_depth表现最佳
- **结论**: alphaTex学习更受益于深度（长程依赖）
- **下一步**: 尝试d10或d12，进一步优化深度

### 情景2: d6_wide表现最佳
- **结论**: alphaTex学习更受益于宽度（表达能力）
- **下一步**: 尝试更大的维度(512)或适度增加深度

### 情景3: d4_baseline仍然最佳
- **结论**: 当前数据量(小数据)下，小模型已足够
- **下一步**: 增加数据量后再尝试大模型

---

## ⚠️ 注意事项

### 显存管理

- **d8_depth**: 安全范围 (~6GB/8GB)
- **d6_wide**: 接近上限 (~7.5GB/8GB)
  - 如果OOM，减小`device-batch-size`到1
  - 或减小`max-seq-len`到384

### 训练中断恢复

如果训练中断，可以使用相同的命令恢复：
```bash
# 脚本会自动从最新的checkpoint恢复
bash workspace/exp1_train_d8_depth.sh
```

### 评估时模型不存在

如果某个实验尚未完成，评估脚本会自动跳过该模型，只对比可用的模型。

---

## 📝 实验记录模板

建议在实验完成后记录以下信息：

```markdown
## 实验结果记录

### 实验1 (d8_depth)
- 实际训练时间: ___小时
- 最终loss: ___
- 显存峰值: ___GB
- 质量评分: ___%
- 观察: ___

### 实验2 (d6_wide)
- 实际训练时间: ___小时
- 最终loss: ___
- 显存峰值: ___GB
- 质量评分: ___%
- 观察: ___

### 结论
- 最佳模型: ___
- 关键发现: ___
- 下一步: ___
```

---

## 🔮 后续优化方向

根据实验结果，可以考虑：

1. **序列长度**: 如果模型表现好但生成长度不够，尝试`max_seq_len=1024`
2. **词表扩展**: 如果OOV问题严重，尝试`vocab_size=16384`
3. **数据增强**: 增加更多样化的alphaTex样本
4. **微调策略**: 在大模型基础上用高质量数据微调

---

**报告完成时间**: 2026-02-04  
**实验设计**: 控制变量法，对比深度vs宽度的影响  
**预期总耗时**: 4-5小时（两个实验）
