# 深度优先v2实验设计报告

**日期**: 2026-02-04  
**目标**: 基于最佳d8_depth模型继续训练到2万步，增强音符补全能力  
**显存预算**: 8GB

---

## 📋 实验目标

### 1. 继续训练d8_depth到2万步
- 当前: Step 5000, Loss 0.66, 质量83.0%
- 目标: Step 20000, 预期Loss < 0.5, 质量提升

### 2. 深度优先策略验证
- 保持8层架构不变
- 验证更多训练步数对深度模型的收益

### 3. 增强音符补全能力评估
新增两类测试：
- **Tab补全**: `| 3.2` → 补全吉他谱数字
- **复杂音符**: `(6.2 1.5).8` → 补全带节奏的和弦

---

## 🧪 实验配置

### 续训脚本: `resume_d8_depth_5k_to_20k.sh`

| 参数 | 值 | 说明 |
|------|-----|------|
| **起始步数** | 5000 | 从最佳checkpoint恢复 |
| **目标步数** | 20000 | 训练15000 iterations |
| **深度** | 8 | 保持深度优先架构 |
| **维度** | 256 | 8×32 |
| **注意力头** | 4 | 256/64 |
| **参数量** | ~20M | 8GB预算内 |
| **Batch** | 4 | 保持总batch 4096 |
| **预计显存** | ~6GB | 安全范围 |
| **预计时间** | 1.5-2小时 | GPU加速 |

### 学习率调整

```bash
--embedding-lr=0.22      # 从0.25降低 (后期训练更稳定)
--matrix-lr=0.016        # 从0.018降低
--unembedding-lr=0.003   # 从0.0035降低
--weight-decay=0.12      # 保持
--warmdown-ratio=0.25    # 最后25%步数学习率衰减
--final-lr-frac=0.05     # 最终学习率5%
```

**调整理由**: 
- 后期训练需要更小的学习率
- 避免过拟合
- warmdown帮助收敛

---

## 📊 评估体系升级

### 增强版评估脚本: `enhanced_compare_with_note_completion.py`

新增**4个模型**对比：
1. **d4_baseline**: 基准 (step 1000)
2. **d8_depth**: 实验1 (step 5000) 
3. **d6_wide**: 实验2 (step 5000)
4. **d8_depth_v2**: 续训模型 (step 20000) ← 新增

### 三类测试

#### 1. 基础结构生成 (40%权重)
测试alphaTex整体结构：
- `\title "`
- `\track (...)`
- `\tempo 120`
- 完整开头

评估指标：has_title, has_track, has_braces, has_instrument等

#### 2. Tab补全能力 (30%权重)
测试吉他谱数字补全：
```
| 3.2              → 补全为 | 3.2 2.3 1.2
| (3.2 2.3).8      → 补全节奏和后续
| 1.2 2.3 3.2 4.2  → 补全序列
```

评估指标：
- has_fret_number: 是否生成数字
- has_dot_separator: 是否有.分隔符
- valid_format: 格式是否正确
- has_rhythm: 是否生成节奏(.8/.4等)

#### 3. 复杂音符补全 (30%权重)
测试alphaTex复杂语法补全：
```
(6.2 1.5).        → 补全为 (6.2 1.5).8
(4.2 1.5{t}).     → 补全装饰音
24.1.             → 补全单弦音符
```

评估指标：
- has_parentheses: 是否有括号配对
- has_fret_numbers: 是否有弦.品格式
- has_rhythm_dot: 是否有节奏标记
- has_effects: 是否支持{t}{dy}等效果

### 综合得分计算
```
综合得分 = 结构×40% + Tab×30% + 音符×30%
```

---

## 🚀 运行方法

### 第一步: 续训d8_depth到2万步

```bash
cd /mnt/f/workspace/SimpleTab/train_model/nanochat
bash workspace/resume_d8_depth_5k_to_20k.sh
```

**监控训练**:
```bash
# 另一个终端
watch -n 1 nvidia-smi
```

**预计**: 1.5-2小时后完成

---

### 第二步: 运行增强版对比评估

当d8_depth_v2训练完成后，运行对比：

```bash
cd /mnt/f/workspace/SimpleTab/train_model/nanochat
source .venv/bin/activate
python workspace/enhanced_compare_with_note_completion.py
```

**输出**:
- 三/四类模型对比结果
- 专项能力排名
- 综合得分排名
- JSON报告: `enhanced_comparison_results.json`

---

### 可选: 顺序执行

如果你想让训练完成后自动运行评估：

```bash
cd /mnt/f/workspace/SimpleTab/train_model/nanochat

# 使用tmux
tmux new-session -d -s nanochat-train
tmux send-keys -t nanochat-train "bash workspace/resume_d8_depth_5k_to_20k.sh && python workspace/enhanced_compare_with_note_completion.py" Enter
tmux attach -t nanochat-train
```

---

## 📈 预期结果分析

### 情景1: d8_depth_v2大幅领先
- **表现**: 综合得分 > 85%，明显优于d8_5k
- **结论**: 更多训练步数对深度模型收益很大
- **下一步**: 考虑训练到5万步或更深的网络

### 情景2: d8_depth_v2小幅提升
- **表现**: 综合得分 80-85%，略优于d8_5k
- **结论**: 收益递减，可能接近当前架构上限
- **下一步**: 尝试其他优化（词表、数据）

### 情景3: 无明显提升
- **表现**: 综合得分与d8_5k接近
- **结论**: 可能过拟合或数据瓶颈
- **下一步**: 增加数据多样性或正则化

---

## 🎯 关键观察指标

### 训练过程
- **Loss曲线**: 能否降到0.5以下
- **收敛速度**: 是否快速稳定
- **显存占用**: 是否稳定在6GB左右

### 评估结果
- **结构生成**: 能否维持或提升83%+
- **Tab补全**: 新能力能否达到70%+
- **音符补全**: 新能力能否达到60%+

### 能力对比
- **5K vs 20K**: 训练步数的边际收益
- **深度vs宽度**: d8_20k vs d6_5k
- **性价比**: 参数量vs质量提升

---

## 💡 设计亮点

### 1. 渐进式训练
- 基于已验证的最佳模型继续训练
- 降低风险，复用成功经验

### 2. 专项能力评估
- 不仅测试整体结构
- 专门测试音符级别的补全能力
- 更接近实际应用场景

### 3. 综合评分体系
- 多维度评估（结构+Tab+音符）
- 权重合理分配
- 避免单一指标偏见

### 4. 控制变量
- 保持架构不变，只改变训练步数
- 清晰验证"更多数据/步数"的效果

---

## 📁 文件索引

| 文件 | 用途 |
|------|------|
| `resume_d8_depth_5k_to_20k.sh` | 续训脚本 |
| `enhanced_compare_with_note_completion.py` | 增强版评估 |
| `EXPERIMENT_V2_DESIGN.md` | 本报告 |

### 输出位置
```
data/t1/base_checkpoints/d8_depth/
├── model_005000.pt      # 当前最佳
├── model_010000.pt      # 续训中间
├── model_015000.pt      # 续训中间
└── model_020000.pt      # 最终目标 ⭐
```

---

## ⚠️ 注意事项

1. **显存管理**: 保持在6GB左右，8GB预算安全
2. **训练中断**: 可随时Ctrl+C中断，重新运行会从最新checkpoint恢复
3. **评估时机**: 建议等20000步完全训练完成后再评估
4. **对比公平性**: d8_v2训练步数更多，对比时注明

---

## 🎸 期望成果

完成本实验后，我们将得到：
1. ✅ 训练到2万步的深度优先模型
2. ✅ 明确更多训练步数对质量的提升效果
3. ✅ 专门的音符补全能力评估
4. ✅ 完整的四模型对比报告

**最终目标**: 找到alphaTex生成的最优模型配置！

---

**开始时间**: 2026-02-04  
**预计完成**: 2-3小时后  
**状态**: 准备就绪，可以开始训练！
