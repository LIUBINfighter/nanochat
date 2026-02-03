#!/bin/bash
# 5GB显存友好的训练脚本

set -e  # 遇到错误立即退出

echo "=============================================================="
echo "nanochat 5GB显存训练流程"
echo "=============================================================="
echo ""

# 检查显存
echo "[系统检查]"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# 配置环境
export NANOCHAT_BASE_DIR="./data/t1"
export OMP_NUM_THREADS=1
export PYTORCH_ALLOC_CONF="expandable_segments:True"

# 确保数据已准备
if [ ! -d "$NANOCHAT_BASE_DIR/base_data" ]; then
    echo "错误: 数据目录不存在，请先运行数据准备脚本"
    exit 1
fi

echo "[配置参数]"
echo "数据目录: $NANOCHAT_BASE_DIR"
echo ""

# ============================================
# 阶段1: Tokenizer训练 (低显存占用)
# ============================================
echo "=============================================================="
echo "阶段1: Tokenizer训练"
echo "=============================================================="
echo "参数: --max-chars=50000000 --vocab-size=8192"
echo ""

# 使用较小的字符数和词表大小以节省内存
python -m scripts.tok_train \
    --max-chars=50000000 \
    --vocab-size=8192 \
    --doc-cap=5000

echo ""
echo "✓ Tokenizer训练完成"
echo ""

# 评估tokenizer
python -m scripts.tok_eval

# ============================================
# 阶段2: 预训练 (5GB显存优化配置)
# ============================================
echo ""
echo "=============================================================="
echo "阶段2: 预训练 (5GB显存优化)"
echo "=============================================================="
echo ""
echo "模型配置:"
echo "  - depth=4 (4层Transformer)"
echo "  - aspect-ratio=32 (小维度)"
echo "  - head-dim=64 (小头维度)"
echo "  - max-seq-len=512 (短序列)"
echo "  - device-batch-size=1 (单样本)"
echo "  - total-batch-size=4096"
echo ""

# 5GB显存友好的配置
python -m scripts.base_train \
    --depth=4 \
    --aspect-ratio=32 \
    --head-dim=64 \
    --max-seq-len=512 \
    --window-pattern=L \
    --device-batch-size=1 \
    --total-batch-size=4096 \
    --num-iterations=2000 \
    --eval-every=200 \
    --eval-tokens=8192 \
    --core-metric-every=-1 \
    --sample-every=500 \
    --save-every=500 \
    --embedding-lr=0.3 \
    --matrix-lr=0.02 \
    --unembedding-lr=0.004 \
    --weight-decay=0.1 \
    --warmup-ratio=0.05 \
    --warmdown-ratio=0.5 \
    --run="d4_5gb_test" \
    --model-tag="d4_5gb"

echo ""
echo "✓ 预训练完成"
echo ""

# 评估模型
echo "=============================================================="
echo "阶段3: 模型评估"
echo "=============================================================="
python -m scripts.base_eval \
    --device-batch-size=1 \
    --split-tokens=8192 \
    --max-per-task=50

echo ""
echo "=============================================================="
echo "🎉 训练流程全部完成!"
echo "=============================================================="
echo ""
echo "输出位置: $NANOCHAT_BASE_DIR/base_checkpoints/d4_5gb/"
echo ""
echo "你可以使用以下命令进行对话测试:"
echo "  python -m scripts.chat_cli -p \"你好\""
echo ""

