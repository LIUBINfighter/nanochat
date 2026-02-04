#!/bin/bash
# 实验1: 深度优先 (d8_depth)
# 对比当前d4模型，深度从4层增加到8层，测试深度对alphaTex学习的影响

cd "$(dirname "$0")/.."

set -e

echo "=============================================================="
echo "实验1: 深度优先模型 (d8_depth)"
echo "配置: 8层, 256维, ~20M参数, 8GB显存"
echo "=============================================================="
echo ""

source .venv/bin/activate

export NANOCHAT_BASE_DIR="./data/t1"
export OMP_NUM_THREADS=4
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export WANDB_MODE=offline
export WANDB_DISABLED=true

echo "✓ GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo "✓ 显存限制: 8GB"
echo ""
echo "模型配置:"
echo "  - 层数: 8 (当前d4的2倍)"
echo "  - 维度: 256 (8 * 32)"
echo "  - 注意力头: 4 (256 / 64)"
echo "  - 参数量: ~20M"
echo "  - 序列长度: 512"
echo ""
echo "训练配置:"
echo "  - 训练步数: 5000 (快速验证)"
echo "  - Batch: 4 (保持总batch 4096)"
echo "  - 预计显存: ~6-7GB"
echo "  - 预计时间: ~2小时"
echo ""

python -m scripts.base_train \
    --depth=8 \
    --aspect-ratio=32 \
    --head-dim=64 \
    --max-seq-len=512 \
    --window-pattern=L \
    --device-batch-size=4 \
    --total-batch-size=4096 \
    --sample-every=-1 \
    --eval-every=-1 \
    --core-metric-every=-1 \
    --save-every=500 \
    --num-iterations=5000 \
    --embedding-lr=0.25 \
    --matrix-lr=0.018 \
    --unembedding-lr=0.0035 \
    --weight-decay=0.12 \
    --warmup-ratio=0.05 \
    --warmdown-ratio=0.25 \
    --run="exp1_d8_depth" \
    --model-tag="d8_depth"

echo ""
echo "✓ 实验1完成: d8_depth模型已保存"
echo "✓ Checkpoint: data/t1/base_checkpoints/d8_depth/model_005000.pt"
