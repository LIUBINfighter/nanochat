#!/bin/bash
# 续训脚本：Step 200 -> 1000
# 禁用通用采样，专注alphaTex训练

cd "$(dirname "$0")/.."

set -e

echo "=============================================================="
echo "nanochat 续训 (Step 200 -> 1000)"
echo "模式：无采样 / GPU加速 / 5GB显存优化"
echo "=============================================================="
echo ""

source .venv/bin/activate

export NANOCHAT_BASE_DIR="./data/t1"
export OMP_NUM_THREADS=4
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export WANDB_MODE=offline
export WANDB_DISABLED=true

if [ ! -f "$NANOCHAT_BASE_DIR/base_checkpoints/d4_5gb/model_000200.pt" ]; then
    echo "错误: 未找到step 200的checkpoint"
    exit 1
fi

echo "✓ checkpoint: model_000200.pt"
echo "✓ GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""
echo "配置:"
echo "  - 训练步数: 200 -> 1000 (800 iterations)"
echo "  - 采样: 禁用 (--sample-every=-1)"
echo "  - 验证: 禁用 (--eval-every=-1)"
echo "  - 保存: 每100步"
echo "  - 预计时间: ~15分钟"
echo ""

python -m scripts.base_train \
    --resume-from-step=200 \
    --num-iterations=1000 \
    --depth=4 \
    --aspect-ratio=32 \
    --head-dim=64 \
    --max-seq-len=512 \
    --window-pattern=L \
    --device-batch-size=4 \
    --total-batch-size=4096 \
    --sample-every=-1 \
    --eval-every=-1 \
    --core-metric-every=-1 \
    --save-every=100 \
    --embedding-lr=0.3 \
    --matrix-lr=0.02 \
    --unembedding-lr=0.004 \
    --weight-decay=0.1 \
    --warmup-ratio=0.0 \
    --warmdown-ratio=0.3 \
    --run="dummy" \
    --model-tag="d4_5gb"

echo ""
echo "✓ 续训完成: model_001000.pt"
