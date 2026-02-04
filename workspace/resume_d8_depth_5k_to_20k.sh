#!/bin/bash
# 续训脚本：实验1 d8_depth 从 Step 5000 -> 20000
# 深度优先策略，继续训练提升alphaTex生成能力

cd "$(dirname "$0")/.."

set -e

echo "=============================================================="
echo "续训: d8_depth (Step 5000 -> 20000)"
echo "策略: 深度优先 | 目标: 高质量alphaTex生成 + 音符补全"
echo "=============================================================="
echo ""

source .venv/bin/activate

export NANOCHAT_BASE_DIR="./data/t1"
export OMP_NUM_THREADS=4
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export WANDB_MODE=offline
export WANDB_DISABLED=true

# 检查checkpoint
if [ ! -f "$NANOCHAT_BASE_DIR/base_checkpoints/d8_depth/model_005000.pt" ]; then
    echo "错误: 未找到step 5000的checkpoint"
    exit 1
fi

echo "✓ checkpoint: model_005000.pt (当前最佳模型)"
echo "✓ GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
echo ""
echo "模型配置:"
echo "  - 层数: 8 (深度优先)"
echo "  - 维度: 256"
echo "  - 参数量: ~20M"
echo ""
echo "训练配置:"
echo "  - 训练步数: 5000 -> 20000 (15000 iterations)"
echo "  - Batch: 4"
echo "  - 保存: 每1000步"
echo "  - 预计时间: ~1.5-2小时"
echo "  - 预计显存: ~6GB (8GB预算内)"
echo ""

python -m scripts.base_train \
    --resume-from-step=5000 \
    --num-iterations=20000 \
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
    --save-every=1000 \
    --embedding-lr=0.22 \
    --matrix-lr=0.016 \
    --unembedding-lr=0.003 \
    --weight-decay=0.12 \
    --warmup-ratio=0.0 \
    --warmdown-ratio=0.25 \
    --final-lr-frac=0.05 \
    --run="exp1_d8_depth_v2" \
    --model-tag="d8_depth_v2"

echo ""
echo "✓ 续训完成: d8_depth_v2 模型已保存"
echo "✓ 最终Checkpoint: data/t1/base_checkpoints/d8_depth/model_020000.pt"
