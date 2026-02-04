#!/bin/bash
# 实验2: 宽度优先 (d6_wide)
# 对比当前d4模型，增加宽度(维度)，测试表达能力对alphaTex学习的影响

cd "$(dirname "$0")/.."

set -e

echo "=============================================================="
echo "实验2: 宽度优先模型 (d6_wide)"
echo "配置: 6层, 384维, ~35M参数, 8GB显存"
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
echo "  - 层数: 6 (适中深度)"
echo "  - 维度: 384 (6 * 64)"
echo "  - 注意力头: 6 (384 / 64)"
echo "  - 参数量: ~35M"
echo "  - 序列长度: 512"
echo ""
echo "训练配置:"
echo "  - 训练步数: 5000 (快速验证)"
echo "  - Batch: 2 (控制显存)"
echo "  - 预计显存: ~7.5GB"
echo "  - 预计时间: ~2.5小时"
echo ""

python -m scripts.base_train \
    --depth=6 \
    --aspect-ratio=64 \
    --head-dim=64 \
    --max-seq-len=512 \
    --window-pattern=L \
    --device-batch-size=2 \
    --total-batch-size=4096 \
    --sample-every=-1 \
    --eval-every=-1 \
    --core-metric-every=-1 \
    --save-every=500 \
    --num-iterations=5000 \
    --embedding-lr=0.2 \
    --matrix-lr=0.015 \
    --unembedding-lr=0.003 \
    --weight-decay=0.15 \
    --warmup-ratio=0.08 \
    --warmdown-ratio=0.25 \
    --run="exp2_d6_wide" \
    --model-tag="d6_wide"

echo ""
echo "✓ 实验2完成: d6_wide模型已保存"
echo "✓ Checkpoint: data/t1/base_checkpoints/d6_wide/model_005000.pt"
