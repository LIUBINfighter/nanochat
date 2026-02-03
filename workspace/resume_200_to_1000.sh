#!/bin/bash
# 续训脚本: 从step 200训练到step 1000
# 使用GPU加速，5GB显存优化配置

cd "$(dirname "$0")/.."  # 切换到项目根目录

set -e

echo "=============================================================="
echo "nanochat 续训 (Step 200 -> 1000)"
echo "=============================================================="
echo ""

# 激活虚拟环境
source .venv/bin/activate

# 配置环境
export NANOCHAT_BASE_DIR="./data/t1"
export OMP_NUM_THREADS=4
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export WANDB_MODE=offline
export WANDB_DISABLED=true

# 检查checkpoint是否存在
if [ ! -f "$NANOCHAT_BASE_DIR/base_checkpoints/d4_5gb/model_000200.pt" ]; then
    echo "✗ 错误: 未找到step 200的checkpoint"
    echo "  期望路径: $NANOCHAT_BASE_DIR/base_checkpoints/d4_5gb/model_000200.pt"
    exit 1
fi

echo "✓ 找到checkpoint: model_000200.pt"
echo ""

# 检查GPU
echo "[GPU状态]"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader
echo ""

# 续训参数
echo "=============================================================="
echo "续训配置:"
echo "=============================================================="
echo "  - 起始步数: 200"
echo "  - 目标步数: 1000"
echo "  - 额外训练: 800 iterations"
echo "  - 模型: d4_5gb (4层, 128维)"
echo "  - 设备: GPU (CUDA)"
echo "  - 预计时间: ~15-20分钟"
echo ""
echo "优化参数:"
echo "  - batch-size: 4 (per device)"
echo "  - total-batch: 4096"
echo "  - seq-len: 512"
echo "  - 学习率: 继承当前"
echo "  - warmup: 0% (已预热过)"
echo "  - warmdown: 30% (最后300步)"
echo ""

# 运行续训
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
    --eval-every=100 \
    --eval-tokens=8192 \
    --core-metric-every=-1 \
    --sample-every=100 \
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
echo "=============================================================="
echo "✓ 续训完成!"
echo "=============================================================="
echo ""
echo "最终模型: $NANOCHAT_BASE_DIR/base_checkpoints/d4_5gb/model_001000.pt"
echo ""
