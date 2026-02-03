#!/bin/bash
# 续训脚本: 从step 100训练到step 200
# 使用GPU加速

cd "$(dirname "$0")/.."  # 切换到项目根目录

set -e

echo "=============================================================="
echo "nanochat 续训 (Step 100 -> 200)"
echo "=============================================================="
echo ""

# 激活虚拟环境
source .venv/bin/activate

# 检查GPU
python -c "import torch; print(f'✓ PyTorch: {torch.__version__}'); print(f'✓ CUDA可用: {torch.cuda.is_available()}'); print(f'✓ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')" 2>/dev/null || echo "⚠️  PyTorch检查失败，继续执行..."
echo ""

# 配置环境
export NANOCHAT_BASE_DIR="./data/t1"
export OMP_NUM_THREADS=4
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export WANDB_MODE=offline

# 检查checkpoint是否存在
if [ ! -f "$NANOCHAT_BASE_DIR/base_checkpoints/d4_5gb/model_000100.pt" ]; then
    echo "✗ 错误: 未找到step 100的checkpoint"
    echo "  期望路径: $NANOCHAT_BASE_DIR/base_checkpoints/d4_5gb/model_000100.pt"
    exit 1
fi

echo "✓ 找到checkpoint: model_000100.pt"
echo ""

# 续训参数
echo "=============================================================="
echo "续训配置:"
echo "=============================================================="
echo "  - 起始步数: 100"
echo "  - 目标步数: 200"
echo "  - 额外训练: 100 iterations"
echo "  - 模型: d4_5gb"
echo "  - 设备: GPU (如果可用)"
echo ""

# 运行续训
python -m scripts.base_train \
    --resume-from-step=100 \
    --num-iterations=200 \
    --depth=4 \
    --aspect-ratio=32 \
    --head-dim=64 \
    --max-seq-len=512 \
    --window-pattern=L \
    --device-batch-size=2 \
    --total-batch-size=4096 \
    --eval-every=50 \
    --eval-tokens=4096 \
    --core-metric-every=-1 \
    --sample-every=50 \
    --save-every=50 \
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
echo "最终模型: $NANOCHAT_BASE_DIR/base_checkpoints/d4_5gb/model_000200.pt"
echo ""
