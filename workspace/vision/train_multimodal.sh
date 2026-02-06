#!/bin/bash
# 多模态模型训练脚本
# 冻结文本模型，训练视觉层和融合层

cd "$(dirname "$0")/.."

echo "=============================================================="
echo "多模态模型训练"
echo "策略: 冻结文本层，训练视觉+融合层"
echo "=============================================================="
echo ""

source ../.venv/bin/activate

export NANOCHAT_BASE_DIR="../data/t1"
export OMP_NUM_THREADS=4
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export WANDB_MODE=offline
export WANDB_DISABLED=true

# 配置
TEXT_CHECKPOINT="../data/t1/base_checkpoints/d8_depth/model_005000.pt"
VISION_DATA_DIR="../data/vision_training"  # 需要创建此目录
BATCH_SIZE=4
LEARNING_RATE=1e-4
NUM_EPOCHS=10
SAVE_EVERY=1000

echo "✓ 使用文本模型: $TEXT_CHECKPOINT"
echo "✓ 批次大小: $BATCH_SIZE"
echo "✓ 学习率: $LEARNING_RATE"
echo "✓ 训练轮数: $NUM_EPOCHS"
echo ""

# 检查数据目录
if [ ! -d "$VISION_DATA_DIR" ]; then
    echo "⚠ 警告: 视觉训练数据目录不存在: $VISION_DATA_DIR"
    echo "  请先准备数据 (参见 VISION_DATA_PREP_GUIDE.md)"
    echo ""
    echo "临时方案: 运行测试模式 (--test-mode)"
    TEST_MODE="--test-mode"
else
    TEST_MODE=""
fi

echo "开始训练..."
echo ""

# 训练命令 (待实现)
# python train_multimodal.py \
#     --text-checkpoint "$TEXT_CHECKPOINT" \
#     --data-dir "$VISION_DATA_DIR" \
#     --batch-size $BATCH_SIZE \
#     --learning-rate $LEARNING_RATE \
#     --num-epochs $NUM_EPOCHS \
#     --save-every $SAVE_EVERY \
#     $TEST_MODE

echo ""
echo "训练脚本模板已准备好"
echo "实际训练前需要:"
echo "  1. 实现 train_multimodal.py"
echo "  2. 准备配对数据"
echo "  3. 配置训练参数"
