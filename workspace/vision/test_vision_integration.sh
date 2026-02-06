#!/bin/bash
# 视觉层集成测试脚本
# 测试多模态模型架构是否正确

cd "$(dirname "$0")"

echo "=============================================================="
echo "视觉层集成测试"
echo "目的: 验证视觉编码器、融合层、文本模型整合是否正常"
echo "=============================================================="
echo ""

source ../../.venv/bin/activate

export NANOCHAT_BASE_DIR="../../data/t1"
export OMP_NUM_THREADS=4

echo "✓ 环境设置完成"
echo ""

# 运行测试
python test_integration.py

exit_code=$?

echo ""
echo "=============================================================="
if [ $exit_code -eq 0 ]; then
    echo "✓ 所有测试通过！视觉层集成就绪"
    echo ""
    echo "下一步:"
    echo "  1. 准备图像-alphaTex配对数据"
    echo "  2. 运行 train_multimodal.py 开始训练"
    echo "  3. 使用 test_inference.py 测试生成效果"
else
    echo "✗ 测试失败，请检查错误信息"
fi
echo "=============================================================="

exit $exit_code
