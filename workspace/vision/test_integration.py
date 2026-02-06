"""
训练脚本：多模态模型训练
冻结文本模型，只训练视觉相关层
"""

import os
import sys
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
import json

# 添加路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from vision_encoder import VisionEncoder, VisionProjection
from multimodal_model import MultimodalModel
from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import get_tokenizer
from nanochat.engine import Engine


def load_text_model(checkpoint_path, device="cuda"):
    """加载预训练的文本模型"""
    config = GPTConfig(
        vocab_size=8192,
        n_layer=8,
        n_embd=256,
        n_head=4,
        n_kv_head=4,
        sequence_len=512,
        window_pattern="L",
    )

    model = GPT(config)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)
        print(f"✓ Loaded text model from {checkpoint_path}")
    else:
        print(f"⚠ Checkpoint not found: {checkpoint_path}")
        print(f"  Will use randomly initialized text model (for testing)")

    model.to(device)
    model.eval()
    return model


def create_multimodal_model(text_checkpoint_path, device="cuda"):
    """创建多模态模型"""
    # 1. 加载文本模型
    text_model = load_text_model(text_checkpoint_path, device)

    # 2. 创建视觉编码器
    vision_encoder = VisionEncoder(
        img_size=224,
        patch_size=16,
        in_channels=3,
        embed_dim=256,
        num_layers=3,
        num_heads=4,
    ).to(device)

    # 3. 创建多模态模型
    multimodal_model = MultimodalModel(
        vision_encoder=vision_encoder,
        text_model=text_model,
        num_fusion_layers=3,
        
        
    ).to(device)

    return multimodal_model


def test_forward_pass(model, device="cuda"):
    """测试前向传播"""
    print("\nTesting forward pass...")

    # 创建假数据
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    text_tokens = torch.randn(batch_size, 50, 256).to(device)

    # 前向传播
    model.eval()
    with torch.no_grad():
        output = model(images, text_tokens)

    print(f"✓ Forward pass successful")
    print(f"  Input images: {images.shape}")
    print(f"  Input text: {text_tokens.shape}")
    print(f"  Output: {output.shape}")

    return True


def test_trainable_params(model):
    """测试可训练参数"""
    print("\nChecking trainable parameters...")

    stats = model.count_parameters()
    print(f"  Trainable: {stats['trainable_M']:.2f}M")
    print(f"  Frozen: {stats['frozen_M']:.2f}M")
    print(f"  Total: {stats['total_M']:.2f}M")

    # 验证冻结状态
    trainable_count = sum(1 for p in model.parameters() if p.requires_grad)
    frozen_count = sum(1 for p in model.parameters() if not p.requires_grad)

    print(f"  Trainable param groups: {trainable_count}")
    print(f"  Frozen param groups: {frozen_count}")

    # 确保文本模型被冻结
    text_model_frozen = all(not p.requires_grad for p in model.text_model.parameters())
    print(f"  Text model frozen: {text_model_frozen}")

    # 确保视觉层可训练
    vision_trainable = any(p.requires_grad for p in model.vision_encoder.parameters())
    print(f"  Vision encoder trainable: {vision_trainable}")

    return text_model_frozen and vision_trainable


def test_training_step(model, device="cuda"):
    """测试训练步骤"""
    print("\nTesting training step...")

    model.train()

    # 创建假数据
    batch_size = 2
    images = torch.randn(batch_size, 3, 224, 224).to(device)
    text_tokens = torch.randn(batch_size, 50, 256).to(device)
    targets = torch.randn(batch_size, 50, 256).to(device)

    # 优化器（只优化可训练参数）
    optimizer = torch.optim.AdamW(
        model.get_trainable_params(), lr=1e-4, weight_decay=0.01
    )

    # 前向传播
    output = model(images, text_tokens)

    # 计算损失（简化的MSE损失）
    loss = nn.functional.mse_loss(output, targets)

    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(f"✓ Training step successful")
    print(f"  Loss: {loss.item():.4f}")

    # 验证文本模型参数未被更新
    text_model_updated = any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in model.text_model.parameters()
    )

    if text_model_updated:
        print(f"  ⚠ Warning: Text model parameters were updated!")
        return False
    else:
        print(f"  ✓ Text model parameters frozen (as expected)")
        return True


def test_vision_only(model, device="cuda"):
    """测试仅视觉输入"""
    print("\nTesting vision-only mode...")

    images = torch.randn(2, 3, 224, 224).to(device)

    model.eval()
    with torch.no_grad():
        vision_tokens = model(images, text_tokens=None)

    print(f"✓ Vision-only mode works")
    print(f"  Output shape: {vision_tokens.shape}")
    print(f"  Expected: (2, 196, 256)")

    return vision_tokens.shape == (2, 196, 256)


def test_text_only(model, device="cuda"):
    """测试仅文本输入"""
    print("\nTesting text-only mode...")

    text_tokens = torch.randn(2, 50, 256).to(device)

    model.eval()
    with torch.no_grad():
        output = model(images=None, text_tokens=text_tokens)

    print(f"✓ Text-only mode works")
    print(f"  Output shape: {output.shape}")

    return True


def main():
    print("=" * 70)
    print("Multimodal Model Integration Test")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print()

    # 设置环境
    os.environ["NANOCHAT_BASE_DIR"] = "./data/t1"

    # 最佳模型路径
    text_checkpoint = "data/t1/base_checkpoints/d8_depth/model_005000.pt"

    # 创建模型
    print("Creating multimodal model...")
    try:
        model = create_multimodal_model(text_checkpoint, device)
        print("✓ Model created successfully")
    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        import traceback

        traceback.print_exc()
        return False

    # 运行测试
    tests = [
        ("Forward Pass", lambda: test_forward_pass(model, device)),
        ("Trainable Params", lambda: test_trainable_params(model)),
        ("Training Step", lambda: test_training_step(model, device)),
        ("Vision Only", lambda: test_vision_only(model, device)),
        ("Text Only", lambda: test_text_only(model, device)),
    ]

    results = []
    print("\n" + "=" * 70)
    print("Running Tests")
    print("=" * 70)

    for test_name, test_func in tests:
        print(f"\n[{test_name}]")
        try:
            result = test_func()
            results.append((test_name, result))
            print(f"Result: {'✓ PASS' if result else '✗ FAIL'}")
        except Exception as e:
            print(f"✗ FAIL with exception: {e}")
            import traceback

            traceback.print_exc()
            results.append((test_name, False))

    # 汇总
    print("\n" + "=" * 70)
    print("Test Summary")
    print("=" * 70)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {test_name:20s}: {status}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Ready for vision training.")
        return True
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
