#!/usr/bin/env python3
import sys
import os
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.tokenizer import get_tokenizer
from nanochat.gpt import GPT, GPTConfig
from nanochat.engine import Engine
from nanochat.common import get_base_dir


def load_model_and_engine(checkpoint_path, tokenizer, device="cuda"):
    print(f"加载模型: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # 使用实验1配置: d8_depth (8层, 256维)
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

    # 兼容不同的checkpoint格式
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.to(torch.bfloat16)
    model.eval()

    engine = Engine(model, tokenizer)

    print(f"✓ 模型加载完成")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"  配置: 8层, 256维, 4头")

    return engine


def generate_text(engine, prompt, max_tokens=80, temperature=0.8):
    tokens = engine.tokenizer(prompt, prepend="<|bos|>")
    result, _ = engine.generate_batch(
        tokens, num_samples=1, max_tokens=max_tokens, temperature=temperature
    )
    generated_text = engine.tokenizer.decode(result[0])
    return generated_text


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    print("=" * 60)
    print("实验1 (d8_depth) 生成效果测试")
    print("=" * 60)

    os.environ["NANOCHAT_BASE_DIR"] = "./data/t1"

    base_dir = get_base_dir()
    checkpoint_path = (
        Path(base_dir) / "base_checkpoints" / "d8_depth" / "model_005000.pt"
    )

    if not checkpoint_path.exists():
        print(f"错误: 未找到实验1的checkpoint: {checkpoint_path}")
        return

    print(f"✓ 使用checkpoint: {checkpoint_path.name}")
    print(f"✓ 训练步数: 5000")
    print(f"✓ 最终Loss: 0.664")
    print("")

    tokenizer = get_tokenizer()
    print(f"✓ Tokenizer加载完成 (词表: {tokenizer.get_vocab_size()})")
    print("")

    engine = load_model_and_engine(checkpoint_path, tokenizer, device)

    prompts = [
        ('\\title "', "标题定义"),
        ('\\track ("Guitar" "Standard Tuning")', "音轨定义"),
        ('\\artist "', "艺术家"),
        ("Song Title: ", "歌曲标题"),
        ("Track 1: ", "音轨1"),
        ("Guitar tab for ", "吉他谱"),
        ("\\ts 4 4", "拍号"),
        ("\\tempo 120", "速度"),
        ("|", "小节线"),
        ('\\title "My Song"\\n\\track ("Guitar") {', "完整开头"),
        ("\\tuning (E4 B3 G3 D3 A2 E2)", "调弦"),
    ]

    print("\n" + "=" * 60)
    print("开始生成测试 (alphaTex风格)")
    print("=" * 60)

    for i, (prompt, desc) in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] {desc}")
        print(f'Prompt: "{prompt}"')
        print("-" * 60)

        try:
            generated = generate_text(engine, prompt, max_tokens=80, temperature=0.8)
            # 提取生成的部分（去掉prompt）
            if generated.startswith(prompt):
                gen_part = generated[len(prompt) :].strip()
            else:
                gen_part = generated
            print(f"生成:\n{gen_part[:200]}{'...' if len(gen_part) > 200 else ''}")

            # 简单质量评估
            has_title = "\\title" in generated
            has_track = "\\track" in generated
            has_braces = "{" in generated and "}" in generated
            print(
                f"[质量: {'✓' if has_title else '✗'}title {'✓' if has_track else '✗'}track {'✓' if has_braces else '✗'}braces]"
            )

        except Exception as e:
            print(f"生成失败: {e}")
            import traceback

            traceback.print_exc()

        print("=" * 60)

    print("\n✓ 实验1 (d8_depth) 生成测试完成")
    print(f"  Checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
