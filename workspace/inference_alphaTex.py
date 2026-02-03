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
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = GPTConfig(
        vocab_size=8192,
        n_layer=4,
        n_embd=128,
        n_head=2,
        n_kv_head=2,
        sequence_len=512,
        window_pattern="L",
    )

    model = GPT(config)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.to(torch.bfloat16)
    model.eval()

    engine = Engine(model, tokenizer)

    print(f"✓ 模型加载完成")
    print(f"  参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    return engine


def generate_text(engine, prompt, max_tokens=50, temperature=0.8):
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

    os.environ["NANOCHAT_BASE_DIR"] = "./data/t1"

    base_dir = get_base_dir()
    checkpoint_path = Path(base_dir) / "base_checkpoints" / "d4_5gb" / "model_001000.pt"

    if not checkpoint_path.exists():
        checkpoint_path = (
            Path(base_dir) / "base_checkpoints" / "d4_5gb" / "model_000200.pt"
        )
        if not checkpoint_path.exists():
            print(f"错误: 未找到checkpoint")
            return

    print(f"使用checkpoint: {checkpoint_path.name}")
    print("")

    tokenizer = get_tokenizer()
    print(f"✓ Tokenizer加载完成 (词表: {tokenizer.get_vocab_size()})")
    print("")

    engine = load_model_and_engine(checkpoint_path, tokenizer, device)

    prompts = [
        '\\title "',
        '\\track ("Guitar" "Standard Tuning")',
        '\\artist "',
        "Song Title: ",
        "Track 1: ",
        "Guitar tab for ",
        "\\ts 4 4",
        "\\tempo 120",
        "|",
        '\\title "My Song"',
        '\\track ("Guitar") {',
    ]

    print("\n" + "=" * 60)
    print("开始生成测试 (alphaTex风格)")
    print("=" * 60)

    for i, prompt in enumerate(prompts, 1):
        print(f'\n[{i}/{len(prompts)}] Prompt: "{prompt}"')
        print("-" * 60)

        try:
            generated = generate_text(engine, prompt, max_tokens=50, temperature=0.8)
            print(f"生成结果:\n{generated}")
        except Exception as e:
            print(f"生成失败: {e}")
            import traceback

            traceback.print_exc()

        print("=" * 60)

    print("\n✓ 生成测试完成")


if __name__ == "__main__":
    main()
