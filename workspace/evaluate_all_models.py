#!/usr/bin/env python3
"""
联合评估脚本：对比三个模型在alphaTex生成任务上的表现
模型：d4_5gb (基准) / d8_depth (实验1) / d6_wide (实验2)
"""

import os
import sys
import json
import torch
from pathlib import Path

# 添加项目根目录到路径
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from nanochat.gpt import GPT, GPTConfig
from nanochat.tokenizer import RustBPETokenizer

# 模型配置
MODELS = {
    "d4_baseline": {
        "checkpoint": "data/t1/base_checkpoints/d4_5gb/model_001000.pt",
        "config": {
            "vocab_size": 8192,
            "n_layer": 4,
            "n_embd": 128,
            "n_head": 2,
            "n_kv_head": 2,
            "sequence_len": 512,
        },
        "desc": "当前模型 (4层, 128维, ~5M)",
    },
    "d8_depth": {
        "checkpoint": "data/t1/base_checkpoints/d8_depth/model_005000.pt",
        "config": {
            "vocab_size": 8192,
            "n_layer": 8,
            "n_embd": 256,
            "n_head": 4,
            "n_kv_head": 4,
            "sequence_len": 512,
        },
        "desc": "实验1-深度优先 (8层, 256维, ~20M)",
    },
    "d6_wide": {
        "checkpoint": "data/t1/base_checkpoints/d6_wide/model_005000.pt",
        "config": {
            "vocab_size": 8192,
            "n_layer": 6,
            "n_embd": 384,
            "n_head": 6,
            "n_kv_head": 6,
            "sequence_len": 512,
        },
        "desc": "实验2-宽度优先 (6层, 384维, ~35M)",
    },
}

# alphaTex风格Prompts
PROMPTS = [
    # 基础结构
    ("标题", '\title "'),
    ("音轨", '\track ("Guitar" "Standard Tuning")'),
    ("艺术家", '\artist "'),
    # 音乐内容
    ("歌曲标题", "Song Title: "),
    ("音轨1", "Track 1: "),
    ("吉他谱", "Guitar tab for "),
    # alphaTex语法
    ("拍号", "\ts 4 4"),
    ("速度", "\tempo 120"),
    ("小节线", "|"),
    # 混合prompt
    ("完整开头", '\title "My Song"\n\track ("Guitar") {'),
    ("调弦", "\tuning (E4 B3 G3 D3 A2 E2)"),
]


def load_model(checkpoint_path, config):
    """加载模型和checkpoint"""
    if not os.path.exists(checkpoint_path):
        return None

    try:
        # 创建配置
        model_config = GPTConfig(
            vocab_size=config["vocab_size"],
            n_layer=config["n_layer"],
            n_embd=config["n_embd"],
            n_head=config["n_head"],
            n_kv_head=config["n_kv_head"],
            sequence_len=config["sequence_len"],
            window_pattern="L",
        )

        # 加载模型
        model = GPT(model_config)

        # 加载权重
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)

        model.eval()

        # 获取参数数量
        param_count = sum(p.numel() for p in model.parameters())

        return model, param_count
    except Exception as e:
        print(f"  加载失败: {e}")
        return None


def generate_text(model, tokenizer, prompt, max_tokens=200, temperature=0.8):
    """生成文本"""
    device = next(model.parameters()).device

    # 编码prompt
    tokens = tokenizer.encode(prompt, bos=True, eos=False)
    input_tensor = torch.tensor([tokens], dtype=torch.long, device=device)

    # 生成
    with torch.no_grad():
        with torch.autocast(
            device_type="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.bfloat16,
        ):
            for _ in range(max_tokens):
                logits = model(input_tensor)
                logits = logits[:, -1, :] / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_tensor = torch.cat([input_tensor, next_token], dim=1)

                # 检查结束符
                if next_token.item() == tokenizer.get_bos_token_id():
                    break

    # 解码
    generated = tokenizer.decode(input_tensor[0].tolist())
    return generated


def evaluate_syntax_quality(text):
    """评估alphaTex语法质量"""
    score = 0
    checks = {
        "has_title": "\title" in text,
        "has_track": "\track" in text,
        "has_braces": "{" in text and "}" in text,
        "has_quotes": '"' in text,
        "valid_tokens": all(c.isprintable() or c.isspace() for c in text[:100]),
        "no_gibberish": sum(1 for c in text[:100] if c.isalpha()) > 20,
    }
    return sum(checks.values()) / len(checks), checks


def main():
    print("=" * 70)
    print("alphaTex模型联合评估")
    print("=" * 70)
    print()

    # 加载tokenizer
    tokenizer_path = project_root / "data" / "t1" / "tokenizer" / "tokenizer.pkl"
    if not tokenizer_path.exists():
        print(f"错误: 未找到tokenizer: {tokenizer_path}")
        return

    tokenizer = RustBPETokenizer.from_directory(str(tokenizer_path.parent))
    print(f"✓ Tokenizer加载成功 (vocab_size={tokenizer.get_vocab_size()})")
    print()

    # 加载所有可用模型
    print("加载模型...")
    loaded_models = {}
    for name, info in MODELS.items():
        result = load_model(info["checkpoint"], info["config"])
        if result:
            model, param_count = result
            loaded_models[name] = {"model": model, "params": param_count, "info": info}
            print(f"  ✓ {name}: {param_count / 1e6:.1f}M参数")
        else:
            print(f"  ✗ {name}: 未找到checkpoint")

    if not loaded_models:
        print("\n错误: 没有可用的模型进行测试")
        return

    print(f"\n✓ 成功加载 {len(loaded_models)} 个模型")
    print()

    # 对每个prompt进行测试
    print("=" * 70)
    print("生成测试")
    print("=" * 70)
    print()

    results = {name: [] for name in loaded_models.keys()}

    for prompt_name, prompt_text in PROMPTS:
        print(f"\n{'─' * 70}")
        print(
            f"Prompt: [{prompt_name}] {prompt_text[:40]}{'...' if len(prompt_text) > 40 else ''}"
        )
        print(f"{'─' * 70}\n")

        for model_name, model_data in loaded_models.items():
            print(f"【{model_name}】{model_data['info']['desc']}")

            # 生成
            try:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = model_data["model"].to(device)
                generated = generate_text(model, tokenizer, prompt_text, max_tokens=150)

                # 提取生成的部分（去掉prompt）
                gen_part = generated[len(prompt_text) :].strip()
                display_text = gen_part[:200] + ("..." if len(gen_part) > 200 else "")

                # 评估质量
                quality_score, checks = evaluate_syntax_quality(generated)

                print(f"  生成: {display_text}")
                print(
                    f"  质量评分: {quality_score:.1%} (语法检查: {sum(checks.values())}/{len(checks)})"
                )

                results[model_name].append(
                    {
                        "prompt": prompt_name,
                        "generated": gen_part,
                        "quality": quality_score,
                        "checks": checks,
                    }
                )

            except Exception as e:
                print(f"  生成失败: {e}")
                results[model_name].append(
                    {
                        "prompt": prompt_name,
                        "generated": "",
                        "quality": 0,
                        "error": str(e),
                    }
                )

            print()

    # 汇总报告
    print("\n" + "=" * 70)
    print("评估汇总")
    print("=" * 70)
    print()

    summary = []
    for model_name, model_results in results.items():
        avg_quality = sum(r["quality"] for r in model_results) / len(model_results)
        total_params = loaded_models[model_name]["params"] / 1e6

        # 统计各项检查通过率
        check_totals = {}
        for r in model_results:
            for check_name, passed in r.get("checks", {}).items():
                check_totals[check_name] = check_totals.get(check_name, 0) + int(passed)

        summary.append(
            {
                "name": model_name,
                "desc": MODELS[model_name]["desc"],
                "params": total_params,
                "avg_quality": avg_quality,
                "check_totals": check_totals,
            }
        )

        print(f"【{model_name}】")
        print(f"  描述: {MODELS[model_name]['desc']}")
        print(f"  参数量: {total_params:.1f}M")
        print(f"  平均质量: {avg_quality:.1%}")
        print(f"  各项检查通过:")
        for check_name, count in sorted(check_totals.items()):
            print(f"    - {check_name}: {count}/{len(model_results)}")
        print()

    # 排名
    print("=" * 70)
    print("模型排名 (按平均质量)")
    print("=" * 70)
    print()

    sorted_summary = sorted(summary, key=lambda x: x["avg_quality"], reverse=True)
    for i, s in enumerate(sorted_summary, 1):
        print(f"{i}. {s['name']}")
        print(f"   质量: {s['avg_quality']:.1%} | 参数: {s['params']:.1f}M")
        print(f"   {s['desc']}")
        print()

    # 保存详细结果
    output_file = project_root / "workspace" / "evaluation_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "models": {
                    k: {"desc": v["info"]["desc"], "params": v["params"]}
                    for k, v in loaded_models.items()
                },
                "results": results,
                "summary": [
                    {k: v for k, v in s.items() if k != "check_totals"} for s in summary
                ],
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"✓ 详细结果已保存: {output_file}")
    print()


if __name__ == "__main__":
    main()
