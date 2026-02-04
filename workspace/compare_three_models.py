#!/usr/bin/env python3
"""
三模型对比评估：d4_baseline vs d8_depth vs d6_wide
联合测试alphaTex生成能力
"""

import sys
import os
import torch
import json
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from nanochat.tokenizer import get_tokenizer
from nanochat.gpt import GPT, GPTConfig
from nanochat.engine import Engine
from nanochat.common import get_base_dir


MODEL_CONFIGS = {
    "d4_baseline": {
        "name": "d4_baseline",
        "desc": "基准模型 (4层, 128维, ~5M)",
        "checkpoint": "data/t1/base_checkpoints/d4_5gb/model_001000.pt",
        "config": {
            "vocab_size": 8192,
            "n_layer": 4,
            "n_embd": 128,
            "n_head": 2,
            "n_kv_head": 2,
            "sequence_len": 512,
        },
        "step": 1000,
        "loss": 1.0,
    },
    "d8_depth": {
        "name": "d8_depth",
        "desc": "实验1-深度优先 (8层, 256维, ~20M)",
        "checkpoint": "data/t1/base_checkpoints/d8_depth/model_005000.pt",
        "config": {
            "vocab_size": 8192,
            "n_layer": 8,
            "n_embd": 256,
            "n_head": 4,
            "n_kv_head": 4,
            "sequence_len": 512,
        },
        "step": 5000,
        "loss": 0.66,
    },
    "d6_wide": {
        "name": "d6_wide",
        "desc": "实验2-宽度优先 (6层, 384维, ~35M)",
        "checkpoint": "data/t1/base_checkpoints/d6_wide/model_005000.pt",
        "config": {
            "vocab_size": 8192,
            "n_layer": 6,
            "n_embd": 384,
            "n_head": 6,
            "n_kv_head": 6,
            "sequence_len": 512,
        },
        "step": 5000,
        "loss": 0.59,
    },
}

PROMPTS = [
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


def load_model(checkpoint_path, config, device="cuda"):
    if not Path(checkpoint_path).exists():
        return None, 0

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model_config = GPTConfig(**config)
        model = GPT(model_config)

        if "model" in checkpoint:
            model.load_state_dict(checkpoint["model"])
        else:
            model.load_state_dict(checkpoint)

        model.to(device)
        model.to(torch.bfloat16)
        model.eval()

        param_count = sum(p.numel() for p in model.parameters())
        return model, param_count
    except Exception as e:
        print(f"  加载失败: {e}")
        return None, 0


def evaluate_generation(generated_text):
    text = generated_text[:300]  # 只检查前300字符

    checks = {
        "has_title": "\\title" in text,
        "has_track": "\\track" in text,
        "has_braces": "{" in text and "}" in text,
        "has_quotes": '"' in text,
        "valid_structure": text.count("{") >= text.count("}"),
        "no_gibberish": sum(1 for c in text if c.isalpha()) > 30,
        "has_instrument": "instrument" in text,
        "has_volume": "volume" in text,
    }

    score = sum(checks.values()) / len(checks)
    return score, checks


def generate_and_evaluate(engine, prompt, max_tokens=100, temperature=0.8):
    try:
        tokens = engine.tokenizer(prompt, prepend="<|bos|>")
        result, _ = engine.generate_batch(
            tokens, num_samples=1, max_tokens=max_tokens, temperature=temperature
        )
        generated = engine.tokenizer.decode(result[0])

        score, checks = evaluate_generation(generated)

        # 提取生成部分
        if generated.startswith(prompt):
            gen_part = generated[len(prompt) :].strip()
        else:
            gen_part = generated

        return {
            "generated": gen_part[:250],
            "score": score,
            "checks": checks,
            "full": generated[:500],
        }
    except Exception as e:
        return {"generated": "", "score": 0, "error": str(e)}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("三模型对比评估: d4_baseline vs d8_depth vs d6_wide")
    print("=" * 80)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {device}")
    print()

    os.environ["NANOCHAT_BASE_DIR"] = "./data/t1"
    tokenizer = get_tokenizer()

    # 加载所有模型
    print("加载模型...")
    models = {}
    for name, info in MODEL_CONFIGS.items():
        model, params = load_model(info["checkpoint"], info["config"], device)
        if model:
            models[name] = {
                "model": model,
                "engine": Engine(model, tokenizer),
                "info": info,
                "params": params,
            }
            print(
                f"  ✓ {name}: {params / 1e6:.1f}M参数 (step {info['step']}, loss {info['loss']})"
            )
        else:
            print(f"  ✗ {name}: 未找到checkpoint")

    if len(models) < 2:
        print("\n错误: 至少需要2个模型进行对比")
        return

    print(f"\n✓ 成功加载 {len(models)} 个模型")
    print()

    # 评估
    print("=" * 80)
    print("开始对比评估")
    print("=" * 80)

    results = {name: [] for name in models.keys()}

    for i, (prompt, desc) in enumerate(PROMPTS, 1):
        print(f"\n{'─' * 80}")
        print(
            f'[{i}/{len(PROMPTS)}] {desc}: "{prompt[:40]}{"..." if len(prompt) > 40 else ""}"'
        )
        print(f"{'─' * 80}\n")

        for model_name in models.keys():
            model_data = models[model_name]
            print(f"【{model_name}】{model_data['info']['desc']}")

            result = generate_and_evaluate(model_data["engine"], prompt)
            results[model_name].append(result)

            gen_text = result.get("generated", "")[:150]
            score = result.get("score", 0)
            checks = result.get("checks", {})

            print(f"  生成: {gen_text}{'...' if len(gen_text) > 150 else ''}")
            print(f"  评分: {score:.1%}", end="")
            if checks:
                passed = sum(checks.values())
                print(f" ({passed}/{len(checks)}项通过)")
            else:
                print()
            print()

    # 汇总
    print("\n" + "=" * 80)
    print("评估汇总")
    print("=" * 80)
    print()

    summary = []
    for model_name, model_results in results.items():
        avg_score = sum(r.get("score", 0) for r in model_results) / len(model_results)
        total_checks = sum(sum(r.get("checks", {}).values()) for r in model_results)
        max_checks = sum(len(r.get("checks", {})) for r in model_results)

        model_info = models[model_name]["info"]

        summary.append(
            {
                "name": model_name,
                "desc": model_info["desc"],
                "params": models[model_name]["params"] / 1e6,
                "step": model_info["step"],
                "train_loss": model_info["loss"],
                "avg_score": avg_score,
                "check_ratio": total_checks / max_checks if max_checks > 0 else 0,
            }
        )

        print(f"【{model_name}】")
        print(f"  {model_info['desc']}")
        print(f"  参数量: {models[model_name]['params'] / 1e6:.1f}M")
        print(f"  训练: step {model_info['step']}, loss {model_info['loss']}")
        print(f"  生成质量: {avg_score:.1%}")
        print(
            f"  检查通过: {total_checks}/{max_checks} ({total_checks / max_checks:.1%})"
        )
        print()

    # 排名
    print("=" * 80)
    print("模型排名 (按生成质量)")
    print("=" * 80)
    print()

    sorted_summary = sorted(summary, key=lambda x: x["avg_score"], reverse=True)

    for i, s in enumerate(sorted_summary, 1):
        print(f"{i}. {s['name']}")
        print(f"   质量: {s['avg_score']:.1%} | 训练Loss: {s['train_loss']}")
        print(f"   参数: {s['params']:.1f}M | Step: {s['step']}")
        print(f"   {s['desc']}")
        print()

    # 保存结果
    output_file = Path(__file__).parent / "comparison_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "models": {
                    k: {**v["info"], "params": v["params"]} for k, v in models.items()
                },
                "results": results,
                "summary": sorted_summary,
            },
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )

    print(f"✓ 详细结果已保存: {output_file}")
    print()

    # 结论
    print("=" * 80)
    print("结论")
    print("=" * 80)
    print()

    best_model = sorted_summary[0]
    print(f"最佳模型: {best_model['name']}")
    print(f"  - 生成质量: {best_model['avg_score']:.1%}")
    print(f"  - 训练Loss: {best_model['train_loss']}")
    print(f"  - 参数量: {best_model['params']:.1f}M")
    print()

    print("观察:")
    if len(sorted_summary) >= 2:
        diff = sorted_summary[0]["avg_score"] - sorted_summary[1]["avg_score"]
        if diff > 0.1:
            print(f"  - {sorted_summary[0]['name']} 明显优于其他模型 (领先{diff:.1%})")
        elif diff > 0.05:
            print(f"  - {sorted_summary[0]['name']} 略微领先 ({diff:.1%})")
        else:
            print(f"  - 各模型表现接近，差异不大 ({diff:.1%})")

    # 深度vs宽度
    d8 = next((s for s in summary if s["name"] == "d8_depth"), None)
    d6 = next((s for s in summary if s["name"] == "d6_wide"), None)

    if d8 and d6:
        if d8["avg_score"] > d6["avg_score"]:
            print(
                f"  - 深度优先(d8) > 宽度优先(d6) ({d8['avg_score']:.1%} vs {d6['avg_score']:.1%})"
            )
        elif d6["avg_score"] > d8["avg_score"]:
            print(
                f"  - 宽度优先(d6) > 深度优先(d8) ({d6['avg_score']:.1%} vs {d8['avg_score']:.1%})"
            )
        else:
            print(f"  - 深度和宽度效果相当")

    print()


if __name__ == "__main__":
    main()
