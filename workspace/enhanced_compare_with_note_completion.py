#!/usr/bin/env python3
"""
增强版三模型对比评估 + 音符补全能力测试
支持：d4_baseline / d8_depth / d6_wide / d8_depth_v2(2万步)
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
        "desc": "实验1-深度优先 (8层, 256维, ~20M, 5K步)",
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
        "desc": "实验2-宽度优先 (6层, 384维, ~35M, 5K步)",
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
    "d8_depth_v2": {
        "name": "d8_depth_v2",
        "desc": "深度优先v2 (8层, 256维, ~20M, 20K步)",
        "checkpoint": "data/t1/base_checkpoints/d8_depth/model_020000.pt",
        "config": {
            "vocab_size": 8192,
            "n_layer": 8,
            "n_embd": 256,
            "n_head": 4,
            "n_kv_head": 4,
            "sequence_len": 512,
        },
        "step": 20000,
        "loss": None,
    },
}

# 基础结构Prompts
STRUCTURE_PROMPTS = [
    ('\\title "', "标题定义"),
    ('\\track ("Guitar" "Standard Tuning")', "音轨定义"),
    ('\\artist "', "艺术家"),
    ("Song Title: ", "歌曲标题"),
    ("\\tempo 120", "速度"),
    ('\\title "My Song"\\n\\track ("Guitar") {', "完整开头"),
]


# 音符补全测试 - 类型1: 小节线开头
def get_tab_completion_prompts():
    return [
        ("| 3.2", "Tab补全-单音", "简单音符"),
        ("| 3.2 2.3", "Tab补全-双音", "双音符"),
        ("| (3.2 2.3).8", "Tab补全-和弦", "和弦节奏"),
        ("| 3.2 .8", "Tab补全-带节奏", "音符+节奏"),
        ("| 1.2 2.3 3.2 4.2", "Tab补全-序列", "音符序列"),
    ]


# 音符补全测试 - 类型2: 复杂音符语法
def get_note_completion_prompts():
    return [
        ("(6.2 1.5).", "音符-和弦基础", "基础和弦"),
        ("(4.2 1.5{t}).", "音符-装饰音", "带装饰音"),
        ("(6.2 1.5).8\\n(4.2 1.5{t}).", "音符-双行", "连续和弦"),
        ("(3.2 1.5{t}).4{", "音符-带效果", "效果器语法"),
        ("24.1.", "音符-单弦", "单弦音符"),
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


def evaluate_structure(generated_text):
    text = generated_text[:300]
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


def evaluate_tab_completion(generated_text):
    text = generated_text[:200]

    checks = {
        "has_fret_number": any(c.isdigit() for c in text),
        "has_dot_separator": "." in text,
        "valid_format": "|" in text
        or any(f"{a}.{b}" in text for a in range(1, 7) for b in range(1, 7)),
        "has_rhythm": any(x in text for x in [".8", ".4", ".16", ".2"]),
        "no_repetition": len(set(text.split())) > 3,
        "reasonable_length": 5 < len(text) < 150,
    }
    score = sum(checks.values()) / len(checks)
    return score, checks


def evaluate_note_completion(generated_text):
    text = generated_text[:200]

    checks = {
        "has_parentheses": "(" in text and ")" in text,
        "has_fret_numbers": any(
            f"{s}.{f}" in text for s in range(1, 7) for f in range(1, 10)
        ),
        "valid_note_format": "." in text and ("(" in text or text.strip()[0].isdigit()),
        "has_rhythm_dot": any(x in text for x in [".8", ".4", ".16"]),
        "has_effects": any(x in text for x in ["{t}", "{dy", "{tu"]),
        "reasonable_content": len([c for c in text if c.isdigit() or c in "()."]) > 5,
    }
    score = sum(checks.values()) / len(checks)
    return score, checks


def generate_and_evaluate(
    engine, prompt, eval_type="structure", max_tokens=100, temperature=0.8
):
    try:
        tokens = engine.tokenizer(prompt, prepend="<|bos|>")
        result, _ = engine.generate_batch(
            tokens, num_samples=1, max_tokens=max_tokens, temperature=temperature
        )
        generated = engine.tokenizer.decode(result[0])

        if eval_type == "structure":
            score, checks = evaluate_structure(generated)
        elif eval_type == "tab":
            score, checks = evaluate_tab_completion(generated)
        elif eval_type == "note":
            score, checks = evaluate_note_completion(generated)
        else:
            score, checks = 0, {}

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
    print("增强版模型对比: 基础结构 + 音符补全能力")
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
            loss_str = f", loss {info['loss']}" if info["loss"] else ""
            print(
                f"  ✓ {name}: {params / 1e6:.1f}M参数 (step {info['step']}{loss_str})"
            )
        else:
            print(f"  ✗ {name}: 未找到checkpoint (将在评估时跳过)")

    if len(models) < 2:
        print("\n错误: 至少需要2个模型进行对比")
        return

    print(f"\n✓ 成功加载 {len(models)} 个模型")
    print()

    results = {name: {"structure": [], "tab": [], "note": []} for name in models.keys()}

    # 1. 基础结构评估
    print("=" * 80)
    print("【测试1】基础结构生成")
    print("=" * 80)

    for i, (prompt, desc) in enumerate(STRUCTURE_PROMPTS, 1):
        print(f"\n[{i}/{len(STRUCTURE_PROMPTS)}] {desc}")
        print(f'  Prompt: "{prompt[:50]}{"..." if len(prompt) > 50 else ""}"')

        for model_name in models.keys():
            result = generate_and_evaluate(
                models[model_name]["engine"], prompt, "structure"
            )
            results[model_name]["structure"].append(result)

            gen_text = result.get("generated", "")[:80]
            score = result.get("score", 0)
            print(
                f"  {model_name:15s}: {score:5.1%} | {gen_text}{'...' if len(gen_text) > 80 else ''}"
            )

    # 2. Tab补全评估
    print("\n" + "=" * 80)
    print("【测试2】Tab补全能力 (| 开头)")
    print("=" * 80)

    tab_prompts = get_tab_completion_prompts()
    for i, (prompt, desc, subtype) in enumerate(tab_prompts, 1):
        print(f"\n[{i}/{len(tab_prompts)}] {desc} ({subtype})")
        print(f'  Prompt: "{prompt}"')

        for model_name in models.keys():
            result = generate_and_evaluate(
                models[model_name]["engine"], prompt, "tab", max_tokens=60
            )
            results[model_name]["tab"].append(result)

            gen_text = result.get("generated", "")[:60]
            score = result.get("score", 0)
            print(
                f"  {model_name:15s}: {score:5.1%} | {gen_text}{'...' if len(gen_text) > 60 else ''}"
            )

    # 3. 复杂音符补全评估
    print("\n" + "=" * 80)
    print("【测试3】复杂音符补全 (括号语法)")
    print("=" * 80)

    note_prompts = get_note_completion_prompts()
    for i, (prompt, desc, subtype) in enumerate(note_prompts, 1):
        print(f"\n[{i}/{len(note_prompts)}] {desc} ({subtype})")
        print(f'  Prompt: "{prompt}"')

        for model_name in models.keys():
            result = generate_and_evaluate(
                models[model_name]["engine"], prompt, "note", max_tokens=60
            )
            results[model_name]["note"].append(result)

            gen_text = result.get("generated", "")[:60]
            score = result.get("score", 0)
            print(
                f"  {model_name:15s}: {score:5.1%} | {gen_text}{'...' if len(gen_text) > 60 else ''}"
            )

    # 汇总报告
    print("\n" + "=" * 80)
    print("评估汇总")
    print("=" * 80)
    print()

    summary = []
    for model_name in models.keys():
        struct_scores = [r.get("score", 0) for r in results[model_name]["structure"]]
        tab_scores = [r.get("score", 0) for r in results[model_name]["tab"]]
        note_scores = [r.get("score", 0) for r in results[model_name]["note"]]

        avg_struct = sum(struct_scores) / len(struct_scores) if struct_scores else 0
        avg_tab = sum(tab_scores) / len(tab_scores) if tab_scores else 0
        avg_note = sum(note_scores) / len(note_scores) if note_scores else 0
        overall = avg_struct * 0.4 + avg_tab * 0.3 + avg_note * 0.3

        model_info = models[model_name]["info"]

        summary.append(
            {
                "name": model_name,
                "desc": model_info["desc"],
                "params": models[model_name]["params"] / 1e6,
                "step": model_info["step"],
                "train_loss": model_info.get("loss"),
                "structure_score": avg_struct,
                "tab_score": avg_tab,
                "note_score": avg_note,
                "overall_score": overall,
            }
        )

        print(f"【{model_name}】")
        print(f"  {model_info['desc']}")
        print(
            f"  参数量: {models[model_name]['params'] / 1e6:.1f}M | Step: {model_info['step']}"
        )
        print(f"  结构生成: {avg_struct:.1%}")
        print(f"  Tab补全:  {avg_tab:.1%}")
        print(f"  音符补全: {avg_note:.1%}")
        print(f"  综合得分: {overall:.1%}")
        print()

    # 排名
    print("=" * 80)
    print("模型排名 (按综合得分)")
    print("=" * 80)
    print()

    sorted_summary = sorted(summary, key=lambda x: x["overall_score"], reverse=True)

    for i, s in enumerate(sorted_summary, 1):
        print(f"{i}. {s['name']}")
        print(
            f"   综合: {s['overall_score']:.1%} | 结构: {s['structure_score']:.1%} | Tab: {s['tab_score']:.1%} | 音符: {s['note_score']:.1%}"
        )
        print(f"   参数: {s['params']:.1f}M | Step: {s['step']}")
        print(f"   {s['desc']}")
        print()

    # 能力对比分析
    print("=" * 80)
    print("专项能力分析")
    print("=" * 80)
    print()

    # 各项最佳
    best_struct = max(summary, key=lambda x: x["structure_score"])
    best_tab = max(summary, key=lambda x: x["tab_score"])
    best_note = max(summary, key=lambda x: x["note_score"])

    print(f"结构生成最强: {best_struct['name']} ({best_struct['structure_score']:.1%})")
    print(f"Tab补全最强:  {best_tab['name']} ({best_tab['tab_score']:.1%})")
    print(f"音符补全最强: {best_note['name']} ({best_note['note_score']:.1%})")
    print()

    # 保存结果
    output_file = Path(__file__).parent / "enhanced_comparison_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "models": {
                    k: {
                        "desc": v["info"]["desc"],
                        "params": v["params"],
                        "step": v["info"]["step"],
                    }
                    for k, v in models.items()
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
    print("结论与建议")
    print("=" * 80)
    print()

    best_model = sorted_summary[0]
    print(f"最佳模型: {best_model['name']}")
    print(f"  - 综合得分: {best_model['overall_score']:.1%}")
    print(f"  - 架构: {best_model['desc']}")
    print()

    # 深度vs宽度vs训练步数
    d8_5k = next((s for s in summary if s["name"] == "d8_depth"), None)
    d8_20k = next((s for s in summary if s["name"] == "d8_depth_v2"), None)
    d6_5k = next((s for s in summary if s["name"] == "d6_wide"), None)

    if d8_5k and d8_20k:
        improvement = d8_20k["overall_score"] - d8_5k["overall_score"]
        print(f"训练步数效果 (d8_depth):")
        print(f"  - 5K步:  {d8_5k['overall_score']:.1%}")
        print(f"  - 20K步: {d8_20k['overall_score']:.1%}")
        print(f"  - 提升:  {improvement:+.1%}")
        print()

    if d8_5k and d6_5k:
        print(f"深度vs宽度 (同为5K步):")
        print(f"  - 深度优先(d8): {d8_5k['overall_score']:.1%}")
        print(f"  - 宽度优先(d6): {d6_5k['overall_score']:.1%}")
        diff = d8_5k["overall_score"] - d6_5k["overall_score"]
        winner = "深度" if diff > 0 else "宽度"
        print(f"  - 结论: {winner}优先更优 ({abs(diff):.1%})")
        print()

    print("=" * 80)


if __name__ == "__main__":
    main()
