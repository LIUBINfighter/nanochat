#!/usr/bin/env python3
"""
d8_depth 全步数评估: 找出最佳checkpoint (500-20000步)
评测所有中间模型，找到质量vs步数的甜点
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


def get_all_checkpoints():
    """获取d8_depth所有checkpoint"""
    base_dir = Path(get_base_dir()) / "base_checkpoints" / "d8_depth"
    checkpoints = []

    # 扫描所有model_*.pt文件
    for f in sorted(base_dir.glob("model_*.pt")):
        step = int(f.stem.split("_")[1])
        checkpoints.append((step, str(f)))

    return sorted(checkpoints)


PROMPTS = [
    ('\\title "', "标题定义"),
    ('\\track ("Guitar" "Standard Tuning")', "音轨定义"),
    ('\\artist "', "艺术家"),
    ("Song Title: ", "歌曲标题"),
    ("\\tempo 120", "速度"),
    ('\\title "My Song"\\n\\track ("Guitar") {', "完整开头"),
    ("| 3.2", "Tab补全-单音"),
    ("| 3.2 2.3", "Tab补全-双音"),
    ("| (3.2 2.3).8", "Tab补全-和弦"),
    ("(6.2 1.5).", "音符-和弦基础"),
    ("(4.2 1.5{t}).", "音符-装饰音"),
    ("24.1.", "音符-单弦"),
]


def load_model(checkpoint_path, device="cuda"):
    config = GPTConfig(
        vocab_size=8192,
        n_layer=8,
        n_embd=256,
        n_head=4,
        n_kv_head=4,
        sequence_len=512,
        window_pattern="L",
    )

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model = GPT(config)

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


def evaluate_generation(generated_text, prompt_type):
    text = generated_text[:300]

    if "Tab" in prompt_type or "音符" in prompt_type:
        # Tab/音符补全评估
        checks = {
            "has_fret": any(c.isdigit() for c in text),
            "has_dot": "." in text,
            "valid": "|" in text
            or any(f"{a}.{b}" in text for a in range(1, 7) for b in range(1, 7)),
            "no_repeat": len(set(text.split())) > 3,
            "reasonable": 5 < len(text) < 200,
        }
    else:
        # 结构生成评估
        checks = {
            "has_title": "\\title" in text,
            "has_track": "\\track" in text,
            "has_braces": "{" in text and "}" in text,
            "has_quotes": '"' in text,
            "valid_struct": text.count("{") >= text.count("}"),
            "no_gibberish": sum(1 for c in text if c.isalpha()) > 20,
            "has_instr": "instrument" in text,
            "has_vol": "volume" in text,
        }

    score = sum(checks.values()) / len(checks)
    return score


def generate_and_evaluate(engine, prompt, prompt_type):
    try:
        tokens = engine.tokenizer(prompt, prepend="<|bos|>")
        result, _ = engine.generate_batch(
            tokens, num_samples=1, max_tokens=80, temperature=0.8
        )
        generated = engine.tokenizer.decode(result[0])
        score = evaluate_generation(generated, prompt_type)
        return score
    except Exception as e:
        return 0


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print("d8_depth 全步数评估: 寻找最佳checkpoint")
    print("=" * 80)
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"设备: {device}")
    print()

    os.environ["NANOCHAT_BASE_DIR"] = "./data/t1"
    tokenizer = get_tokenizer()

    # 获取所有checkpoint
    checkpoints = get_all_checkpoints()
    print(f"发现 {len(checkpoints)} 个checkpoint:")
    print(f"  范围: {checkpoints[0][0]}步 ~ {checkpoints[-1][0]}步")
    print()

    # 评估每个checkpoint
    results = []

    for i, (step, ckpt_path) in enumerate(checkpoints):
        print(f"\n[{i + 1}/{len(checkpoints)}] Step {step}")
        print("-" * 80)

        model, params = load_model(ckpt_path, device)
        if not model:
            print(f"  ✗ 加载失败，跳过")
            continue

        engine = Engine(model, tokenizer)

        # 测试所有prompts
        scores = []
        for prompt, prompt_type in PROMPTS:
            score = generate_and_evaluate(engine, prompt, prompt_type)
            scores.append(score)

        avg_score = sum(scores) / len(scores)

        # 分类得分
        struct_scores = scores[:6]  # 前6个是结构
        tab_scores = scores[6:9]  # 接下来3个是Tab
        note_scores = scores[9:]  # 最后3个是音符

        struct_avg = sum(struct_scores) / len(struct_scores)
        tab_avg = sum(tab_scores) / len(tab_scores)
        note_avg = sum(note_scores) / len(note_scores)

        results.append(
            {
                "step": step,
                "avg_score": avg_score,
                "struct_score": struct_avg,
                "tab_score": tab_avg,
                "note_score": note_avg,
                "all_scores": scores,
            }
        )

        print(
            f"  综合: {avg_score:.1%} | 结构: {struct_avg:.1%} | Tab: {tab_avg:.1%} | 音符: {note_avg:.1%}"
        )

        # 释放显存
        del model, engine
        torch.cuda.empty_cache()

    # 汇总
    print("\n" + "=" * 80)
    print("评估结果汇总")
    print("=" * 80)
    print()

    # 按综合得分排序
    sorted_results = sorted(results, key=lambda x: x["avg_score"], reverse=True)

    print("Top 10 最佳模型:")
    print()
    for i, r in enumerate(sorted_results[:10], 1):
        print(
            f"{i:2d}. Step {r['step']:5d} | 综合: {r['avg_score']:.1%} | 结构: {r['struct_score']:.1%} | Tab: {r['tab_score']:.1%} | 音符: {r['note_score']:.1%}"
        )

    print()
    print("=" * 80)
    print("关键里程碑")
    print("=" * 80)
    print()

    milestones = [500, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000, 15000, 20000]
    print("步数    | 综合   | 结构   | Tab    | 音符")
    print("-" * 60)
    for step in milestones:
        r = next((x for x in results if x["step"] == step), None)
        if r:
            print(
                f"{step:6d}  | {r['avg_score']:5.1%} | {r['struct_score']:5.1%} | {r['tab_score']:5.1%} | {r['note_score']:5.1%}"
            )
        else:
            print(f"{step:6d}  | -      | -      | -      | -")

    # 趋势分析
    print()
    print("=" * 80)
    print("趋势分析")
    print("=" * 80)
    print()

    best = sorted_results[0]
    print(f"✓ 最佳checkpoint: Step {best['step']} (综合 {best['avg_score']:.1%})")
    print()

    # 找结构最佳的
    best_struct = max(results, key=lambda x: x["struct_score"])
    print(
        f"✓ 结构生成最佳: Step {best_struct['step']} ({best_struct['struct_score']:.1%})"
    )

    # 找Tab最佳的
    best_tab = max(results, key=lambda x: x["tab_score"])
    print(f"✓ Tab补全最佳: Step {best_tab['step']} ({best_tab['tab_score']:.1%})")

    # 找音符最佳的
    best_note = max(results, key=lambda x: x["note_score"])
    print(f"✓ 音符补全最佳: Step {best_note['step']} ({best_note['note_score']:.1%})")

    # 找最早的90%+模型
    first_90 = next(
        (r for r in sorted(results, key=lambda x: x["step"]) if r["avg_score"] >= 0.9),
        None,
    )
    if first_90:
        print(f"✓ 最早达到90%: Step {first_90['step']}")

    # 找过拟合点
    max_score = max(r["avg_score"] for r in results)
    overfit_threshold = max_score * 0.95
    overfit_point = None
    for r in sorted(results, key=lambda x: x["step"]):
        if r["avg_score"] < overfit_threshold:
            overfit_point = r
            break
    if overfit_point and overfit_point["step"] > 5000:
        print(f"⚠ 过拟合可能发生在: Step {overfit_point['step']} 之后")

    # 保存结果
    output_file = Path(__file__).parent / "d8_all_steps_evaluation.json"
    with open(output_file, "w") as f:
        json.dump(
            {
                "timestamp": datetime.now().isoformat(),
                "model": "d8_depth",
                "results": results,
                "best_overall": best["step"],
                "best_struct": best_struct["step"],
                "best_tab": best_tab["step"],
                "best_note": best_note["step"],
            },
            f,
            indent=2,
        )

    print()
    print(f"✓ 详细结果已保存: {output_file}")
    print()


if __name__ == "__main__":
    main()
