#!/usr/bin/env python3
"""
训练监控脚本 - 实时监控显存使用、训练进度和系统状态
"""

import os
import sys
import time
import subprocess
import json
from datetime import datetime
from pathlib import Path


# 颜色代码
class Colors:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"


def log(message, color=Colors.ENDC):
    """带时间戳的日志输出"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"{color}[{timestamp}] {message}{Colors.ENDC}")


def get_gpu_info():
    """获取GPU信息"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.used,memory.total,utilization.gpu,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            if len(parts) >= 5:
                return {
                    "name": parts[0],
                    "memory_used": float(parts[1]),  # MB
                    "memory_total": float(parts[2]),  # MB
                    "utilization": float(parts[3]),  # %
                    "temperature": float(parts[4]),  # °C
                }
    except Exception as e:
        pass
    return None


def get_training_progress(log_file=None):
    """从日志文件或检查点获取训练进度"""
    progress = {"step": 0, "loss": 0.0, "tokens_per_sec": 0, "status": "等待开始"}

    # 检查是否有wandb日志
    try:
        # 尝试从最新的日志文件中读取
        if log_file and Path(log_file).exists():
            with open(log_file, "r") as f:
                lines = f.readlines()
                # 查找最后几行包含训练信息的
                for line in reversed(lines[-20:]):
                    if "step" in line and "loss" in line:
                        # 简单解析
                        if "step" in line:
                            try:
                                step_str = line.split("step")[1].split()[0]
                                progress["step"] = int(step_str.split("/")[0])
                            except:
                                pass
                        break
    except:
        pass

    return progress


def display_dashboard(gpu_info, progress, phase, elapsed_time):
    """显示监控面板"""
    os.system("clear" if os.name != "nt" else "cls")

    print(f"{Colors.HEADER}{'=' * 70}{Colors.ENDC}")
    print(f"{Colors.HEADER}{Colors.BOLD}  nanochat 训练监控面板{' ' * 48}{Colors.ENDC}")
    print(f"{Colors.HEADER}{'=' * 70}{Colors.ENDC}")
    print()

    # 阶段信息
    phase_colors = {
        "Tokenizer训练": Colors.CYAN,
        "预训练": Colors.GREEN,
        "评估": Colors.YELLOW,
        "完成": Colors.BLUE,
    }
    phase_color = phase_colors.get(phase, Colors.ENDC)
    print(f"{Colors.BOLD}当前阶段:{Colors.ENDC} {phase_color}{phase}{Colors.ENDC}")
    print(
        f"{Colors.BOLD}运行时间:{Colors.ENDC} {elapsed_time // 60}分{elapsed_time % 60}秒"
    )
    print()

    # GPU信息
    if gpu_info:
        print(f"{Colors.BOLD}GPU状态:{Colors.ENDC}")
        print(f"  设备: {gpu_info['name']}")

        # 显存使用条
        mem_percent = (gpu_info["memory_used"] / gpu_info["memory_total"]) * 100
        bar_length = 30
        filled = int(bar_length * mem_percent / 100)
        bar = "█" * filled + "░" * (bar_length - filled)

        # 根据使用率选择颜色
        if mem_percent > 90:
            mem_color = Colors.RED
        elif mem_percent > 70:
            mem_color = Colors.YELLOW
        else:
            mem_color = Colors.GREEN

        print(f"  显存: {mem_color}[{bar}]{Colors.ENDC} {mem_percent:.1f}%")
        print(
            f"        {gpu_info['memory_used']:.0f} MB / {gpu_info['memory_total']:.0f} MB"
        )
        print(f"  利用率: {gpu_info['utilization']:.0f}%")
        print(f"  温度: {gpu_info['temperature']:.0f}°C")
    else:
        print(f"{Colors.YELLOW}无法获取GPU信息{Colors.ENDC}")
    print()

    # 训练进度
    print(f"{Colors.BOLD}训练进度:{Colors.ENDC}")
    print(f"  步骤: {progress['step']}")
    print(f"  损失: {progress['loss']:.4f}" if progress["loss"] > 0 else "  损失: --")
    print(
        f"  速度: {progress['tokens_per_sec']} tok/s"
        if progress["tokens_per_sec"] > 0
        else "  速度: --"
    )
    print()

    # 状态提示
    print(f"{Colors.HEADER}{'=' * 70}{Colors.ENDC}")
    if mem_percent > 90:
        print(f"{Colors.RED}⚠️  警告: 显存使用超过90%！{Colors.ENDC}")
    elif mem_percent > 70:
        print(f"{Colors.YELLOW}ℹ️  提示: 显存使用正常{Colors.ENDC}")
    else:
        print(f"{Colors.GREEN}✓ 显存使用良好{Colors.ENDC}")
    print(f"{Colors.HEADER}{'=' * 70}{Colors.ENDC}")
    print()
    print("按 Ctrl+C 停止监控")


def main():
    """主监控循环"""
    print("启动训练监控...")
    print("请确保训练脚本已在运行")
    print()

    start_time = time.time()
    phase = "等待开始"

    # 检测当前阶段
    base_dir = os.environ.get("NANOCHAT_BASE_DIR", "./data/t1")

    try:
        while True:
            elapsed = int(time.time() - start_time)

            # 检测当前阶段
            if Path(f"{base_dir}/tokenizer/tokenizer.pkl").exists():
                if Path(f"{base_dir}/base_checkpoints/d4_5gb").exists():
                    checkpoints = list(
                        Path(f"{base_dir}/base_checkpoints/d4_5gb").glob("*.pt")
                    )
                    if checkpoints:
                        phase = "预训练"
                    else:
                        phase = "评估"
                else:
                    phase = "Tokenizer训练完成"

            # 获取信息
            gpu_info = get_gpu_info()
            progress = get_training_progress()

            # 显示面板
            display_dashboard(gpu_info, progress, phase, elapsed)

            # 每秒更新
            time.sleep(1)

    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}监控已停止{Colors.ENDC}")
        sys.exit(0)


if __name__ == "__main__":
    main()
