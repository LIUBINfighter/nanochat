#!/usr/bin/env python3
"""
快速测试脚本 - 验证数据处理pipeline
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_env_loading():
    """测试.env文件加载"""
    print("=" * 70)
    print("测试1: 加载.env配置")
    print("=" * 70)

    try:
        from dotenv import load_dotenv
    except ImportError:
        print("✗ 错误: 需要安装 python-dotenv")
        print("  运行: pip install python-dotenv")
        return False

    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        print(f"✗ 错误: 未找到.env文件: {env_path}")
        return False

    load_dotenv(env_path)

    original_dir = os.getenv("ORIGINAL_DATASET_DIR")
    nanochat_dir = os.getenv("NANOCHAT_BASE_DIR")

    print(f"✓ .env文件加载成功")
    print(f"  ORIGINAL_DATASET_DIR = {original_dir}")
    print(f"  NANOCHAT_BASE_DIR = {nanochat_dir}")

    return True


def test_dependencies():
    """测试必需的依赖"""
    print("\n" + "=" * 70)
    print("测试2: 检查依赖")
    print("=" * 70)

    missing = []

    try:
        import pyarrow

        print("✓ pyarrow 已安装")
    except ImportError:
        print("✗ pyarrow 未安装")
        missing.append("pyarrow")

    try:
        import pyarrow.parquet

        print("✓ pyarrow.parquet 可用")
    except ImportError:
        print("✗ pyarrow.parquet 不可用")
        missing.append("pyarrow")

    try:
        from dotenv import load_dotenv

        print("✓ python-dotenv 已安装")
    except ImportError:
        print("✗ python-dotenv 未安装")
        missing.append("python-dotenv")

    if missing:
        print(f"\n缺少依赖，请运行:")
        print(f"  pip install {' '.join(set(missing))}")
        return False

    return True


def test_data_format():
    """测试生成的parquet文件格式"""
    print("\n" + "=" * 70)
    print("测试3: 验证输出数据格式")
    print("=" * 70)

    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("✗ 需要安装 pyarrow")
        return False

    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)

    nanochat_base = Path(os.getenv("NANOCHAT_BASE_DIR", "./data/t1")).resolve()
    data_dir = nanochat_base / "base_data"

    if not data_dir.exists():
        print(f"✗ 数据目录不存在: {data_dir}")
        print("  请先运行: python prepare_data.py")
        return False

    parquet_files = sorted([f for f in data_dir.glob("*.parquet")])

    if not parquet_files:
        print(f"✗ 没有找到parquet文件")
        return False

    print(f"✓ 找到 {len(parquet_files)} 个parquet文件")

    all_valid = True
    for pf_path in parquet_files:
        try:
            pf = pq.ParquetFile(pf_path)
            schema = pf.schema

            # 检查必需的'text'列
            if "text" not in schema.names:
                print(f"✗ {pf_path.name}: 缺少'text'列")
                all_valid = False
                continue

            # 读取第一行验证
            table = pf.read_row_group(0)
            texts = table.column("text").to_pylist()

            print(f"✓ {pf_path.name}:")
            print(f"    - Row groups: {pf.num_row_groups}")
            print(f"    - 文档数: {len(texts)}")
            if texts:
                sample = texts[0][:100] if len(texts[0]) > 100 else texts[0]
                print(f"    - 样例: {sample}...")

        except Exception as e:
            print(f"✗ {pf_path.name}: 验证失败 - {e}")
            all_valid = False

    return all_valid


def test_file_discovery():
    """测试是否能找到.atex文件"""
    print("\n" + "=" * 70)
    print("测试4: 检查原始.atex文件")
    print("=" * 70)

    try:
        from dotenv import load_dotenv
    except ImportError:
        print("✗ 需要 python-dotenv")
        return False

    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)

    original_dir = Path(
        os.getenv("ORIGINAL_DATASET_DIR", r"F:\workspace\effect-alphaTex\data\test\p1")
    )

    print(f"搜索目录: {original_dir}")

    if not original_dir.exists():
        print(f"✗ 目录不存在!")
        print(f"  注意: 当前路径是Windows格式，如果在Linux上运行将无法访问")
        print(f"  请修改.env文件中的路径为当前系统的有效路径")
        return False

    atex_files = list(original_dir.rglob("*.atex"))
    print(f"✓ 找到 {len(atex_files)} 个 .atex 文件")

    for f in atex_files[:5]:
        print(f"  - {f.name}")
    if len(atex_files) > 5:
        print(f"  ... 还有 {len(atex_files) - 5} 个")

    return len(atex_files) > 0


def main():
    """运行所有测试"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 20 + "nanochat 数据准备测试" + " " * 25 + "║")
    print("╚" + "=" * 68 + "╝")
    print()

    tests = [
        ("环境配置", test_env_loading),
        ("依赖检查", test_dependencies),
        ("文件发现", test_file_discovery),
        ("数据格式", test_data_format),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ 测试 '{name}' 出错: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, False))

    # 打印总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)

    for name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {status}: {name}")

    passed = sum(1 for _, r in results if r)
    total = len(results)

    print(f"\n总计: {passed}/{total} 测试通过")

    if passed == total:
        print("\n🎉 所有测试通过！数据准备就绪，可以开始训练了。")
        print("\n下一步:")
        print("  1. 训练tokenizer: python -m scripts.tok_train")
        print(
            "  2. 预训练模型: python -m scripts.base_train --depth=4 --device-batch-size=4"
        )
        return 0
    else:
        print("\n⚠️  部分测试失败，请检查上面的错误信息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
