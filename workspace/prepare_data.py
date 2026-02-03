#!/usr/bin/env python3
"""
nanochat 数据预处理 Pipeline
读取 .env 配置，将 .atex 文件转换为 nanochat 可用的 parquet 格式
"""

import os
import re
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from dotenv import load_dotenv

# ------------------------------------------------------------------------------
# 配置加载
# ------------------------------------------------------------------------------


def load_config():
    """从.env文件加载配置"""
    # 加载.env文件 (从workspace/../ 即项目根目录加载)
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✓ 已加载配置: {env_path}")
    else:
        print(f"✗ 未找到.env文件: {env_path}")
        print("  将使用默认配置")

    config = {
        "original_dataset_dir": os.getenv(
            "ORIGINAL_DATASET_DIR", r"F:\workspace\effect-alphaTex\data\test\p1"
        ),
        "nanochat_base_dir": os.getenv("NANOCHAT_BASE_DIR", "./data/t1"),
    }

    # 转换为绝对路径
    config["original_dataset_dir"] = Path(config["original_dataset_dir"]).resolve()
    config["nanochat_base_dir"] = Path(config["nanochat_base_dir"]).resolve()
    config["output_data_dir"] = config["nanochat_base_dir"] / "base_data"

    return config


# ------------------------------------------------------------------------------
# 文本处理函数
# ------------------------------------------------------------------------------


def clean_atex_content(content: str) -> str:
    """
    清理.atex文件内容
    根据实际需要可以在这里添加更多清理逻辑
    """
    # 移除多余的空白行
    content = re.sub(r"\n\s*\n\s*\n+", "\n\n", content)

    # 移除行首行尾空白
    content = content.strip()

    # 标准化换行符
    content = content.replace("\r\n", "\n").replace("\r", "\n")

    return content


def split_into_documents(content: str, min_length: int = 20) -> list:
    """
    将内容分割成文档

    策略:
    1. 按段落分割（双换行）
    2. 过滤掉太短的段落
    3. 如果段落太长，尝试按句子分割
    4. 如果整个内容太短，将整个内容作为一个文档
    """
    # 首先尝试按段落分割
    paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]

    documents = []
    for para in paragraphs:
        if len(para) >= min_length:
            documents.append(para)
        # 如果段落很长，按句子分割
        elif len(para) > 1000:
            sentences = re.split(r"(?<=[。！？.!?])\s+", para)
            current_doc = ""
            for sent in sentences:
                if len(current_doc) + len(sent) < 500:
                    current_doc += sent
                else:
                    if len(current_doc) >= min_length:
                        documents.append(current_doc)
                    current_doc = sent
            if len(current_doc) >= min_length:
                documents.append(current_doc)

    # 如果没有提取到文档但内容不为空，将整个内容作为一个文档
    if not documents and content.strip():
        documents.append(content.strip())

    return documents


def process_atex_file(filepath: Path) -> list:
    """
    处理单个.atex文件，返回文档列表
    """
    print(f"  处理: {filepath.name}")

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        # 尝试其他编码
        try:
            with open(filepath, "r", encoding="gbk") as f:
                content = f.read()
            print(f"    注意: 使用GBK编码读取")
        except Exception as e:
            print(f"    ✗ 读取失败: {e}")
            return []
    except Exception as e:
        print(f"    ✗ 读取失败: {e}")
        return []

    # 清理内容
    content = clean_atex_content(content)

    # 分割成文档
    documents = split_into_documents(content)

    print(f"    ✓ 提取 {len(documents)} 个文档")

    return documents


# ------------------------------------------------------------------------------
# Parquet 写入
# ------------------------------------------------------------------------------


def write_parquet_shard(
    documents: list, output_dir: Path, shard_idx: int, row_group_size: int = 1024
):
    """写入单个parquet shard"""
    shard_path = output_dir / f"shard_{shard_idx:05d}.parquet"

    table = pa.Table.from_pydict({"text": documents})
    pq.write_table(
        table,
        shard_path,
        row_group_size=row_group_size,
        use_dictionary=False,
        compression="zstd",
        compression_level=3,
        write_statistics=False,
    )

    total_chars = sum(len(d) for d in documents)
    print(f"    ✓ 写入 {shard_path.name}: {len(documents)} 文档, {total_chars:,} 字符")

    return shard_path


def create_validation_shard(train_shards: list, output_dir: Path):
    """从训练数据创建验证集（复制最后一个shard）"""
    if not train_shards:
        return None

    import shutil

    last_shard = train_shards[-1]
    val_shard = output_dir / f"shard_{len(train_shards):05d}.parquet"

    shutil.copy(last_shard, val_shard)
    print(f"  ✓ 创建验证集: {val_shard.name}")

    return val_shard


# ------------------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------------------


def process_pipeline(
    shard_size_chars: int = 250_000_000,  # 每个shard约2.5亿字符
    min_docs_per_shard: int = 1000,  # 每个shard最少文档数
):
    """
    主处理流程

    Args:
        shard_size_chars: 每个shard的目标字符数
        min_docs_per_shard: 每个shard的最小文档数
    """
    print("=" * 70)
    print("nanochat 数据预处理 Pipeline")
    print("=" * 70)

    # 1. 加载配置
    print("\n[1/5] 加载配置...")
    config = load_config()
    print(f"  原始数据目录: {config['original_dataset_dir']}")
    print(f"  输出目录: {config['output_data_dir']}")

    # 2. 检查输入目录
    print("\n[2/5] 检查输入数据...")
    if not config["original_dataset_dir"].exists():
        print(f"  ✗ 错误: 原始数据目录不存在: {config['original_dataset_dir']}")
        return False

    # 查找所有.atex文件
    atex_files = list(config["original_dataset_dir"].rglob("*.atex"))
    print(f"  ✓ 找到 {len(atex_files)} 个 .atex 文件")

    if len(atex_files) == 0:
        print("  ✗ 错误: 没有找到任何.atex文件")
        return False

    # 显示前5个文件
    for f in atex_files[:5]:
        print(f"    - {f.relative_to(config['original_dataset_dir'])}")
    if len(atex_files) > 5:
        print(f"    ... 还有 {len(atex_files) - 5} 个文件")

    # 3. 创建输出目录
    print("\n[3/5] 准备输出目录...")
    config["output_data_dir"].mkdir(parents=True, exist_ok=True)
    print(f"  ✓ 输出目录已就绪: {config['output_data_dir']}")

    # 4. 处理文件
    print("\n[4/5] 处理.atex文件...")
    all_documents = []
    total_chars = 0

    for filepath in atex_files:
        docs = process_atex_file(filepath)
        all_documents.extend(docs)
        total_chars += sum(len(d) for d in docs)

    print(f"\n  总计: {len(all_documents)} 个文档, {total_chars:,} 字符")

    if len(all_documents) == 0:
        print("  ✗ 错误: 没有提取到任何文档")
        return False

    # 5. 写入parquet文件
    print("\n[5/5] 写入parquet文件...")

    shard_idx = 0
    shard_docs = []
    shard_chars = 0
    train_shards = []

    for doc in all_documents:
        shard_docs.append(doc)
        shard_chars += len(doc)

        # 当达到目标大小时写入shard
        if shard_chars >= shard_size_chars or (
            shard_chars > 0 and len(shard_docs) >= min_docs_per_shard
        ):
            shard_path = write_parquet_shard(
                shard_docs, config["output_data_dir"], shard_idx
            )
            train_shards.append(shard_path)
            shard_idx += 1
            shard_docs = []
            shard_chars = 0

    # 写入最后一个shard
    if shard_docs:
        shard_path = write_parquet_shard(
            shard_docs, config["output_data_dir"], shard_idx
        )
        train_shards.append(shard_path)
        shard_idx += 1

    # 创建验证集
    print("\n  创建验证集...")
    val_shard = create_validation_shard(train_shards, config["output_data_dir"])

    # 6. 验证结果
    print("\n" + "=" * 70)
    print("处理完成!")
    print("=" * 70)
    print(f"训练集 shards: {len(train_shards)}")
    if val_shard:
        print(f"验证集 shards: 1")
    print(f"总 shards: {len(train_shards) + (1 if val_shard else 0)}")
    print(f"\n输出位置: {config['output_data_dir']}")
    print("\n现在可以运行:")
    print(f"  python -m scripts.tok_train")
    print(f"  python -m scripts.base_train --depth=4 --device-batch-size=4")

    return True


def verify_output():
    """验证生成的parquet文件"""
    print("\n" + "=" * 70)
    print("验证输出文件")
    print("=" * 70)

    config = load_config()
    data_dir = config["output_data_dir"]

    if not data_dir.exists():
        print(f"✗ 数据目录不存在: {data_dir}")
        return False

    parquet_files = sorted([f for f in data_dir.glob("*.parquet")])

    if not parquet_files:
        print("✗ 没有找到parquet文件")
        return False

    print(f"✓ 找到 {len(parquet_files)} 个parquet文件:\n")

    total_docs = 0
    total_chars = 0

    for pf_path in parquet_files:
        try:
            pf = pq.ParquetFile(pf_path)
            # 读取所有行
            table = pf.read()
            texts = table.column("text").to_pylist()

            docs = len(texts)
            chars = sum(len(t) for t in texts)
            total_docs += docs
            total_chars += chars

            print(f"  {pf_path.name}:")
            print(f"    - 文档数: {docs}")
            print(f"    - 字符数: {chars:,}")
            print(f"    - Row groups: {pf.num_row_groups}")
            if texts:
                print(f"    - 样例: {texts[0][:80]}...")
            print()
        except Exception as e:
            print(f"  {pf_path.name}: ✗ 验证失败 - {e}")

    print(f"总计: {total_docs} 文档, {total_chars:,} 字符")
    return True


# ------------------------------------------------------------------------------
# 命令行入口
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="将.atex文件转换为nanochat parquet格式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python prepare_data.py              # 运行完整流程
  python prepare_data.py --verify     # 只验证输出
  python prepare_data.py --shard-size 1000000  # 指定shard大小
        """,
    )
    parser.add_argument(
        "--verify", action="store_true", help="只验证输出文件，不重新处理"
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=250_000_000,
        help="每个shard的目标字符数 (默认: 250000000)",
    )
    parser.add_argument(
        "--min-docs", type=int, default=1000, help="每个shard的最小文档数 (默认: 1000)"
    )

    args = parser.parse_args()

    if args.verify:
        verify_output()
    else:
        success = process_pipeline(
            shard_size_chars=args.shard_size, min_docs_per_shard=args.min_docs
        )
        if success:
            print("\n" + "=" * 70)
            verify_output()
        else:
            print("\n✗ 处理失败!")
            exit(1)
