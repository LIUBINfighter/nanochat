# nanochat 自定义数据预处理 Pipeline

## 📁 文件说明

| 文件 | 用途 |
|------|------|
| `prepare_data.py` | 主数据处理脚本，将.atex文件转换为parquet格式 |
| `test_data_pipeline.py` | 测试脚本，验证数据格式和环境配置 |
| `.env` | 配置文件，定义输入/输出路径 |

## ⚙️ 配置说明 (.env)

```bash
ORIGINAL_DATASET_DIR = "F:\\workspace\\effect-alphaTex\\data\\test\\p1"  # 原始.atex文件目录
NANOCHAT_BASE_DIR = "./data/t1"                                       # nanochat数据输出目录
```

### 路径说明
- **ORIGINAL_DATASET_DIR**: 包含你的.atex文件的目录（支持子目录递归搜索）
- **NANOCHAT_BASE_DIR**: 输出的nanochat格式数据存放位置，实际数据会存放在 `{NANOCHAT_BASE_DIR}/base_data/` 下

## 🚀 使用方法

### 1. 配置环境

确保已安装依赖：

```bash
pip install pyarrow python-dotenv
```

### 2. 编辑 .env 文件

根据你的实际路径修改 `.env` 文件：

```bash
# Windows 路径示例
ORIGINAL_DATASET_DIR = "F:\\workspace\\effect-alphaTex\\data\\test\\p1"
NANOCHAT_BASE_DIR = "./data/t1"

# Linux/Mac 路径示例
ORIGINAL_DATASET_DIR = "/home/user/my_atex_data"
NANOCHAT_BASE_DIR = "./nanochat_data"
```

### 3. 运行数据处理

```bash
python prepare_data.py
```

处理完成后，数据会存放在 `./data/t1/base_data/` 目录下，包含：
- `shard_00000.parquet` - 训练数据
- `shard_00001.parquet` - 验证数据（自动从训练数据复制）

### 4. 验证数据

```bash
python test_data_pipeline.py
```

或只验证输出：

```bash
python prepare_data.py --verify
```

### 5. 开始训练

数据准备就绪后，可以开始训练：

```bash
# 训练tokenizer
python -m scripts.tok_train --max-chars=1000000

# 预训练模型（小模型配置，适合测试）
python -m scripts.base_train \
    --depth=4 \
    --device-batch-size=4 \
    --max-seq-len=512 \
    --num-iterations=1000
```

## 🔧 高级配置

### 调整shard大小

```bash
# 每个shard包含约100万字符
python prepare_data.py --shard-size 1000000

# 每个shard最少500个文档
python prepare_data.py --min-docs 500
```

### 自定义文本处理

编辑 `prepare_data.py` 中的以下函数来自定义处理逻辑：

- `clean_atex_content()`: 清理原始文本内容
- `split_into_documents()`: 将内容分割成文档

## 📋 数据格式说明

### 输入 (.atex文件)

.atex文件是纯文本文件，内容示例：

```
这是第一个段落。

这是第二个段落，会被当作独立文档。

多段落内容支持。
```

### 输出 (parquet文件)

- **格式**: Parquet
- **必需列**: `text` (字符串类型)
- **压缩**: zstd
- **row_group_size**: 1024

parquet文件结构：
```
shard_XXXXX.parquet
├── text: string  # 文档文本内容
```

## ✅ 测试状态

当前测试数据已准备就绪：

```
✓ 环境配置 - 通过
✓ 依赖检查 - 通过  
✓ 文件发现 - 通过
✓ 数据格式 - 通过

输出位置: ./data/t1/base_data/
├── shard_00000.parquet (训练集: 3个文档)
└── shard_00001.parquet (验证集: 3个文档)
```

## 🐛 故障排除

### 问题: "没有找到.atex文件"

**解决**: 检查 `.env` 中的 `ORIGINAL_DATASET_DIR` 路径是否正确

### 问题: "缺少pyarrow"

**解决**: 
```bash
pip install pyarrow
```

### 问题: "目录不存在"

**解决**: 
- Windows路径在Linux上无法访问，请修改 `.env` 使用当前系统的有效路径
- 确保路径存在且有读取权限

### 问题: "提取到0个文档"

**解决**: 检查.atex文件内容，确保：
- 文件不是空的
- 内容包含可读的文本
- 文件编码为UTF-8或GBK

## 📝 自定义开发

如需修改数据处理逻辑，编辑 `prepare_data.py`：

```python
def clean_atex_content(content: str) -> str:
    """在这里添加自定义清理逻辑"""
    # 你的处理代码
    return content

def split_into_documents(content: str, min_length: int = 20) -> list:
    """在这里自定义文档分割策略"""
    # 你的分割逻辑
    return documents
```

## 🎯 下一步

数据准备完成后：

1. **训练Tokenizer**: 学习你的领域词汇
2. **预训练模型**: 在你的数据上训练语言模型  
3. **评估模型**: 使用base_eval评估模型效果
4. **SFT微调**: 使用chat_sft进行对话微调

祝训练顺利！🚀
