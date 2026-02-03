# Debug Report: UV Cache Corruption

## 问题描述

在执行 `uv sync --extra gpu` 安装CUDA/GPU支持时遇到以下错误：

```
error: Failed to install: nvidia_cufft_cu12-11.3.3.83-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl (nvidia-cufft-cu12==11.3.3.83)
  Caused by: The wheel is invalid: Metadata field Name not found
```

## 根本原因

uv工具缓存中的wheel文件metadata已损坏或不完整。这可能是由于：
- 下载过程中断导致文件不完整
- 缓存文件系统损坏
- 网络问题导致的损坏下载

## 解决方案

清除uv缓存并重新同步：

```bash
rm -rf ~/.cache/uv && uv sync --extra gpu
```

## 验证

安装成功后，验证PyTorch和CUDA：

```bash
source .venv/bin/activate
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

预期输出：
```
PyTorch: 2.9.1+cu128
CUDA available: True
CUDA version: 12.8
```

## 推荐做法

### 1. 遇到UV安装错误时

| 错误类型 | 解决方案 |
|---------|---------|
| `Metadata field Name not found` | 清除缓存：`rm -rf ~/.cache/uv` |
| `Failed to hardlink files` | 设置环境变量：`export UV_LINK_MODE=copy` |
| 下载超时/中断 | 检查网络，重试或更换镜像源 |

### 2. 通用Debug流程

1. **查看详细错误信息**：添加 `-v` 或 `--verbose` 标志
2. **清除缓存**：uv缓存问题很常见，优先尝试清除
3. **检查网络**：确保能访问PyTorch官方源
4. **检查磁盘空间**：大文件下载需要足够空间

### 3. 预防措施

- 定期清理uv缓存：`rm -rf ~/.cache/uv`
- 使用稳定的网络连接
- 对于WSL环境，注意Windows和Linux文件系统之间的hardlink限制

## 相关配置

项目使用CUDA 12.8（cu128）：

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true
```

安装时会下载以下NVIDIA包：
- nvidia-cuda-runtime-cu12
- nvidia-cufft-cu12
- nvidia-curand-cu12
- nvidia-cusolver-cu12
- nvidia-nvjitlink-cu12
- 等

## 参考

- UV文档: https://docs.astral.sh/uv/
- PyTorch CUDA安装: https://pytorch.org/get-started/locally/

---
记录时间: 2026-02-03
