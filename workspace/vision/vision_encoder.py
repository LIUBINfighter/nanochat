"""
视觉编码器模块
基于ViT架构，用于将图像编码为视觉token
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class PatchEmbedding(nn.Module):
    """图像patch嵌入层"""

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 256,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # 使用卷积实现patch embedding
        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, embed_dim, H//patch_size, W//patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)
        return x


class VisionTransformerBlock(nn.Module):
    """视觉Transformer块"""

    def __init__(
        self, embed_dim: int = 256, num_heads: int = 4, mlp_ratio: float = 4.0
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 自注意力
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out

        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class VisionEncoder(nn.Module):
    """
    视觉编码器
    将图像编码为视觉token序列

    配置:
        img_size: 输入图像尺寸 (默认224)
        patch_size: patch大小 (默认16)
        embed_dim: 嵌入维度 (与文本模型一致，256)
        num_layers: Transformer层数 (默认3)
        num_heads: 注意力头数 (默认4)
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        embed_dim: int = 256,
        num_layers: int = 3,
        num_heads: int = 4,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim

        # Patch嵌入
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)

        # 可学习的位置编码
        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))

        # Transformer块
        self.blocks = nn.ModuleList(
            [VisionTransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(embed_dim)

        # 初始化
        nn.init.normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: 输入图像 (B, C, H, W)
        Returns:
            视觉token (B, num_patches, embed_dim)
        """
        B = x.shape[0]

        # Patch嵌入
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # 加位置编码
        x = x + self.pos_embed

        # Transformer处理
        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x

    def get_num_patches(self) -> int:
        """获取patch数量"""
        return self.patch_embed.num_patches


class VisionProjection(nn.Module):
    """视觉-文本投影层"""

    def __init__(self, vision_dim: int = 256, text_dim: int = 256):
        super().__init__()
        self.proj = nn.Linear(vision_dim, text_dim)

    def forward(self, vision_tokens: torch.Tensor) -> torch.Tensor:
        return self.proj(vision_tokens)


if __name__ == "__main__":
    # 测试VisionEncoder
    print("Testing VisionEncoder...")

    # 创建模型
    encoder = VisionEncoder(
        img_size=224, patch_size=16, embed_dim=256, num_layers=3, num_heads=4
    )

    # 测试输入
    batch_size = 2
    test_images = torch.randn(batch_size, 3, 224, 224)

    # 前向传播
    with torch.no_grad():
        output = encoder(test_images)

    print(f"Input shape: {test_images.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected: ({batch_size}, 196, 256)")
    print(f"Num patches: {encoder.get_num_patches()}")

    # 测试投影层
    proj = VisionProjection(256, 256)
    projected = proj(output)
    print(f"Projected shape: {projected.shape}")

    # 计算参数量
    total_params = sum(p.numel() for p in encoder.parameters())
    print(f"VisionEncoder parameters: {total_params / 1e6:.2f}M")

    print("\n✓ VisionEncoder test passed!")
