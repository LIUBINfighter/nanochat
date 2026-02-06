"""
融合层模块 - 适配nanochat架构的简化版本
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class SimpleFusionBlock(nn.Module):
    """简化的融合Transformer块"""
    
    def __init__(self, embed_dim=256, num_heads=4, mlp_ratio=4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, embed_dim)
        )
        
    def forward(self, x):
        x_norm = self.ln1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.ln2(x))
        return x


class MultimodalModel(nn.Module):
    """多模态模型 - 结合视觉编码器和预训练文本模型"""
    
    def __init__(self, vision_encoder, text_model, num_fusion_layers=3):
        super().__init__()
        
        self.vision_encoder = vision_encoder
        self.text_model = text_model
        
        # 视觉投影
        vision_dim = vision_encoder.embed_dim
        text_dim = 256
        self.vision_proj = nn.Linear(vision_dim, text_dim)
        
        # 融合层
        self.fusion_layers = nn.ModuleList([
            SimpleFusionBlock(text_dim, num_heads=4)
            for _ in range(num_fusion_layers)
        ])
        
        # 冻结文本模型
        for param in self.text_model.parameters():
            param.requires_grad = False
        self.text_model.eval()
        
    def forward(self, images=None, text_tokens=None, return_vision_tokens=False):
        """前向传播"""
        # 编码视觉
        vision_tokens = None
        if images is not None:
            vision_tokens = self.vision_encoder(images)
            vision_tokens = self.vision_proj(vision_tokens)
        
        # 如果没有文本输入，只返回视觉token
        if text_tokens is None:
            return vision_tokens
        
        # 如果没有视觉输入，只用文本模型
        if vision_tokens is None:
            with torch.no_grad():
                return self.text_model(text_tokens)
        
        # 处理文本token
        if text_tokens.dim() == 2:  # (B, T) - token IDs
            with torch.no_grad():
                text_embeds = self.text_model.transformer.wte(text_tokens)
        else:  # (B, T, D) - 已经嵌入
            text_embeds = text_tokens
        
        # 拼接视觉和文本
        combined = torch.cat([vision_tokens, text_embeds], dim=1)
        
        # 融合层处理
        for layer in self.fusion_layers:
            combined = layer(combined)
        
        # 分离并生成
        vision_len = vision_tokens.shape[1]
        text_out = combined[:, vision_len:, :]
        
        with torch.no_grad():
            output = self.text_model.lm_head(text_out)
        
        if return_vision_tokens:
            return output, vision_tokens
        return output
    
    def get_trainable_params(self):
        """获取可训练参数"""
        return (
            list(self.vision_encoder.parameters()) +
            list(self.vision_proj.parameters()) +
            list(self.fusion_layers.parameters())
        )
    
    def count_parameters(self):
        """统计参数量"""
        trainable = sum(p.numel() for p in self.get_trainable_params())
        frozen = sum(p.numel() for p in self.text_model.parameters())
        total = trainable + frozen
        
        return {
            'trainable': trainable,
            'frozen': frozen,
            'total': total,
            'trainable_M': trainable / 1e6,
            'frozen_M': frozen / 1e6,
            'total_M': total / 1e6
        }


if __name__ == "__main__":
    print("Testing MultimodalModel...")
    
    from vision_encoder import VisionEncoder
    from nanochat.gpt import GPT, GPTConfig
    
    vision_encoder = VisionEncoder(img_size=224, patch_size=16, embed_dim=256, num_layers=3)
    
    text_config = GPTConfig(
        vocab_size=8192, n_layer=8, n_embd=256,
        n_head=4, n_kv_head=4, sequence_len=512, window_pattern="L"
    )
    text_model = GPT(text_config)
    
    model = MultimodalModel(vision_encoder, text_model, num_fusion_layers=3)
    
    batch_size = 2
    test_images = torch.randn(batch_size, 3, 224, 224)
    test_text_ids = torch.randint(0, 8192, (batch_size, 50))
    
    print("\nTesting forward pass...")
    with torch.no_grad():
        output = model(test_images, test_text_ids)
    
    print(f"Images: {test_images.shape}")
    print(f"Text: {test_text_ids.shape}")
    print(f"Output: {output.shape}")
    
    stats = model.count_parameters()
    print(f"\nParameters: {stats['trainable_M']:.2f}M trainable, {stats['frozen_M']:.2f}M frozen")
    
    print("\n✓ MultimodalModel test passed!")
