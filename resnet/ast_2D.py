"""
Audio Spectrogram Transformer (AST-lite)
========================================
Inspired by: Gong et al., "AST: Audio Spectrogram Transformer", Interspeech 2021
             https://arxiv.org/abs/2104.01778

Architecture
------------
Input  : mel spectrogram  [B, C, 128, 126]  (CLAR 3-channel format)
         Channels = Magnitude + Phase + MelSpec

1. PatchEmbed  — 16×16 conv projection → [B, N_patches, embed_dim]
                 N_patches = (128//16) × (126//16) = 8 × 7 = 56
2. Prepend [CLS] token, add learnable positional embedding
3. TransformerEncoder  — depth=4, heads=4, FFN dim = 4×embed_dim
4. Take CLS output, project to 512-d to match ResNet2D output shape
Output : [B, 512, 1, 1]  — drop-in replacement for ResNet2D in net.py
"""

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Split spectrogram into non-overlapping patches and linearly project each."""

    def __init__(self, img_size=(128, 126), patch_size=(16, 16),
                 in_channels=3, embed_dim=192):
        super().__init__()
        self.patch_size = patch_size
        # Conv2d with stride==kernel gives non-overlapping patches
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        self.n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])

    def forward(self, x):
        x = self.proj(x)                        # [B, embed_dim, H', W']
        x = x.flatten(2).transpose(1, 2)        # [B, n_patches, embed_dim]
        return x


class AudioSpecTransformer(nn.Module):
    """
    Lightweight Audio Spectrogram Transformer.

    Args:
        img_size    : (freq_bins, time_frames) of input spectrogram — default (128, 126)
        patch_size  : patch height and width — default (16, 16)
        in_channels : number of input channels — default 3 (Mag + Phase + Mel)
        embed_dim   : transformer embedding dimension — default 192
        depth       : number of transformer encoder layers — default 4
        num_heads   : number of attention heads — default 4
        mlp_ratio   : FFN hidden dim = embed_dim × mlp_ratio — default 4.0
        drop        : dropout probability — default 0.1
    """

    def __init__(self, img_size=(128, 126), patch_size=(16, 16),
                 in_channels=3, embed_dim=192,
                 depth=4, num_heads=4, mlp_ratio=4.0, drop=0.1):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size, patch_size, in_channels, embed_dim)
        n_patches = self.patch_embed.n_patches  # 56

        # Learnable [CLS] token and positional embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, embed_dim))
        self.pos_drop  = nn.Dropout(p=drop)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=drop,
            batch_first=True,
            norm_first=True,   # Pre-LN (more stable training)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)

        # Project CLS output to 512 to match ResNet output dimension
        self.proj_out = nn.Linear(embed_dim, 512)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.proj_out.weight, std=0.02)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, x):
        B = x.shape[0]

        x   = self.patch_embed(x)                       # [B, N, D]
        cls = self.cls_token.expand(B, -1, -1)          # [B, 1, D]
        x   = torch.cat([cls, x], dim=1)                # [B, N+1, D]
        x   = self.pos_drop(x + self.pos_embed)

        x   = self.transformer(x)                       # [B, N+1, D]
        x   = self.norm(x[:, 0])                        # CLS token → [B, D]
        x   = self.proj_out(x)                          # [B, 512]

        return x.unsqueeze(-1).unsqueeze(-1)            # [B, 512, 1, 1]


def CreateAST2D(img_channels=3, num_classes=10):
    """Factory function matching the CreateResNet2D signature."""
    return AudioSpecTransformer(in_channels=img_channels)
