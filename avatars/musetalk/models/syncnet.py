"""
This file is modified from LatentSync (https://github.com/bytedance/LatentSync/blob/main/latentsync/models/stable_syncnet.py).
"""

import torch
from torch import nn
from einops import rearrange
from torch.nn import functional as F

import torch.nn as nn
import torch.nn.functional as F

from diffusers.models.attention import Attention as CrossAttention, FeedForward
from diffusers.utils.import_utils import is_xformers_available
from einops import rearrange


class SyncNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.audio_encoder = DownEncoder2D(
            in_channels=config["audio_encoder"]["in_channels"],
            block_out_channels=config["audio_encoder"]["block_out_channels"],
            downsample_factors=config["audio_encoder"]["downsample_factors"],
            dropout=config["audio_encoder"]["dropout"],
            attn_blocks=config["audio_encoder"]["attn_blocks"],
        )

        self.visual_encoder = DownEncoder2D(
            in_channels=config["visual_encoder"]["in_channels"],
            block_out_channels=config["visual_encoder"]["block_out_channels"],
            downsample_factors=config["visual_encoder"]["downsample_factors"],
            dropout=config["visual_encoder"]["dropout"],
            attn_blocks=config["visual_encoder"]["attn_blocks"],
        )

        self.eval()

    def forward(self, image_sequences, audio_sequences):
        vision_embeds = self.visual_encoder(image_sequences)  # (b, c, 1, 1)
        audio_embeds = self.audio_encoder(audio_sequences)  # (b, c, 1, 1)

        vision_embeds = vision_embeds.reshape(vision_embeds.shape[0], -1)  # (b, c)
        audio_embeds = audio_embeds.reshape(audio_embeds.shape[0], -1)  # (b, c)

        # Make them unit vectors
        vision_embeds = F.normalize(vision_embeds, p=2, dim=1)
        audio_embeds = F.normalize(audio_embeds, p=2, dim=1)

        return vision_embeds, audio_embeds
    
    def get_image_embed(self, image_sequences):
        vision_embeds = self.visual_encoder(image_sequences)  # (b, c, 1, 1)

        vision_embeds = vision_embeds.reshape(vision_embeds.shape[0], -1)  # (b, c)

        # Make them unit vectors
        vision_embeds = F.normalize(vision_embeds, p=2, dim=1)

        return vision_embeds

    def get_audio_embed(self, audio_sequences):
        audio_embeds = self.audio_encoder(audio_sequences)  # (b, c, 1, 1)

        audio_embeds = audio_embeds.reshape(audio_embeds.shape[0], -1)  # (b, c)
        
        audio_embeds = F.normalize(audio_embeds, p=2, dim=1)

        return audio_embeds

class ResnetBlock2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        eps: float = 1e-6,
        act_fn: str = "silu",
        downsample_factor=2,
    ):
        super().__init__()

        self.norm1 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=in_channels, eps=eps, affine=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=norm_num_groups, num_channels=out_channels, eps=eps, affine=True)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        if act_fn == "relu":
            self.act_fn = nn.ReLU()
        elif act_fn == "silu":
            self.act_fn = nn.SiLU()

        if in_channels != out_channels:
            self.conv_shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        else:
            self.conv_shortcut = None

        if isinstance(downsample_factor, list):
            downsample_factor = tuple(downsample_factor)

        if downsample_factor == 1:
            self.downsample_conv = None
        else:
            self.downsample_conv = nn.Conv2d(
                out_channels, out_channels, kernel_size=3, stride=downsample_factor, padding=0
            )
            self.pad = (0, 1, 0, 1)
            if isinstance(downsample_factor, tuple):
                if downsample_factor[0] == 1:
                    self.pad = (0, 1, 1, 1)  # The padding order is from back to front
                elif downsample_factor[1] == 1:
                    self.pad = (1, 1, 0, 1)

    def forward(self, input_tensor):
        hidden_states = input_tensor

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.act_fn(hidden_states)

        hidden_states = self.conv1(hidden_states)
        hidden_states = self.norm2(hidden_states)
        hidden_states = self.act_fn(hidden_states)

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.conv2(hidden_states)

        if self.conv_shortcut is not None:
            input_tensor = self.conv_shortcut(input_tensor)

        hidden_states += input_tensor

        if self.downsample_conv is not None:
            hidden_states = F.pad(hidden_states, self.pad, mode="constant", value=0)
            hidden_states = self.downsample_conv(hidden_states)

        return hidden_states


class AttentionBlock2D(nn.Module):
    def __init__(self, query_dim, norm_num_groups=32, dropout=0.0):
        super().__init__()
        if not is_xformers_available():
            raise ModuleNotFoundError(
                "You have to install xformers to enable memory efficient attetion", name="xformers"
            )
        # inner_dim = dim_head * heads
        self.norm1 = torch.nn.GroupNorm(num_groups=norm_num_groups, num_channels=query_dim, eps=1e-6, affine=True)
        self.norm2 = nn.LayerNorm(query_dim)
        self.norm3 = nn.LayerNorm(query_dim)

        self.ff = FeedForward(query_dim, dropout=dropout, activation_fn="geglu")

        self.conv_in = nn.Conv2d(query_dim, query_dim, kernel_size=1, stride=1, padding=0)
        self.conv_out = nn.Conv2d(query_dim, query_dim, kernel_size=1, stride=1, padding=0)

        self.attn = CrossAttention(query_dim=query_dim, heads=8, dim_head=query_dim // 8, dropout=dropout, bias=True)
        self.attn._use_memory_efficient_attention_xformers = True

    def forward(self, hidden_states):
        assert hidden_states.dim() == 4, f"Expected hidden_states to have ndim=4, but got ndim={hidden_states.dim()}."

        batch, channel, height, width = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm1(hidden_states)
        hidden_states = self.conv_in(hidden_states)
        hidden_states = rearrange(hidden_states, "b c h w -> b (h w) c")

        norm_hidden_states = self.norm2(hidden_states)
        hidden_states = self.attn(norm_hidden_states, attention_mask=None) + hidden_states
        hidden_states = self.ff(self.norm3(hidden_states)) + hidden_states

        hidden_states = rearrange(hidden_states, "b (h w) c -> b c h w", h=height, w=width)
        hidden_states = self.conv_out(hidden_states)

        hidden_states = hidden_states + residual
        return hidden_states


class DownEncoder2D(nn.Module):
    def __init__(
        self,
        in_channels=4 * 16,
        block_out_channels=[64, 128, 256, 256],
        downsample_factors=[2, 2, 2, 2],
        layers_per_block=2,
        norm_num_groups=32,
        attn_blocks=[1, 1, 1, 1],
        dropout: float = 0.0,
        act_fn="silu",
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        # in
        self.conv_in = nn.Conv2d(in_channels, block_out_channels[0], kernel_size=3, stride=1, padding=1)

        # down
        self.down_blocks = nn.ModuleList([])

        output_channels = block_out_channels[0]
        for i, block_out_channel in enumerate(block_out_channels):
            input_channels = output_channels
            output_channels = block_out_channel
            # is_final_block = i == len(block_out_channels) - 1

            down_block = ResnetBlock2D(
                in_channels=input_channels,
                out_channels=output_channels,
                downsample_factor=downsample_factors[i],
                norm_num_groups=norm_num_groups,
                dropout=dropout,
                act_fn=act_fn,
            )

            self.down_blocks.append(down_block)

            if attn_blocks[i] == 1:
                attention_block = AttentionBlock2D(query_dim=output_channels, dropout=dropout)
                self.down_blocks.append(attention_block)

        # out
        self.norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.act_fn_out = nn.ReLU()

    def forward(self, hidden_states):
        hidden_states = self.conv_in(hidden_states)

        # down
        for down_block in self.down_blocks:
            hidden_states = down_block(hidden_states)

        # post-process
        hidden_states = self.norm_out(hidden_states)
        hidden_states = self.act_fn_out(hidden_states)

        return hidden_states
