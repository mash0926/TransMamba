import torch
import torch.nn as nn

from einops import rearrange
from torch.nn.functional import dropout
from utils.utils import *
from .TransMambaLayer import PatchAttention, TemporalLayer, FrequencyMamba
from .embed import DataEmbedding
from .RevIN import RevIN
import time
from tkinter import _flatten

class Temporal_Encoder(nn.Module):
    def __init__(self, Temporal_layers, norm_layer=None):
        super(Temporal_Encoder, self).__init__()
        self.temporal_attn_layers = nn.ModuleList(Temporal_layers)
        self.layer_norm = norm_layer

    def forward(self, x_patch, patch_size, attn_mask=None):
        attention_list = []
        for temporal_layer in self.temporal_attn_layers:
            patch_attention = (
                temporal_layer(x_patch, patch_size, attn_mask=attn_mask))
            attention_list.append(patch_attention)

        return attention_list

class TransMamba(nn.Module):
    def __init__(self, config, output_attention=True):
        super(TransMamba, self).__init__()

        self.config = config
        self.patch_size_high = config.patch_size_high
        self.patch_size_low = config.patch_size_low
        self.aug_rate = config.aug_rate
        self.channel = config.channel
        self.win_size = config.win_size
        self.output_attention = output_attention
        if self.config.revin == 1:
            self.revin_layer = RevIN(self.config.channel)

        # Patch List Embedding
        self.embedding_high_patch = nn.ModuleList()
        self.embedding_low_patch = nn.ModuleList()

        for i, patch_size_high in enumerate(self.patch_size_high):
            self.embedding_high_patch.append(DataEmbedding(self.win_size // patch_size_high, config.d_model, config.dropout))

        for i, patch_size_low in enumerate(self.patch_size_low):
            self.embedding_low_patch.append(DataEmbedding(self.win_size // patch_size_low, config.d_model, config.dropout))

        self.embedding_window_size = DataEmbedding(config.channel, config.d_model, config.dropout)

        # Dual Domain Encoder
        self.tem_encoder = Temporal_Encoder(
            [
                TemporalLayer(
                    PatchAttention(win_size=config.win_size,
                                   channel=config.channel,
                                   mask_flag=False,
                                   attention_dropout=config.dropout),
                    d_model=config.d_model,
                    n_heads=config.n_heads)
                for l in range(config.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(config.d_model)
        )

        self.fre_encoder = FrequencyMamba(config)

    def forward(self, x):
        # Normalization
        if self.config.revin == 1:  # Instance Normalization Operation
            x = self.revin_layer(x, 'norm')

        # Time Domain
        high_attention_mean = []
        low_attention_mean = []
        x_reshape = x.permute(0, 2, 1)  # Batch channel win_size
        # High Resolution Patch
        for high_index, high_size in enumerate(self.patch_size_high):
            x_high_patch =x_reshape # Batch channel win_size
            x_high_patch = rearrange(x_high_patch, 'b c (p n) -> (b c) p n', p=high_size) # Batch channel patch num
            x_high_patch = self.embedding_high_patch[high_index](x_high_patch) # Batch channel patch d_model
            high_attention = self.tem_encoder(x_high_patch, high_size) # Batch channel win win
            high_attention_mean.append(high_attention)
        high_attention_mean = list(_flatten(high_attention_mean))

        # Low Resolution Patch
        for low_index, low_size in enumerate(self.patch_size_low):
            x_low_patch =x_reshape # Batch channel win_size
            x_low_patch = rearrange(x_low_patch, 'b c (p n) -> (b c) p n', p=low_size) # Batch channel patch num
            x_low_patch = self.embedding_low_patch[low_index](x_low_patch) # Batch channel patch d_model
            low_attention = self.tem_encoder(x_low_patch, low_size) # Batch channel win win
            low_attention_mean.append(low_attention)

        low_attention_mean = list(_flatten(low_attention_mean))

        # Frequency Domain
        output, loss_cl, recon_loss_f = self.fre_encoder(x)
        # DeNormalization
        if self.config.revin == 1:
            output = self.revin_layer(output, 'denorm')

        return high_attention_mean, low_attention_mean, output, loss_cl, recon_loss_f
