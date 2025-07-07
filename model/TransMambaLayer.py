import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from einops import rearrange, reduce, repeat
from mamba_ssm import Mamba
from utils.utils import *


class PatchAttention(nn.Module):
    def __init__(self, win_size, channel,
                 mask_flag=True, scale=None, attention_dropout=0.05):
        super(PatchAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.dropout = nn.Dropout(attention_dropout)
        self.window_size = win_size
        self.channel = channel

    def forward(self, queries_patch, keys_patch,
                patch_size, attn_mask):

        # Patch Representation
        B, L, H, E = queries_patch.shape  # batch*ch, patch_size, n_heads, d_model/n_heads
        scale_patch = self.scale or 1. / sqrt(E)
        scores_patch = torch.einsum("blhe,bshe->bhls", queries_patch,
                                          keys_patch)  # batch*ch, n_heads, patch_size, patch_size
        attn_patch = scale_patch * scores_patch
        patch_attention = self.dropout(torch.softmax(attn_patch, dim=-1))  # batch*ch, n_heads, patch_size, patch_size

        # Up-sampling
        patch_attention = patch_attention.repeat(1, 1, self.window_size // patch_size,
                                                             self.window_size // patch_size)
        patch_attention = reduce(patch_attention, '(b reduce_b) l m n-> b l m n', 'mean',
                                       reduce_b=self.channel)

        return patch_attention


class TemporalLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None):
        super(TemporalLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        self.inner_attention = attention
        self.n_heads = n_heads
        self.patch_query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.patch_key_projection = nn.Linear(d_model, d_keys * n_heads)

    def forward(self, x_patch, patch_size, attn_mask):
        H = self.n_heads
        B, L, M = x_patch.shape
        queries_patch, keys_patch = x_patch, x_patch
        queries_patch = self.patch_query_projection(queries_patch).view(B, L, H, -1)
        keys_patch = self.patch_key_projection(keys_patch).view(B, L, H, -1)

        patch_attention = self.inner_attention(
            queries_patch, keys_patch,
            patch_size, attn_mask)

        return patch_attention


def fft_embed(x, fre_embed):
    x_f = torch.fft.rfft(x, dim=1)  # x_f: (B, L_f, d), complex-valued
    x_f_cat = torch.cat((x_f.real, x_f.imag), dim=-1)  # (B, L_f, 2d)
    emb_x = fre_embed(x_f_cat)
    return x_f, emb_x

def loss_NTXent(z_anc, z_pos):
    batch_size, w, k = z_anc.size()
    T = 0.5
    x1 = z_anc.contiguous().view(batch_size, -1)
    x2 = z_pos.contiguous().view(batch_size, -1).detach()

    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()

    return loss

class FrequencyMamba(nn.Module):
    def __init__(self, config):
        super(FrequencyMamba, self).__init__()
        self.config = config
        self.MSE = nn.MSELoss()
        self.MAE = nn.L1Loss()
        self.fre_embed = nn.Linear(config.channel * 2, config.d_model)

        self.encoder = Mamba(d_model=config.d_model)
        self.get_r = nn.Linear(config.d_model, config.channel)
        self.get_i = nn.Linear(config.d_model, config.channel)

    def forward(self, x):
        x_fine, x_coarse = dual_augmentation(x, self.config.aug_rate)
        x_f, emb_x = fft_embed(x, self.fre_embed)
        x_coarse_f, emb_x_coarse = fft_embed(x_coarse, self.fre_embed)
        x_fine_f, emb_x_fine = fft_embed(x_fine, self.fre_embed)
        loss_cl = 0
        enc_out= self.encoder(emb_x)
        if self.config.mode == 'train':
            enc_out_coarse= self.encoder(emb_x_coarse)
            enc_out_fine= self.encoder(emb_x_fine)
            loss_coarse_cl = loss_NTXent(enc_out, enc_out_coarse)
            loss_fine_cl = loss_NTXent(enc_out, enc_out_fine)
            loss_cl = loss_coarse_cl + loss_fine_cl

        real = self.get_r(enc_out)
        imag = self.get_i(enc_out)
        output_f = torch.complex(real, imag)

        recon_loss_f = self.MSE(x_f.real, real) + self.MSE(x_f.imag, imag)
        output = torch.fft.irfft(output_f, n=self.config.win_size, dim=1)

        return output, loss_cl, recon_loss_f

