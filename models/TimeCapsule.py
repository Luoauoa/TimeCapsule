__all__ = ['TimeCapsule']

import torch
import torch.nn as nn
import copy
import numpy as np
from utils.tools import ema_update, mean_filter
from layers.SelfAttention_Family import FullAttention, AttentionLayer, MoAttention
from layers.PatchTST_layers import *
from layers.RevIN import RevIN


class Model(nn.Module):
    def __init__(self, configs, d_inputs, d_folds, d_compress, pe='zeros', learn_pe=True):
        """

        """
        super(Model, self).__init__()
        self.d_folds = d_folds
        self.rev_in = configs.revin
        self.pred_len = configs.pred_len
        self.L = configs.level_dim
        if self.rev_in:
            self.revin_layer = RevIN(d_inputs[-1], affine=True, subtract_last=False)
        self.jepa = configs.jepa

        dropout = configs.dropout
        d_model = configs.d_model
        d_ff = configs.d_ff
        n_heads = configs.n_heads
        output_attn = configs.output_attention
        self.x_encoder = CapEncoder(d_inputs, d_folds, d_compress, d_ff=d_ff, n_block=configs.n_block, dropout=dropout,
                                    d_model=d_model, n_heads=n_heads, pe=pe, learn_pe=learn_pe, output_attn=output_attn)

        self.decoder = CapDecoder(d_compress, d_inputs, self.pred_len, d_ff, dropout=dropout)
        if self.jepa:
            self.y_encoder = copy.deepcopy(self.x_encoder).requires_grad_(False)

        d = np.prod(d_compress)
        self.emb_predictor = nn.Linear(d, d)
        self.real_predictor = nn.Linear(configs.pred_len, configs.pred_len)
        self.level_weight = nn.Linear(d_inputs[1], 1, bias=False) 
        
    def obtain_res(self, store):
        m_1, m_2, m_3 = store['ms']
        r_1, r_2, r_3 = store['rs']
        e_1, e_2, e_3 = store['es']
        residuals = []
        res_1 = m_1.transpose(1, 2) - r_1.transpose(1, 2) @ e_1.T
        residuals.append(res_1)
        res_2 = m_2.transpose(1, 2) - r_2.transpose(1, 2) @ e_2.T
        residuals.append(res_2)
        res_3 = m_3.transpose(1, 2) - r_3.transpose(1, 2) @ e_3.T # (B, T_c*V_c, L) 
        residuals.append(res_3)

        return residuals
    
    @torch.no_grad()
    def jepa_forward(self, shape, y):
        B, T_x, V = shape  # input shape
        T_y = y.shape[1]
        n = None
        with torch.no_grad():
            if self.rev_in:
                y = self.revin_layer(y, 'norm').view(B, T_y, V)
            y = y.unsqueeze(-1).repeat(1, 1, 1, self.L)
            if T_y < T_x:
                pad_content = torch.zeros(size=(B, (T_x - T_y), V, self.L), requires_grad=False).cuda()
                y_new = torch.cat([y, pad_content], dim=1)
            elif T_y > T_x:
                n = T_y // T_x
                t = T_x * (n + 1) - T_y
                pad_content = torch.zeros(size=(B, t, V, self.L), requires_grad=False).cuda()
                y_new = torch.cat([y, pad_content], dim=1).chunk(n+1, dim=1)  # list of tensors (n+1, B, T_x, V, L)
            else: y_new = y
            if n is None:
                capsule_y, store_y = self.y_encoder(y_new)
            else:
                y_chunk = y_new[0]
                for i in range(1, len(y_new)):
                    y_chunk = ema_update(y_chunk, y_new[i], type='value')
                capsule_y, store_y = self.y_encoder(y_chunk)
        return capsule_y, store_y

    def forward(self, x):
        B = x.shape[0]  # (B, T, V)
        Y_tilde = None
        if self.rev_in:
            x = self.revin_layer(x, 'norm')
        x = x.unsqueeze(-1).repeat(1, 1, 1, self.L)  # if L > 1, a proper initialization is needed
        capsule_x, store_x = self.x_encoder(x)
        residuals = self.obtain_res(store_x)

        rep = self.emb_predictor(capsule_x.reshape(B, -1)).view(capsule_x.shape)
        Y_tilde = self.decoder(rep, residuals)  # (B, L, V, T)

        y_top = self.real_predictor(Y_tilde[-1]).permute(0, 3, 2, 1)
        y_hat = self.level_weight(y_top).squeeze()  # (B, T_pred, V)
        if self.rev_in:
            y_hat = self.revin_layer(y_hat, 'denorm')
        return y_hat, rep, Y_tilde, store_x
    

class CapEncoder(nn.Module):
    def __init__(self, d_inputs, d_folds, d_compress, n_block=1, d_model=128, d_ff=256, n_heads=4, output_attn=False,
                 pe='zeros', learn_pe=True, add_noise=True, dropout=0.1):
        super(CapEncoder, self).__init__()
        self.d_compress = d_compress
        self.d_model = d_model

        self.input_proj1 = nn.Linear(d_folds[0], d_model)
        self.input_proj2 = nn.Linear(d_folds[1], d_model)
        self.input_proj3 = nn.Linear(d_folds[2], d_model)

        self.Mo1Trans = TransBlock(d_model, n_heads, attn_type='mmo', output_attn=output_attn, d_ff=d_ff, d_folds=d_folds, dropout=dropout,
                                   d_trans=d_inputs[0] * 4, d_compress=d_compress[0], d_input=d_inputs[0], mode=1)
        self.Mo2Trans = TransBlock(d_model, n_heads, attn_type='mmo', output_attn=output_attn, d_ff=d_ff, d_folds=d_folds, dropout=dropout,
                                   d_trans=d_inputs[1] * 4, d_compress=d_compress[1], d_input=d_inputs[1], mode=2)
        self.Mo3Trans = TransBlock(d_model, n_heads, attn_type='mmo', output_attn=output_attn, d_ff=d_ff, d_folds=d_folds, dropout=dropout,
                                   d_trans=d_inputs[2] * 4, d_compress=d_compress[2], d_input=d_inputs[2], mode=3)
        
        self.tunnel1 = nn.ModuleList([TransBlock(d_model, n_heads, d_folds=d_folds, d_compress=d_compress[0], d_ff=d_ff, mode=1, dropout=dropout) for _ in range(n_block)])
        self.tunnel2 = nn.ModuleList([TransBlock(d_model, n_heads, d_folds=d_folds, d_compress=d_compress[1], d_ff=d_ff, mode=2, dropout=dropout) for _ in range(n_block)])
        self.tunnel3 = nn.ModuleList([TransBlock(d_model, n_heads, d_folds=d_folds, d_compress=d_compress[2], d_ff=d_ff, mode=3, dropout=dropout) for _ in range(n_block)])
        
        self.add_noise = add_noise
        self.pos_embedding = positional_encoding(pe, learn_pe, q_len=d_inputs[0], d_model=d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, V, L = x.shape  # original shape (batch, time, variate, level)
        T_c, L_c, V_c = self.d_compress
        store = {}

        # encode travel
        m_1 = x.view(B, T, -1)

        # temporal compress
        r_1 = self.input_proj1(m_1)
        if self.add_noise:
            r_1 += torch.randn_like(r_1)  

        # embedding
        r_1 = self.dropout(r_1 + self.pos_embedding)
        # r_1 = self.dropout(r_1)

        r_1, attn_1, e_1 = self.Mo1Trans(r_1)  # r1: (B, T_c, d_fold[0] = V*L)
        for mod in self.tunnel1:
            r_1, attn_1, _ = mod(r_1)  # (B, T_c, V*L)
        m_2 = r_1.reshape(B, T_c, V, L).transpose(1, 3).reshape(B, L, -1) 

        # level compress
        r_2 = self.input_proj2(m_2)
        if self.add_noise:
            r_2 += torch.randn_like(r_2) 
        r_2, attn_2, e_2 = self.Mo2Trans(r_2)
        for mod in self.tunnel2:
            r_2, attn_2, _ = mod(r_2)    # (B, L_c, T_c*V)
        m_3 = r_2.reshape(B, L_c, T_c, V).transpose(1, 3).reshape(B, V, -1)

        # variable compress
        r_3 = self.input_proj3(m_3)  # d_fold3=T_c*V_c
        if self.add_noise:
            r_3 += torch.randn_like(r_3) 
        r_3, attn_3, e_3 = self.Mo3Trans(r_3)
        for mod in self.tunnel3:
            r_3, attn_3, _ = mod(r_3)    # (B, V_c, d_fold3)
        assert r_3.shape == (B, V_c, T_c * L_c)

        store['ms'] = [m_1, m_2, m_3]
        store['rs'] = [r_1, r_2, r_3]
        store['attns'] = [attn_1, attn_2, attn_3]
        store['es'] = [e_1, e_2, e_3]
        capsule_x = r_3.view(B, V_c, L_c, T_c).permute(0, 3, 2, 1) 

        return capsule_x, store


class TransBlock(nn.Module):
    def __init__(self, d_model, n_heads, attn_type='normal', d_ff=256, d_k=None, d_v=None, dropout=0.1, output_attn=False,
                 # the following params are required for Multi-mode transformer
                 d_trans=None, d_folds=None, d_compress=None, d_input=None, mode=None):
        super(TransBlock, self).__init__()

        d = d_folds[mode - 1]
        if attn_type == 'normal':
            self.MSA = AttentionLayer(
                FullAttention(output_attention=output_attn), d_model, n_heads, d
            )
        elif attn_type == 'mmo':
            self.MSA = MoAttention(
                FullAttention(output_attention=output_attn), n_heads, d_model, d_trans, d_compress, d_input, d
            )
        else: raise ValueError("Invalid attention type, either 'mmo' or 'normal' is permitted")

        self.norm_attn = nn.BatchNorm1d(d_compress)
        self.norm_ffn = nn.BatchNorm1d(d_compress)

        self.FFN = nn.Sequential(nn.Linear(d, d_ff),
                                nn.GELU(),
                                nn.Dropout(dropout),
                                nn.Linear(d_ff, d))

        self.attn_type = attn_type
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        factor = None
        # Multi-Head Attention
        if self.attn_type == 'normal':
            x1, attn = self.MSA(x, x, x)
        else:
            x1, attn, factor = self.MSA(x)
        x1 = self.norm_attn(x1)
        # FFN
        # Add & Norm
        x2 = self.dropout(self.FFN(x1) + x1)
        x2 = self.norm_ffn(x2)

        return x2, attn, factor


class CapDecoder(nn.Module):
    def __init__(self, d_compress, d_inputs, d_pre, d_ff=256, dropout=0.1):
        super(CapDecoder, self).__init__()
        self.receiver1 = DecoderBlock(d_ff, d_compress[0] + d_inputs[0], d_pre, dropout=dropout)
        self.receiver2 = DecoderBlock(d_ff, d_compress[1] + d_inputs[1], d_inputs[1], dropout=dropout)
        self.receiver3 = DecoderBlock(d_ff, d_compress[2] + d_inputs[2], d_inputs[2], dropout=dropout)

        self.d_inputs = d_inputs
        self.pred_len = d_pre

    def forward(self, y_tilde, residuals):
        B, T_c, L_c, V_c = y_tilde.shape

        Ys = []
        _, L, V = self.d_inputs
        y_3 = self.receiver3(torch.cat([y_tilde.reshape(B, -1, V_c), residuals[2]], dim=-1))\
            .view(B, T_c, L_c, V)  
        Ys.append(y_3)
        y_2 = self.receiver2(torch.cat([y_3.transpose(2, 3).reshape(B, -1, L_c), residuals[1]], dim=-1))\
            .view(B, T_c, V, L)  
        Ys.append(y_2)
        y_1 = self.receiver1(torch.cat([y_2.transpose(1, 3).reshape(B, -1, T_c), residuals[0]], dim=-1))\
            .view(B, L, V, self.pred_len)  
        Ys.append(y_1)

        return Ys


class DecoderBlock(nn.Module):
    def __init__(self, d_ff, d_in, d_input, dropout=0.1):
        super(DecoderBlock, self).__init__()
        self.MLP = nn.Sequential(nn.Linear(d_in, d_ff),
                                 nn.Linear(d_ff, d_ff),
                                 nn.Dropout(dropout),
                                 nn.GELU(),
                                 nn.Linear(d_ff, d_input))

    def forward(self, x):
        out = self.MLP(x)

        return out
