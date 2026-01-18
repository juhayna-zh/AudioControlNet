from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from einops.layers.torch import Rearrange

from audio_controlnet.ext.rotary_embeddings import apply_rope
from audio_controlnet.model.low_level import MLP, ChannelLastConv1d, ConvMLP


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return x * (1 + scale) + shift  # scale is actually the add term for x (res connect for modulation)


def attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
):
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    if attention_mask is not None:
        # F.scaled_dot_product_attention expects mask in shape [batch, nheads, q_len, k_len]
        # and additive form: masked positions = -inf
        # if you pass bool mask, True = keep, False = mask out
        # so make sure the user passes correct format
        attn_mask = attention_mask
    else:
        attn_mask = None

    with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

    out = rearrange(out, "b h n d -> b n (h d)").contiguous()
    return out


class SelfAttention(nn.Module):

    def __init__(self, dim: int, nheads: int, use_lora: bool = False, lora_rank: int | None = None):
        super().__init__()
        self.dim = dim
        self.nheads = nheads

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.q_norm = nn.RMSNorm(dim // nheads)
        self.k_norm = nn.RMSNorm(dim // nheads)

        self.split_into_heads = Rearrange('b n (h d j) -> b h n d j',
                                          h=nheads,
                                          d=dim // nheads,
                                          j=3)

        self.use_lora = use_lora
        if use_lora:
            assert lora_rank != None
            self.qkv_lora = LoRALinear(dim, dim * 3, lora_rank=lora_rank, bias=True)

    def pre_attention(  # get qkv for input x, apply rotary pos embedding if needed
            self, x: torch.Tensor,
            rot: Optional[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: batch_size * n_tokens * n_channels
        if self.use_lora:
            qkv = self.qkv(x) + self.qkv_lora(x)
        else:
            qkv = self.qkv(x)
        q, k, v = self.split_into_heads(qkv).chunk(3, dim=-1)  # chunk: split the input into 3 components 
        q = q.squeeze(-1)
        k = k.squeeze(-1)
        v = v.squeeze(-1)
        q = self.q_norm(q)
        k = self.k_norm(k)

        if rot is not None:
            q = apply_rope(q, rot)
            k = apply_rope(k, rot)

        return q, k, v

    def forward(
            self,
            x: torch.Tensor,  # batch_size * n_tokens * n_channels
    ) -> torch.Tensor:
        q, v, k = self.pre_attention(x)
        out = attention(q, k, v)  
        return out


class MMDitSingleBlock(nn.Module):

    def __init__(self,
                 dim: int,
                 nhead: int,
                 mlp_ratio: float = 4.0,
                 pre_only: bool = False,
                 kernel_size: int = 7,
                 padding: int = 3,
                 use_lora: bool = False,
                 lora_rank: int | None = None):
        super().__init__()
        self.use_lora = use_lora
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = SelfAttention(dim, nhead, use_lora=use_lora, lora_rank=lora_rank)

        self.pre_only = pre_only
        if pre_only:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))
        else:
            if kernel_size == 1:
                self.linear1 = nn.Linear(dim, dim)
            else:
                self.linear1 = ChannelLastConv1d(dim, dim, kernel_size=kernel_size, padding=padding)
            self.norm2 = nn.LayerNorm(dim, elementwise_affine=False)

            if kernel_size == 1:
                self.ffn = MLP(dim, int(dim * mlp_ratio))
            else:
                self.ffn = ConvMLP(dim,
                                   int(dim * mlp_ratio),
                                   kernel_size=kernel_size,
                                   padding=padding)

            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 6 * dim, bias=True))

    def pre_attention(self, x: torch.Tensor, c: torch.Tensor, rot: Optional[torch.Tensor]):
        """get qkv from x and modulation coefficients from condition"""
        # x: BS * N * D
        # cond: BS * D
        modulation = self.adaLN_modulation(c)  # get modulation coefficients 
        if self.pre_only:
            (shift_msa, scale_msa) = modulation.chunk(2, dim=-1)
            gate_msa = shift_mlp = scale_mlp = gate_mlp = None
        else:
            (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp,
             gate_mlp) = modulation.chunk(6, dim=-1)

        x = modulate(self.norm1(x), shift_msa, scale_msa)  # first AdaLN
        q, k, v = self.attn.pre_attention(x, rot)  # linear for qkv
        return (q, k, v), (gate_msa, shift_mlp, scale_mlp, gate_mlp)

    def post_attention(self, x: torch.Tensor, attn_out: torch.Tensor, c: tuple[torch.Tensor], latent_adapters=None):
        if self.pre_only:
            return x

        (gate_msa, shift_mlp, scale_mlp, gate_mlp) = c
        x = x + self.linear1(attn_out) * gate_msa  # first linear/ConvMLP & scaling & residual
        
        if latent_adapters is not None:
            x = x + sum([latent_adapter(x) for latent_adapter in latent_adapters])
            
        r = modulate(self.norm2(x), shift_mlp, scale_mlp)  # second AdaLN
        x = x + self.ffn(r) * gate_mlp  # second linear/ConvMLP & scaling & residual 

        return x

    def forward(self, x: torch.Tensor, cond: torch.Tensor,
                rot: Optional[torch.Tensor], latent_adapters=None) -> torch.Tensor:
        # x: BS * N * D
        # cond: BS * D
        x_qkv, x_conditions = self.pre_attention(x, cond, rot) # first AdaLN
        attn_out = attention(*x_qkv) # self-attn block
        x = self.post_attention(x, attn_out, x_conditions, latent_adapters=latent_adapters) # the scaling and second AdaLN

        return x


class JointBlock(nn.Module):

    def __init__(self, dim: int, nhead: int, mlp_ratio: float = 4.0, pre_only: bool = False, use_lora: bool = False, lora_rank: int | None = None):
        super().__init__()
        self.pre_only = pre_only
        self.latent_block = MMDitSingleBlock(dim,
                                             nhead,
                                             mlp_ratio,
                                             pre_only=False,
                                             kernel_size=3,
                                             padding=1,
                                             use_lora=use_lora,
                                             lora_rank=lora_rank)
        self.text_block = MMDitSingleBlock(dim, nhead, mlp_ratio, pre_only=pre_only, kernel_size=1, use_lora=use_lora, lora_rank=lora_rank)

    def forward(self, latent: torch.Tensor, text_f: torch.Tensor,
                global_c: torch.Tensor, extended_c: torch.Tensor, 
                latent_rot: torch.Tensor, text_rot: torch.Tensor, latent_adapters = None,
                ) -> tuple[torch.Tensor, torch.Tensor]:  
        # latent: BS * N1 * D
        # c: BS * (1/N) * D
        # latent: torch.Size([1, 430, 896])
        # extended_c: torch.Size([1, 1, 896])
        # latent_rot: torch.Size([1, 430, 32, 2, 2])
        # text_f: torch.Size([1, 77, 896])
        # global_c: torch.Size([1, 1, 896])
        # text_rot: torch.Size([1, 77, 32, 2, 2])
        x_qkv, x_mod = self.latent_block.pre_attention(latent, extended_c, rot=latent_rot)  # fine-grained features are only used for the audio branch
        t_qkv, t_mod = self.text_block.pre_attention(text_f, global_c, rot=text_rot)  
        # x_qkv: [torch.Size([1, 14, 430, 64]), torch.Size([1, 14, 430, 64]), torch.Size([1, 14, 430, 64])]
        # t_qkv: [torch.Size([1, 14, 77, 64]), torch.Size([1, 14, 77, 64]), torch.Size([1, 14, 77, 64])]

        latent_len = latent.shape[1]
        text_len = text_f.shape[1]

        joint_qkv = [torch.cat([x_qkv[i], t_qkv[i]], dim=2) for i in range(3)]
        # joint_qkv: [torch.Size([1, 14, 507, 64]), torch.Size([1, 14, 507, 64]), torch.Size([1, 14, 507, 64])]

        attn_out = attention(*joint_qkv)  # core of joint block: joint attention
        x_attn_out = attn_out[:, :latent_len]  
        t_attn_out = attn_out[:, latent_len:]
        # attn_out: torch.Size([1, 507, 896])
        # x_attn_out: torch.Size([1, 430, 896])
        # t_attn_out: torch.Size([1, 77, 896])

        latent = self.latent_block.post_attention(latent, x_attn_out, x_mod, latent_adapters)
        # latent: torch.Size([1, 430, 896])
        if not self.pre_only:
            text_f = self.text_block.post_attention(text_f, t_attn_out, t_mod)  # for pre-only layer we don't do post attention for condition features

        return latent, text_f


class FinalBlock(nn.Module):

    def __init__(self, dim, out_dim):
        super().__init__()
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 2 * dim, bias=True))
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.conv = ChannelLastConv1d(dim, out_dim, kernel_size=7, padding=3)

    def forward(self, latent, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        latent = modulate(self.norm(latent), shift, scale)
        latent = self.conv(latent)
        return latent



class Conv1dFeatureExtractor(nn.Module):
    def __init__(self, conv_configs, activation="SiLU"):
        """
        Args:
            conv_configs (list of list): 每一层 Conv1d 的参数（位置参数形式）; 顺序: in_channels, out_channels, kernel_size, stride, padding
            activation (str): 激活函数名，必须是 torch.nn 里存在的类，比如 "ReLU", "SiLU", "GELU"
        Example:
            conv_configs:
            - [2, 16, 3, 2, 1]      # conv1d_1
            - [16, 16, 3, 1, 1]     # conv1d_2
            - [16, 128, 3, 2, 1]    # conv1d_3
            - [128, 128, 3, 1, 1]   # conv1d_4
            - [128, 256, 3, 2, 1]   # conv1d_5
            
            activation: SiLU
        
        """
        super().__init__()
        layers = []

        # 获取激活层类（构造时实例化）
        act_layer = getattr(nn, activation)()

        for cfg in conv_configs:
            conv = nn.Conv1d(*cfg)  # 用位置参数
            layers.append(conv)
            layers.append(act_layer)

        # 最后一层不加激活
        if len(layers) > 0:
            layers = layers[:-1]

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



class CrossAttention(nn.Module):
    def __init__(self, dim: int, nheads: int):
        super().__init__()
        self.dim = dim
        self.nheads = nheads

        self.q_proj = nn.Linear(dim, dim, bias=True)
        self.kv_proj = nn.Linear(dim, dim * 2, bias=True)

        self.q_norm = nn.RMSNorm(dim // nheads)
        self.k_norm = nn.RMSNorm(dim // nheads)

        self.split_q = Rearrange('b n (h d) -> b h n d', h=nheads, d=dim // nheads)
        self.split_kv = Rearrange('b n (h d j) -> b h n d j',
                                  h=nheads,
                                  d=dim // nheads,
                                  j=2)

    def pre_attention(
        self,
        x: torch.Tensor,       # queries
        context: torch.Tensor, # keys + values
        rot: Optional[torch.Tensor] = None,
        rot_kv: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Q from x, K and V from context.
        """
        # Q from input
        q = self.q_proj(x)
        q = self.split_q(q)
        q = self.q_norm(q)

        # K, V from context
        kv = self.kv_proj(context)
        k, v = self.split_kv(kv).unbind(dim=-1)  # split last dim into (k, v)
        k = self.k_norm(k)
        # optional rotary embedding
        if rot is not None:
            q = apply_rope(q, rot)
            if rot_kv is None:
                rot_kv = rot
            k = apply_rope(k, rot_kv)

        return q, k, v

    def forward(
        self,
        x: torch.Tensor,       # (batch, n_tokens_q, dim)
        context: torch.Tensor, # (batch, n_tokens_ctx, dim)
        rot: Optional[torch.Tensor] = None,
        rot_kv: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None, # (batch, 1, q_len, k_len) or broadcastable
    ) -> torch.Tensor:
        q, k, v = self.pre_attention(x, context, rot=rot, rot_kv=rot_kv)
        out = attention(q, k, v, attention_mask=attention_mask)  # (batch, n_tokens_q, dim)
        return out

class CrossAttentionAdapter(nn.Module):
    def __init__(self, dim: int, nheads: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=True)  # 对输入做LayerNorm
        self.cross_attn = CrossAttention(dim, nheads)           # 内部cross attention

    def forward(
        self,
        x: torch.Tensor,       # (batch, n_tokens_q, dim)
        context: torch.Tensor, # (batch, n_tokens_ctx, dim)
        rot: Optional[torch.Tensor] = None,
        rot_kv: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        直接执行cross attention
        """
        x_norm = self.norm(x)
        out = self.cross_attn(x_norm, context, rot, rot_kv)
        return out

class LoRALinear(nn.Module):
    def __init__(self, input_features: int, output_features: int, lora_rank: int = 4, bias: bool = True):
        super().__init__()
        self.encoder = nn.Linear(input_features, lora_rank, bias = False)
        self.decoder = nn.Linear(lora_rank, output_features, bias = bias)
        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.encoder.weight, 0, 1)
        nn.init.constant_(self.decoder.weight, 0)
        if self.decoder.bias is not None:
            nn.init.constant_(self.decoder.bias, 0)

    def forward(self, X: torch.Tensor):
        return self.decoder(self.encoder(X))