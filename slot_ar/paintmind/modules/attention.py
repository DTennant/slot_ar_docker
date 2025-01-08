import torch, pdb
import numpy as np
from torch import nn, einsum
from einops import rearrange
from inspect import isfunction
from typing import Optional, Any
from collections import OrderedDict

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d    

class SlotCausalAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        # inner_dim = dim_head * heads
        # context_dim = default(context_dim, query_dim)

        # self.scale = dim_head ** -0.5
        self.heads = heads

        self.attn = nn.MultiheadAttention(query_dim, heads, dropout=dropout, batch_first=True)
        # self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        # self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        # self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        # self.to_out = nn.Sequential(
        #     nn.Linear(inner_dim, query_dim),
        #     nn.Dropout(dropout)
        # )
        
    def generate_attention_mask(self, n, m):
        # Create an initial mask of size (n + m) x (n + m) filled with zeros
        mask = np.zeros((n + m, n + m))

        # First n tokens can see themselves only
        mask[:n, :n] = 1

        # Second m tokens can see all first n tokens and themselves causally
        mask[n:, :n] = 1
        mask[n:, n:] = np.tril(np.ones((m, m)))

        return torch.from_numpy(mask)#.to(self.device)
        
    def forward(self, x, slots, attn_mask=None):
        # __import__("ipdb").set_trace()
        n, m = x.shape[1], slots.shape[1]
        attn_mask = self.generate_attention_mask(n, m)
        qkv = torch.cat((x, slots), dim=1)
        attn_output, attn_output_weights = self.attn(query=qkv, key=qkv, value=qkv, attn_mask=attn_mask.to(qkv.device).type(qkv.dtype))
        # return x, slots
        return attn_output[:, :n], attn_output[:, n:]


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, attn_mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))
        q = q * self.scale
        
        sim = q @ k.transpose(-2, -1)
        
        if attn_mask is not None:
            # __import__("ipdb").set_trace()
            # attn_mask.shape: batchsize, n, m
            attn_mask = attn_mask.unsqueeze(1).repeat(1, h, 1, 1)  # expand mask to match the shape
            attn_mask = rearrange(attn_mask, 'b h n m -> (b h) n m')
            sim = sim.masked_fill(attn_mask == 0, float('-inf'))

        sim = sim.softmax(dim=-1)

        out = sim @ v
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)    


class MemoryEfficientCrossAttention(nn.Module):
    # https://github.com/MatthieuTPHR/diffusers/blob/d80b531ff8060ec1ea982b65a1b8df70f73aa67c/src/diffusers/models/attention.py#L223
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        print(f"Setting up {self.__class__.__name__}. Query dim is {query_dim}, context_dim is {context_dim} and using "
              f"{heads} heads.")
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.heads = heads
        self.dim_head = dim_head

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim), 
            nn.Dropout(dropout)
        )
        self.attention_op: Optional[Any] = None

    def forward(self, x, context=None, attn_mask=None):
        assert attn_mask is None, "Attention mask is not supported for this attention module."
        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        b, _, _ = q.shape
        q, k, v = map(
            lambda t: t.unsqueeze(3)
            .reshape(b, t.shape[1], self.heads, self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b * self.heads, t.shape[1], self.dim_head)
            .contiguous(),
            (q, k, v),
        )

        out = xformers.ops.memory_efficient_attention(q, k, v, attn_bias=None, op=self.attention_op)

        out = (
            out.unsqueeze(0)
            .reshape(b, self.heads, out.shape[1], self.dim_head)
            .permute(0, 2, 1, 3)
            .reshape(b, out.shape[1], self.heads * self.dim_head)
        )
        return self.to_out(out)    

# copied fron TiTok
class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model,
            n_head,
            mlp_ratio = 4.0,
            act_layer = nn.GELU,
            norm_layer = nn.LayerNorm
        ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.mlp_ratio = mlp_ratio
        # optionally we can disable the FFN
        if mlp_ratio > 0:
            self.ln_2 = norm_layer(d_model)
            mlp_width = int(d_model * mlp_ratio)
            self.mlp = nn.Sequential(OrderedDict([
                ("c_fc", nn.Linear(d_model, mlp_width)),
                ("gelu", act_layer()),
                ("c_proj", nn.Linear(mlp_width, d_model))
            ]))

    def attention(
            self,
            x: torch.Tensor
    ):
        return self.attn(x, x, x, need_weights=False)[0]

    def forward(
            self,
            x: torch.Tensor,
    ):
        attn_output = self.attention(x=self.ln_1(x))
        x = x + attn_output
        if self.mlp_ratio > 0:
            x = x + self.mlp(self.ln_2(x))
        return x