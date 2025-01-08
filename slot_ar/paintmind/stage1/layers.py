from torch import nn
import numpy as np
import torch, math, pdb
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from ..modules.mlp import SwiGLUFFNFused
from ..engine.util import instantiate_from_config
from ..modules.attention import CrossAttention, MemoryEfficientCrossAttention, SlotCausalAttention, XFORMERS_IS_AVAILBLE, ResidualAttentionBlock


def pair(t):

    return t if isinstance(t, tuple) else (t, t)


class FeedForward(nn.Module):
    def __init__(self, dim, mlp_dim, dropout=0.):
        super().__init__()
        self.w_1 = nn.Linear(dim, mlp_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(p=dropout)
        self.w_2 = nn.Linear(mlp_dim, dim)
    
    def forward(self, x):
        x = self.w_1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.w_2(x)

        return x


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Layer(nn.Module):
    ATTENTION_MODES = {
        "vanilla": CrossAttention,
        "xformer": MemoryEfficientCrossAttention,
        'slot': SlotCausalAttention
    }
    def __init__(self, dim, dim_head, mlp_dim, attn_type='normal', num_head=8, dropout=0.0):
        super().__init__()
        attn_mode = "xformer" if XFORMERS_IS_AVAILBLE else "vanilla"
        attn_mode = attn_mode if attn_type == 'normal' else 'slot'
        self.attn_mode = attn_mode
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.norm1 = nn.LayerNorm(dim)
        self.attn1 = attn_cls(query_dim=dim, heads=num_head, dim_head=dim_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ffnet = SwiGLUFFNFused(in_features=dim, hidden_features=mlp_dim)
        
        if self.attn_mode == 'slot':
            self.norm_slot1 = nn.LayerNorm(dim)
            self.norm_slot2 = nn.LayerNorm(dim)
        
    def forward(self, x, slots=None, attn_mask=None):
        # __import__("ipdb").set_trace()
        if self.attn_mode == 'slot':
            assert slots is not None
            
            x_, slots_ = self.attn1(self.norm1(x), self.norm_slot1(slots))
            x, slots = x + x_, slots + slots_
            
            x_slots = torch.cat((self.norm2(x), self.norm_slot2(slots)), dim=1)
            x_slots = self.ffnet(x_slots)
            x_, slots_ = x_slots[:, :x.size(1)], x_slots[:, x.size(1):]
            x, slots = x + x_, slots + slots_
            return x, slots

        x = self.attn1(self.norm1(x), attn_mask=attn_mask) + x
        x = self.ffnet(self.norm2(x)) + x

        return x

class DecoderLayer(Layer):

    def __init__(self, dim, dim_head, mlp_dim, num_head=8, dropout=0.0):
        
        super().__init__(dim, dim_head, mlp_dim, num_head, dropout)
        attn_mode = "xformer" if XFORMERS_IS_AVAILBLE else "vanilla"
        attn_cls = self.ATTENTION_MODES[attn_mode]
        self.attn2 = attn_cls(query_dim=dim, heads=num_head, dim_head=dim_head, dropout=dropout)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, slots):

        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), slots) + x
        x = self.ffnet(self.norm3(x)) + x

        return x

class Transformer(nn.Module):
    def __init__(self, layer_type, dim, depth, num_head, dim_head, mlp_dim, attn_type='normal', dropout=0.):
        super().__init__()

        assert layer_type in ['normal', 'dec_layer']
        layers = {'normal': Layer, 'dec_layer': DecoderLayer}
        self.attn_type = attn_type
        self.layers = nn.ModuleList([layers[layer_type](dim, dim_head, mlp_dim, num_head=num_head, dropout=dropout, attn_type=attn_type) for i in range(depth)])
    
    def forward(self, x, slots = None, attn_mask=None):
        
        for i, layer in enumerate(self.layers):
            if self.attn_type == 'normal':
                x = layer(x, slots, attn_mask=attn_mask)
            else: # 'causal'
                x, slots = layer(x, slots, attn_mask=attn_mask)
        if self.attn_type == 'causal':
            return x, slots
        return x


class Encoder(nn.Module):

    def __init__(self, image_size, layer_type, num_slots,
                 patch_size, dim, depth, num_head, mlp_dim,
                 in_channels=3, out_channels=3, dim_head=64, 
                 visual_encoder_config = None, dropout=0., attn_type='normal', **kwargs):

        super().__init__()
        self.backbone = None
        if visual_encoder_config is not None:
            self.backbone = instantiate_from_config(visual_encoder_config)

        self.image_size = image_size
        self.patch_size = patch_size

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size, bias=False),
            Rearrange('b c h w -> b (h w) c'),
        )
        
        scale = dim ** -0.5
        num_patches = (image_size // patch_size) ** 2
        self.num_patches = num_patches

        self.position_embedding = nn.Parameter(torch.randn(1, num_patches + num_slots, dim) * scale)
        self.norm_pre = nn.LayerNorm(dim)
        self.attn_type = attn_type
        if attn_type == 'causal':
            self.slot_position_embedding = nn.Parameter(torch.randn(1, num_slots, dim) * scale)
            self.norm_slot_pre = nn.LayerNorm(dim)
        self.transformer = Transformer(layer_type, dim, depth, num_head, dim_head, mlp_dim, attn_type=self.attn_type, dropout=dropout)
        
        self.initialize_weights()

    def initialize_weights(self):

        if self.backbone:
            assert self.backbone.is_trainable is False
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.norm_pre.apply(self._init_weights)
        for m in self.transformer.parameters():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def generate_attention_mask(self, n, m):
        # Create an initial mask of size (n + m) x (n + m) filled with zeros
        mask = np.zeros((n + m, n + m))

        # First n tokens can see themselves only
        mask[:n, :n] = 1

        # Second m tokens can see all first n tokens and themselves causally
        mask[n:, :n] = 1
        mask[n:, n:] = np.tril(np.ones((m, m)))

        return torch.from_numpy(mask)#.to(self.device)

    def forward(self, x, slots):
        if self.backbone is not None:
            x = self.backbone(x)
        x = self.to_patch_embedding(x)
        if self.attn_type == 'causal':
            x = x + self.position_embedding
            slots = slots + self.slot_position_embedding
            x, slots = self.norm_pre(x), self.norm_slot_pre(slots)
            x, slots = self.transformer(x, slots)
            return x, slots
            
        x = torch.cat((x + self.position_embedding[:, :self.num_patches], 
                       slots + self.position_embedding[:, self.num_patches:]), dim=1)
        x = self.norm_pre(x)
        attn_mask = self.generate_attention_mask(x.size(1) - slots.size(1), slots.size(1))
        attn_mask = attn_mask.unsqueeze(0).expand(x.size(0), -1, -1).to(x.device).long()
        # __import__("ipdb").set_trace()
        x = self.transformer(x, None, attn_mask=attn_mask)
        
        return x
       
class Decoder(nn.Module):
    def __init__(self, layer_type, image_size, patch_size, in_channels, dim, num_slots,
                 depth, num_head, mlp_dim,  out_channels=3, dim_head=64, dropout=0.):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size

        assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'

        scale = dim ** -0.5
        num_patches = (image_size // patch_size) ** 2
        self.num_slots = num_slots
        self.num_patches = num_patches
        self.dim = dim

        self.mask_token = nn.Embedding(1, dim)
        self.position_embedding = nn.Embedding(num_patches, dim)

        self.transformer = Transformer(layer_type, dim, depth, num_head, dim_head, mlp_dim, dropout=dropout)
        self.norm = nn.LayerNorm(dim)
        self.proj = nn.Linear(dim, out_channels * patch_size * patch_size, bias=True)
        
        self.initialize_weights()

    @property
    def pos_embedding(self):
        
        return self.position_embedding

    def initialize_weights(self):
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, slots):
        
        bs = slots.size(0)
        position_embedding = repeat(self.position_embedding.weight, 'f c -> b f c', b=bs)

        mask_tokens = position_embedding + self.mask_token.weight.unsqueeze(0)

        z = torch.cat((mask_tokens, slots), dim=1)
        z = self.transformer(z)
        z = self.norm(z)

        x = z[:, :self.num_patches]
        x = self.proj(x)
        x = rearrange(x, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)', h=self.image_size//self.patch_size, p1=self.patch_size, p2=self.patch_size)
        
        return x


# class TiTokDecoder(nn.Module):
#     def __init__(self, image_size, patch_size, dim, num_slots,
#                  depth, num_head):
#         super().__init__()
        
#         self.image_size = image_size
#         self.patch_size = patch_size

#         assert image_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        
#         self.dim = dim
#         self.decoder_embed = nn.Linear(self.dim, self.dim, bias=True)

#         scale = dim ** -0.5
#         grid_size = image_size // patch_size
#         num_patches = grid_size ** 2 + 1
#         self.num_slots = num_slots
#         self.grid_size = grid_size
#         self.num_patches = num_patches
#         self.dim = dim
#         self.depth = depth

#         self.class_embedding = nn.Parameter(scale * torch.randn(1, dim))
#         self.position_embedding = nn.Parameter(
#             scale * torch.randn(num_patches, dim))
#         self.mask_token = nn.Parameter(scale * torch.randn(1, 1, dim))
#         self.slot_positional_embedding = nn.Parameter(
#             scale * torch.randn(self.num_slots, dim))
        
#         self.ln_pre = nn.LayerNorm(self.dim)
#         self.transformer = nn.ModuleList()
#         for i in range(depth):
#             self.transformer.append(ResidualAttentionBlock(
#                 dim, num_head, mlp_ratio=4.0
#             ))
#         self.ln_post = nn.LayerNorm(dim)

#         self.ffn = nn.Sequential(
#             nn.Conv2d(dim, 2 * dim, 1, padding=0, bias=True),
#             nn.Tanh(),
#             nn.Conv2d(2 * dim, 1024, 1, padding=0, bias=True),
#         )
#         self.conv_out = nn.Identity()

#         # self.initialize_weights()
    
#     def forward(self, slots):
#         # __import__("ipdb").set_trace()
#         slots = self.decoder_embed(slots)
#         batchsize, seq_len, _ = slots.shape

#         cls_tokens = repeat(self.class_embedding, 'n d -> b n d', b=batchsize)
#         mask_tokens = self.mask_token.repeat(batchsize, self.num_patches - 1, 1)
#         mask_tokens = torch.cat([cls_tokens, mask_tokens], dim=1)
#         mask_tokens = mask_tokens + self.position_embedding
#         slots = slots + self.slot_positional_embedding[:seq_len]
#         z = torch.cat([mask_tokens, slots], dim=1)
        
#         z = self.ln_pre(z)
#         z = z.permute(1, 0, 2)  # NLD -> LND
#         for i in range(self.depth):
#             z = self.transformer[i](z)
#         z = z.permute(1, 0, 2)  # LND -> NLD
#         z = z[:, 1:self.num_patches] # remove cls embed
#         z = self.ln_post(z)
#         # N L D -> N D H W
#         z = z.permute(0, 2, 1).reshape(batchsize, -1, self.grid_size, self.grid_size)
#         z = self.ffn(z.contiguous())
#         z = self.conv_out(z)
#         return z
