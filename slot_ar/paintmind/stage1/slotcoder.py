import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp, DropPath

import numpy as np


import math
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision
import torch
from diffusers import DDPMScheduler, UNet2DModel
from einops import rearrange

from .pos_embed import get_2d_sincos_pos_embed

class XAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q, k, v, mask=None):
        B, N, C = q.shape
        M = k.shape[1]
        q = self.q_proj(q).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.k_proj(k).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.v_proj(v).reshape(B, M, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask[:, None, None, :] == 0, float('-inf'))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
    

class DecoderBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.xattn = XAttention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        self.norm3 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path3 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, mem, mask=None):
        x = x + self.drop_path1(self.attn(self.norm1(x)))
        x = x + self.drop_path2(self.xattn(self.norm2(x), mem, mem, mask=mask))
        x = x + self.drop_path3(self.mlp(self.norm3(x)))
        return x


class SlotDecoder(nn.Module):
    def __init__(self, img_size=224, in_chans=3, patch_size=16, embed_dim=768, 
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches**.5), cls_token=False)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, slots, scores, use_mask=False):
        mask = (torch.zeros_like(scores).scatter_(1, scores.argmax(1, keepdim=True), 1).sum(-1).sum(-1) > 0).long().detach()
        if not use_mask:
            mask = None

        slots = self.decoder_embed(slots)

        x = self.mask_token.repeat(slots.shape[0], self.num_patches, 1)

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x, slots, mask)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x


def has_batchnorms(model):
    bn_types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)
    for name, module in model.named_modules():
        if isinstance(module, bn_types):
            return True
    return False

def _build_mlp_dino2d(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=True):
    if nlayers == 1:
        return nn.Conv2d(in_dim, bottleneck_dim, 1)
    else:
        layers = [nn.Conv2d(in_dim, hidden_dim, 1)]
        if use_bn:
            layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, 1))
            if use_bn:
                layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Conv2d(hidden_dim, bottleneck_dim, 1))
        return nn.Sequential(*layers)

def _build_mlp_moco(num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
    mlp = []
    for l in range(num_layers):
        dim1 = input_dim if l == 0 else mlp_dim
        dim2 = output_dim if l == num_layers - 1 else mlp_dim

        mlp.append(nn.Linear(dim1, dim2, bias=False))

        if l < num_layers - 1:
            mlp.append(nn.BatchNorm1d(dim2))
            mlp.append(nn.ReLU(inplace=True))
        elif last_bn:
            # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
            # for simplicity, we further removed gamma in BN
            mlp.append(nn.BatchNorm1d(dim2, affine=False))

    return nn.Sequential(*mlp)

class MoCoHead(nn.Module):
    def __init__(self, in_dim, nlayers=3, hidden_dim=4096, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp_moco(nlayers, in_dim, hidden_dim, bottleneck_dim)

    def forward(self, x):
        x = self.mlp(x)
        return x

class DINOHead2d(nn.Module):
    def __init__(self, in_dim, use_bn=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_mlp_dino2d(nlayers, in_dim, bottleneck_dim, hidden_dim, use_bn)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Conv2d) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        return x

class SemanticGrouping(nn.Module):
    def __init__(self, num_slots, dim_slot, eps=1e-7, **kwargs):
        super().__init__()
        self.num_slots = num_slots
        self.dim_slot = dim_slot
        self.eps = eps

        self.slot_embed = nn.Embedding(num_slots, dim_slot)

    def forward(self, key, val=None, temp=0.07):
        val = key if val is None else val
        # __import__("ipdb").set_trace()
        slots = self.slot_embed(torch.arange(0, self.num_slots, device=key.device)).unsqueeze(0).repeat(key.size(0), 1, 1)
        dots = torch.einsum('bkd,bdhw->bkhw', F.normalize(slots, dim=2), F.normalize(key, dim=1))
        attn = (dots / temp).softmax(dim=1) + self.eps
        slots = torch.einsum('bdhw,bkhw->bkd', val, attn / attn.sum(dim=(2, 3), keepdim=True))
        return slots, dots

class CausalSemanticGrouping(SemanticGrouping):
    def forward(self, key, val=None, temp=0.07):
        slots = self.slot_embed(torch.arange(0, self.num_slots, device=key.device)).unsqueeze(0).repeat(key.size(0), 1, 1)
        if key.dim() == 4:
            key = rearrange(key, 'b c h w -> b (h w) c')
        assert key.dim() == 3
        val = key if val is None else val

        slot_key = torch.cat([slots, key], dim=1) # (b, n_slots + n_patches, c)
        dots = torch.einsum('bkd,bcd->bkc', F.normalize(slots, dim=2), F.normalize(slot_key, dim=2)) # (b, n_slots, n_slots + n_patches)

        slot2slot_attn_mask = torch.tril(torch.ones(slots.size(1), slots.size(1), device=dots.device), diagonal=-1) # slot to slot attention, causal
        slot2key_attn_mask = torch.ones(slots.size(1), key.size(1), device=dots.device) # slot to key attention, non-causal
        attn_mask = torch.cat([slot2slot_attn_mask, slot2key_attn_mask], dim=1) # (n_slots, n_slots + n_patches)
        attn_mask = attn_mask.unsqueeze(0).repeat(dots.size(0), 1, 1)

        dots = dots.masked_fill(attn_mask == 0, float('-inf')) # (b, n_slots, n_slots + n_patches)
        attn = (dots / temp).softmax(dim=-1)# + self.eps 
        
        slot_val = torch.cat([slots, val], dim=1) # (b, n_slots + n_patches, c)
        slots = torch.einsum('bkc,bdk->bdc', slot_val, attn / (attn.sum(dim=2, keepdim=True) + self.eps))
        
        return slots, dots

class CausalSemanticGroupingPCA(SemanticGrouping):
    def forward(self, key, val=None, temp=0.07):
        slots = self.slot_embed(torch.arange(0, self.num_slots, device=key.device)).unsqueeze(0).repeat(key.size(0), 1, 1)
        key = rearrange(key, 'b c h w -> b (h w) c')
        val = key if val is None else val

        slot_key = torch.cat([slots, key], dim=1) # (b, n_slots + n_patches, c)
        dots = torch.einsum('bkd,bcd->bkc', F.normalize(slots, dim=2), F.normalize(slot_key, dim=2)) # (b, n_slots, n_slots + n_patches)

        slot2slot_attn_mask = torch.tril(torch.ones(slots.size(1), slots.size(1), device=dots.device), diagonal=-1) # slot to slot attention, causal
        slot2key_attn_mask = torch.ones(slots.size(1), key.size(1), device=dots.device) # slot to key attention, non-causal
        attn_mask = torch.cat([slot2slot_attn_mask, slot2key_attn_mask], dim=1) # (n_slots, n_slots + n_patches)
        attn_mask = attn_mask.unsqueeze(0).repeat(dots.size(0), 1, 1)

        dots = dots.masked_fill(attn_mask == 0, float('-inf')) # (b, n_slots, n_slots + n_patches)
        attn = (dots / temp).softmax(dim=-1)# + self.eps 
        
        slot_val = torch.cat([slots, val], dim=1) # (b, n_slots + n_patches, c)
        slots = torch.einsum('bkc,bdk->bdc', slot_val, attn / (attn.sum(dim=2, keepdim=True) + self.eps))
        
        # NOTE: first run slot over the image tokens.
        
        return slots, dots

class CausalSemanticComponents(SemanticGrouping):
    def __init__(self, num_slots, dim_slot, cond_num=4, eps=1e-7, **kwargs):
        super().__init__(num_slots, dim_slot, eps)
        self.cond_num = cond_num
    
    def forward(self, key, val=None, temp=0.07):
        # NOTE: this module will only allow the first slot to attend to image tokens, the other slots can only attend to the previous slots
        slots = self.slot_embed(torch.arange(0, self.num_slots, device=key.device)).unsqueeze(0).repeat(key.size(0), 1, 1)
        key = rearrange(key, 'b c h w -> b (h w) c')
        val = key if val is None else val

        slot_key = torch.cat([slots, key], dim=1) # (b, n_slots + n_patches, c)
        dots = torch.einsum('bkd,bcd->bkc', F.normalize(slots, dim=2), F.normalize(slot_key, dim=2)) # (b, n_slots, n_slots + n_patches)

        slot2slot_attn_mask = torch.tril(torch.ones(slots.size(1), slots.size(1), device=dots.device), diagonal=-1) # slot to slot attention, causal

        slot2key_attn_mask_1 = torch.ones(self.cond_num, key.size(1), device=dots.device) # slot to key attention, first slot can attend all image tokens
        slot2key_attn_mask_2 = torch.zeros(slots.size(1) - self.cond_num, key.size(1), device=dots.device) # slot to key attention, other slots cannot attend any image tokens.
        slot2key_attn_mask = torch.cat([slot2key_attn_mask_1, slot2key_attn_mask_2], dim=0)

        attn_mask = torch.cat([slot2slot_attn_mask, slot2key_attn_mask], dim=1) # (n_slots, n_slots + n_patches)
        attn_mask = attn_mask.unsqueeze(0).repeat(dots.size(0), 1, 1)

        dots = dots.masked_fill(attn_mask == 0, float('-inf')) # (b, n_slots, n_slots + n_patches)
        attn = (dots / temp).softmax(dim=-1)# + self.eps 
        
        slot_val = torch.cat([slots, val], dim=1) # (b, n_slots + n_patches, c)
        slots = torch.einsum('bkc,bdk->bdc', slot_val, attn / (attn.sum(dim=2, keepdim=True) + self.eps))
        
        return slots, dots

class SlotConditionedUnet(nn.Module):
    def __init__(self, slot_dim, cond_dim, image_size):
        super().__init__()
        # self.cond_proj = nn.Sequential(nn.Linear(slot_dim, cond_dim), nn.GELU(), nn.Linear(cond_dim, cond_dim))
        self.cond_proj = nn.Identity()
        self.unet2d = UNet2DModel(
            sample_size=image_size,
            in_channels=3 + cond_dim,
            out_channels=3,
            layers_per_block=2,
            block_out_channels=[64, 128, 128],
            down_block_types=(
                "DownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "UpBlock2D",
            )
        )
        
    def forward(self, x, t, slot):
        bs, ch, w, h = x.shape
        slot_cond = self.cond_proj(slot) # TODO: not sure about the shapes yet.
        net_input = torch.cat((x, slot_cond), dim=1)
        return self.unet2d(net_input, t).sample


class SlotCon(nn.Module):
    def __init__(self, dim_slot, num_slot, head_type, 
                 use_causal_grouping=False, 
                 use_componenting=False, comp_cond_num=4,
                 pred_pca_comp=False,
                 use_diffusion=False, diffusion_timesteps=1000,
                 diffusion_cond_dim=3,  slot_cond_decoder_depth=4,
                 ckpt_path=None, norm_pix_loss=False, recon_loss_weight=1., 
                 image_size=224, decoder_depth=8, drop_path_rate=0.1, encoder='vit_base_patch16'):
        super().__init__()

        self.dim_out = dim_slot
        self.num_prototypes = num_slot

        import paintmind.stage1.vision_transformers as vision_transformer
        encoder_fn = vision_transformer.__dict__[encoder]

        self.encoder = encoder_fn(head_type=head_type, drop_path_rate=drop_path_rate)
        self.num_channels = self.encoder.num_features

        self.proj_slot = nn.Sequential(nn.Linear(self.num_channels, self.dim_out), nn.GELU(), nn.Linear(self.dim_out, self.dim_out))

        if use_causal_grouping:
            grouping_mod = CausalSemanticGrouping if not use_componenting else CausalSemanticComponents
        else:
            grouping_mod = SemanticGrouping#(self.num_prototypes, self.dim_out)
            
        self.grouping = grouping_mod(self.num_prototypes, self.dim_out, cond_num=comp_cond_num)
        self.downsample = nn.Identity()
        
        self.proj_back = nn.Sequential(nn.Linear(self.dim_out, self.num_channels), nn.GELU(), nn.Linear(self.num_channels, self.num_channels))

        self.norm_pix_loss = norm_pix_loss
        self.recon_loss_weight = recon_loss_weight
        
        self.img_size = image_size
        
        self.use_diffusion = use_diffusion
        if use_diffusion:
            self.num_timesteps = diffusion_timesteps
            self.diffuse_sched = DDPMScheduler(diffusion_timesteps)
            # TODO: this mlp need to have timestep embeddings
            # NOTE: mlp is used to diffuse for latent components
            # self.diffuse_mlp = nn.Sequential(nn.Linear(self.dim_out, self.num_channels), nn.GELU(), nn.Linear(self.num_channels, self.num_channels))
            
            self.slot_cond_decoder = SlotDecoder(img_size=image_size, decoder_depth=slot_cond_decoder_depth)
            self.rgb_diffuse = SlotConditionedUnet(dim_slot, diffusion_cond_dim, image_size)
        else:
            # NOTE: directly decode to RGB, use MSE and l1 losses
            self.slot_decoder = SlotDecoder(img_size=image_size, decoder_depth=decoder_depth)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.encoder.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x
    
    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.encoder.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def decode_loss(self, imgs, pred):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        # loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        loss = loss.mean()
        return loss
    
    def decode_components(self, slots):
        pass
    
    def diffuse_train(self, slot_cond, gt):
        # slots are the condition
        # gt is the ground truth image components
        batch_size = slot_cond.size(0)
        device = slot_cond.device

        noise = torch.randn_like(gt)
        timesteps = torch.randint(0, self.num_timesteps, (batch_size,)).long().to(device)
        
        noisy_gt = self.diffuse_sched.add_noise(gt, noise, timesteps)
        pred = self.rgb_diffuse(noisy_gt, timesteps, slot_cond)
        loss = F.mse_loss(pred, noise)
        return loss
    
    @torch.no_grad()
    def diffuse_rgb(self, slot_cond, gt=None):
        from tqdm import trange
        batch_size = slot_cond.size(0)
        x = torch.randn(batch_size, 3, self.img_size, self.img_size).to(slot_cond.device)
        for i, t in enumerate(trange(self.num_timesteps)):
            residual = self.rgb_diffuse(x, t, slot_cond)
            x = self.diffuse_sched.step(residual, t, x).prev_sample
        return x

    def forward(self, input, targets, sample=False):
        # TODO: let's implement loading CLIP latent and loading LDM VAE and learning nested dropout next.
        # __import__("ipdb").set_trace()
        losses = {}

        # __import__("ipdb").set_trace()
        x1 = self.downsample(self.encoder(input))
        
        # (q1, score_q1), (q2, score_q2) = self.grouping_q(x1_proj, x1, self.teacher_temp), self.grouping_q(x2_proj, x2, self.teacher_temp)
        x1 = self.proj_slot(rearrange(x1, 'b c h w -> b (h w) c'))
        x1 = rearrange(x1, 'b (h w) c -> b c h w', h=int(x1.size(1)**.5))
        q, score = self.grouping(x1) # q.shape = (N, num_prototypes, slot_dim)
        q = self.proj_back(q) # q.shape = (N, num_prototypes, num_channels)
        
        if sample:
            # NOTE: sample from diffusion
            assert self.use_diffusion
            slot_cond = self.slot_cond_decoder(q, score)
            slot_cond = self.unpatchify(slot_cond)
            recon = self.diffuse_rgb(slot_cond)
            return recon
        
        if not self.use_diffusion:
            # NOTE: lets implement matching to PCA comp next.
            
            recon = self.slot_decoder(q, score)
            recon_loss = self.decode_loss(input, recon)
            losses['recon_loss'] = recon_loss * self.recon_loss_weight
            
            recon = self.unpatchify(recon)
            
            return recon, losses
        else:
            slot_cond = self.slot_cond_decoder(q, score)
            slot_cond = self.unpatchify(slot_cond)
            loss = self.diffuse_train(slot_cond, input)
            losses['diffuse_loss'] = loss
            
            return losses


    # def re_init(self, args):
    #     self.k = int(args.num_instances * 1. / args.world_size / args.batch_size * (args.start_epoch - 1))

    # @torch.no_grad()
    # def _momentum_update_key_encoder(self):
    #     """
    #     Momentum update of the key encoder
    #     """
    #     momentum = 1. - (1. - self.teacher_momentum) * (math.cos(math.pi * self.k / self.K) + 1.) * 0.5
    #     self.k += 1
    #     for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
    #         param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
    #     for param_q, param_k in zip(self.projector_q.parameters(), self.projector_k.parameters()):
    #         param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)
    #     for param_q, param_k in zip(self.grouping_q.parameters(), self.grouping_k.parameters()):
    #         param_k.data = param_k.data * momentum + param_q.data * (1. - momentum) 
    #     for param_q, param_k in zip(self.proj_slot_q.parameters(), self.proj_slot_k.parameters()):
    #         param_k.data = param_k.data * momentum + param_q.data * (1. - momentum)

    # def invaug(self, x, coords, flags):
    #     N, C, H, W = x.shape

    #     batch_idxs = torch.arange(N, device=coords.device).view(N, 1)
    #     coords_rescaled = coords.clone()
    #     coords_rescaled[:, 0] = coords_rescaled[:, 0] * W  # x1
    #     coords_rescaled[:, 2] = coords_rescaled[:, 2] * W  # x2
    #     coords_rescaled[:, 1] = coords_rescaled[:, 1] * H  # y1
    #     coords_rescaled[:, 3] = coords_rescaled[:, 3] * H  # y2
    #     coords_with_idxs = torch.cat([batch_idxs, coords_rescaled], dim=1)
        
    #     x_aligned = torchvision.ops.roi_align(x, coords_with_idxs, (H, W), aligned=True)
    #     x_flipped = torch.stack([feat.flip(-1) if flag else feat for feat, flag in zip(x_aligned, flags)])
    #     return x_flipped

    # def self_distill(self, q, k):
    #     q = F.log_softmax(q / self.student_temp, dim=-1)
    #     k = F.softmax((k - self.center) / self.teacher_temp, dim=-1)
    #     return torch.sum(-k * q, dim=-1).mean()

    # def ctr_loss_filtered(self, q, k, score_q, score_k, tau=0.2):
    #     q = self.proj_slot_q(q.flatten(0, 1))
    #     k = F.normalize(self.proj_slot_k(k.flatten(0, 1)), dim=1)

    #     mask_q = (torch.zeros_like(score_q).scatter_(1, score_q.argmax(1, keepdim=True), 1).sum(-1).sum(-1) > 0).long().detach()
    #     mask_k = (torch.zeros_like(score_k).scatter_(1, score_k.argmax(1, keepdim=True), 1).sum(-1).sum(-1) > 0).long().detach()
    #     mask_intersection = (mask_q * mask_k).view(-1)
    #     idxs_q = mask_intersection.nonzero().squeeze(-1)

    #     mask_k = concat_all_gather(mask_k.view(-1))
    #     idxs_k = mask_k.nonzero().squeeze(-1)

    #     N = k.shape[0]
    #     logits = torch.einsum('nc,mc->nm', [F.normalize(self.predictor_slot(q[idxs_q]), dim=1), concat_all_gather(k)[idxs_k]]) / tau
    #     labels = mask_k.cumsum(0)[idxs_q + N * torch.distributed.get_rank()] - 1
    #     return F.cross_entropy(logits, labels) * (2 * tau)

    # def patchify(self, imgs):
    #     """
    #     imgs: (N, 3, H, W)
    #     x: (N, L, patch_size**2 *3)
    #     """
    #     p = self.encoder_q.patch_embed.patch_size[0]
    #     assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    #     h = w = imgs.shape[2] // p
    #     x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    #     x = torch.einsum('nchpwq->nhwpqc', x)
    #     x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
    #     return x

    # def decode_loss(self, imgs, pred):
    #     target = self.patchify(imgs)
    #     if self.norm_pix_loss:
    #         mean = target.mean(dim=-1, keepdim=True)
    #         var = target.var(dim=-1, keepdim=True)
    #         target = (target - mean) / (var + 1.e-6)**.5

    #     loss = (pred - target) ** 2
    #     # loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
    #     loss = loss.mean()
    #     return loss

    # def forward(self, input, epoch):
    #     crops, coords, flags = input
    #     self.teacher_temp = self.teacher_temp_schedule[epoch]
    #     losses = {}

    #     x1, x2 = self.downsample(self.encoder_q(crops[0])), self.downsample(self.encoder_q(crops[1]))

    #     if not self.detach_group_loss:
    #         x1_proj, x2_proj = self.projector_q(x1), self.projector_q(x2)
    #     else:
    #         x1_proj, x2_proj = self.projector_q(x1.detach()), self.projector_q(x2.detach())
        
    #     with torch.no_grad():  # no gradient to keys
    #         self._momentum_update_key_encoder()  # update the key encoder
    #         y1, y2 = self.downsample(self.encoder_k(crops[0])), self.downsample(self.encoder_k(crops[1]))
    #         y1_proj, y2_proj = self.projector_k(y1), self.projector_k(y2)
        
    #     (q1, score_q1), (q2, score_q2) = self.grouping_q(x1_proj, x1, self.teacher_temp), self.grouping_q(x2_proj, x2, self.teacher_temp)
    #     q1_aligned, q2_aligned = self.invaug(score_q1, coords[0], flags[0]), self.invaug(score_q2, coords[1], flags[1])
    #     with torch.no_grad():
    #         (k1, score_k1), (k2, score_k2) = self.grouping_k(y1_proj, y1, self.teacher_temp), self.grouping_k(y2_proj, y2, self.teacher_temp)
    #         k1_aligned, k2_aligned = self.invaug(score_k1, coords[0], flags[0]), self.invaug(score_k2, coords[1], flags[1])
        
    #     group_loss = self.self_distill(q1_aligned.permute(0, 2, 3, 1).flatten(0, 2), k2_aligned.permute(0, 2, 3, 1).flatten(0, 2)) \
    #                + self.self_distill(q2_aligned.permute(0, 2, 3, 1).flatten(0, 2), k1_aligned.permute(0, 2, 3, 1).flatten(0, 2))
        
    #     losses['group_loss'] = group_loss * self.group_loss_weight * .5

    #     self.update_center(torch.cat([score_k1, score_k2]).permute(0, 2, 3, 1).flatten(0, 2))

    #     ctr_loss = self.ctr_loss_filtered(q1, k2, score_q1, score_k2) \
    #              + self.ctr_loss_filtered(q2, k1, score_q2, score_k1)

    #     losses['ctr_loss'] = ctr_loss * (1. - self.group_loss_weight) * .5
        
    #     if self.use_slot_decoder:
    #         recon1 = self.slot_decoder(q1, score_q1)
    #         recon2 = self.slot_decoder(q2, score_q1)
    #         recon_loss = (self.decode_loss(crops[0], recon1) + self.decode_loss(crops[1], recon2))
    #         losses['recon_loss'] = recon_loss * self.recon_loss_weight * .5
        
    #     return losses
    
    # @torch.no_grad()
    # def update_center(self, teacher_output):
    #     """
    #     Update center used for teacher output.
    #     """
    #     batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
    #     dist.all_reduce(batch_center)
    #     batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

    #     # ema update
    #     self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

# class CLIPSlot

