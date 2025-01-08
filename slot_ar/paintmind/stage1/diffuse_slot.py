import torch
import torch.nn as nn
from timm.models.vision_transformer import Attention, Mlp, DropPath

import os
import json
import numpy as np


import math
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision
import torch
from diffusers import DDPMScheduler, AutoencoderKL
from einops import rearrange, repeat
# from vector_quantize_pytorch import LFQ

from paintmind.stage1.slotcoder import CausalSemanticGrouping
from paintmind.stage1.diffusion import create_diffusion
from paintmind.stage1.diffusion_transfomers import DiT
from paintmind.stage1.quantize import VectorQuantizer


class DiT_with_autoenc_cond(DiT):
    def __init__(
        self, *args, num_autoenc=32, autoenc_dim=4, cond_method="adaln", **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.autoenc_dim = autoenc_dim
        self.cond_drop_prob = self.y_embedder.dropout_prob  # 0.1 without cond guidance
        self.null_cond = nn.Parameter(torch.randn(num_autoenc, autoenc_dim))
        # NOTE: adaln is adaptive layer normalization, token fed the cond to the attention layer
        assert cond_method in [
            "adaln",
            "token",
            "token+adaln",
        ], f"Invalid cond_method: {cond_method}"
        self.cond_method = cond_method
        if "token" in cond_method:
            self.autoenc_cond_embedder = nn.Linear(autoenc_dim, self.hidden_size)
        else:
            self.autoenc_cond_embedder = nn.Linear(
                num_autoenc * autoenc_dim, self.hidden_size
            )

        if cond_method == "token+adaln":
            self.autoenc_proj_ln = nn.Linear(self.hidden_size, self.hidden_size)

    def embed_cond(self, autoenc_cond):
        # autoenc_cond: (N, K, D)
        # drop_ids: (N)
        # self.null_cond: (K, D)
        # NOTE: this dropout will replace some condition from the autoencoder to null condition
        # this is to enable classifier-free guidance.
        batch_size = autoenc_cond.shape[0]
        if self.training:
            drop_ids = (
                torch.rand(batch_size, 1, 1, device=autoenc_cond.device)
                < self.cond_drop_prob
            )
            autoenc_cond_drop = torch.where(drop_ids, self.null_cond, autoenc_cond)
        else:
            autoenc_cond_drop = autoenc_cond
        if "token" in self.cond_method:
            return self.autoenc_cond_embedder(autoenc_cond_drop)
        return self.autoenc_cond_embedder(autoenc_cond_drop.reshape(batch_size, -1))

    # def forward(self, x, t, y, autoenc_cond):
    def forward(self, x, t, autoenc_cond):
        """
        Forward pass of DiT.
        x: (N, C, H, W) tensor of spatial inputs (images or latent representations of images)
        t: (N,) tensor of diffusion timesteps
        y: (N,) tensor of class labels
        autoenc_cond: (N, K, D) tensor of autoencoder conditions (slots)
        """
        # __import__("ipdb").set_trace()
        x = (
            self.x_embedder(x) + self.pos_embed
        )  # (N, T, D), where T = H * W / patch_size ** 2
        t = self.t_embedder(t)  # (N, D)
        # y = self.y_embedder(y, self.training)    # (N, D)
        # c = t + y                                # (N, D)
        # TODO: process the autoenc_cond
        # autoenc = self.autoenc_cond_embedd(autoenc_cond, self.training)
        autoenc = self.embed_cond(autoenc_cond)
        if self.cond_method == "adaln":
            c = t + autoenc  # add the encoder condition to adaln
        elif self.cond_method == "token":
            c = t
            num_tokens = x.shape[1]
            # append the autoencoder condition to the token sequence
            # TODO: maybe we also need to implement classifier-free guidance here
            # i.e. randomly drop the autoencoder condition with a probability
            x = torch.cat((x, autoenc), dim=1)
        elif self.cond_method == "token+adaln":
            c = t + self.autoenc_proj_ln(autoenc.mean(dim=1))
            num_tokens = x.shape[1]
            x = torch.cat((x, autoenc), dim=1)
        else:
            raise ValueError(f"Invalid cond_method: {self.cond_method}")
        for block in self.blocks:
            x = block(x, c)  # (N, T, D)
        if "token" in self.cond_method:
            x = x[:, :num_tokens, :]
        x = self.final_layer(x, c)  # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)  # (N, out_channels, H, W)
        return x


#################################################################################
#                                   DiT Configs                                  #
#################################################################################


def DiT_with_autoenc_cond_XL_2(**kwargs):
    return DiT_with_autoenc_cond(
        depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs
    )


def DiT_with_autoenc_cond_XL_4(**kwargs):
    return DiT_with_autoenc_cond(
        depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs
    )


def DiT_with_autoenc_cond_XL_8(**kwargs):
    return DiT_with_autoenc_cond(
        depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs
    )


def DiT_with_autoenc_cond_L_2(**kwargs):
    return DiT_with_autoenc_cond(
        depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs
    )


def DiT_with_autoenc_cond_L_4(**kwargs):
    return DiT_with_autoenc_cond(
        depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs
    )


def DiT_with_autoenc_cond_L_8(**kwargs):
    return DiT_with_autoenc_cond(
        depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs
    )


def DiT_with_autoenc_cond_B_2(**kwargs):
    return DiT_with_autoenc_cond(
        depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs
    )


def DiT_with_autoenc_cond_B_4(**kwargs):
    return DiT_with_autoenc_cond(
        depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs
    )


def DiT_with_autoenc_cond_B_8(**kwargs):
    return DiT_with_autoenc_cond(
        depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs
    )


def DiT_with_autoenc_cond_S_2(**kwargs):
    return DiT_with_autoenc_cond(
        depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs
    )


def DiT_with_autoenc_cond_S_4(**kwargs):
    return DiT_with_autoenc_cond(
        depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs
    )


def DiT_with_autoenc_cond_S_8(**kwargs):
    return DiT_with_autoenc_cond(
        depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs
    )


DiT_with_autoenc_cond_models = {
    "DiT-XL-2": DiT_with_autoenc_cond_XL_2,
    "DiT-XL-4": DiT_with_autoenc_cond_XL_4,
    "DiT-XL-8": DiT_with_autoenc_cond_XL_8,
    "DiT-L-2": DiT_with_autoenc_cond_L_2,
    "DiT-L-4": DiT_with_autoenc_cond_L_4,
    "DiT-L-8": DiT_with_autoenc_cond_L_8,
    "DiT-B-2": DiT_with_autoenc_cond_B_2,
    "DiT-B-4": DiT_with_autoenc_cond_B_4,
    "DiT-B-8": DiT_with_autoenc_cond_B_8,
    "DiT-S-2": DiT_with_autoenc_cond_S_2,
    "DiT-S-4": DiT_with_autoenc_cond_S_4,
    "DiT-S-8": DiT_with_autoenc_cond_S_8,
}

from torch.distributions import Geometric, Uniform


class NestedAttention(nn.Module):
    def __init__(
        self,
        slot_dim,
        sampler_dim,
        num_slots,
        enable=True,
        rho=0.03,
        nest_dist="geometric",
        num_heads=1,
        output_dim=None,
    ):
        super().__init__()
        self.attn = nn.MultiheadAttention(sampler_dim, num_heads, batch_first=True)
        self.slot2attn = nn.Linear(slot_dim, sampler_dim)
        self.num_slots = num_slots
        self.enable = enable
        self.rho = rho
        self.geometric = Geometric(rho)
        self.uniform = Uniform(1.0, self.num_slots + 1.0)
        self.nest_dist = nest_dist
        self.register_buffer("arange", torch.arange(num_slots))
        self.output_dim = output_dim
        if output_dim is not None:
            self.slot2output = nn.Linear(sampler_dim, output_dim)

    def geometric_sample(self, num):
        return self.geometric.sample([num]) + 1

    def uniform_sample(self, num):
        return self.uniform.sample([num]).long()

    def sample(self, num):
        if self.nest_dist == "geometric":
            return self.geometric_sample(num)
        elif self.nest_dist == "uniform":
            return self.uniform_sample(num)
        else:
            raise ValueError(f"Invalid nest_dist: {self.nest_dist}")

    def set_enable(self, enable):
        self.enable = enable

    def forward(self, slots, sampler, inference_with_n_slots=-1):
        slots = self.slot2attn(slots)
        batch_size = sampler.shape[0]

        if self.training and self.enable:
            #     b = self.sample(batch_size)
            #     b = torch.clamp(b, max=self.num_slots).to(slots.device)
            #     # Create an arange tensor for comparison
            #     arange = self.arange #torch.arange(self.num_slots).expand(batch_size, -1).to(slots.device)
            #     # Create the mask using broadcasting
            #     mask = (arange > b.unsqueeze(1)).unsqueeze(1) # now the mask is (batch_size, 1, num_slots)
            #     attn_mask = mask.expand(-1, sampler.shape[1], -1) # (batch_size, sampler.shape[1], num_slots)
            # elif not self.training and inference_with_n_slots != -1:
            #     b = torch.tensor([inference_with_n_slots] * batch_size).to(slots.device)
            #     # Create an arange tensor for comparison
            #     arange = self.arange # torch.arange(self.num_slots).expand(batch_size, -1).to(slots.device)
            #     # Create the mask using broadcasting
            #     mask = (arange > b.unsqueeze(1)).unsqueeze(1) # now the mask is (batch_size, 1, num_slots)
            #     attn_mask = mask.expand(-1, sampler.shape[1], -1) # (batch_size, sampler.shape[1], num_slots)

            # else:
            #     attn_mask = None

            # attn_output, attn_score = self.attn(sampler, slots, slots, attn_mask=attn_mask)
            # return attn_output, attn_score
            # NOTE: below is cursor generated code, using key_padding_mask,
            #       to mask out the slots that are not in the current nested attention
            #       it should be correct, but did not test yet
            b = self.sample(batch_size).to(slots.device)
        elif not self.training and inference_with_n_slots != -1:
            b = torch.full((batch_size,), inference_with_n_slots, device=slots.device)
        else:
            attn_output, attn_score = self.attn(sampler, slots, slots)
            if self.output_dim is not None:
                attn_output = self.slot2output(attn_output)
            return attn_output, attn_score

        b = torch.clamp(b, max=self.num_slots)
        key_padding_mask = self.arange[None, :] >= b[:, None]  # (batch_size, num_slots)

        attn_output, attn_score = self.attn(
            sampler, slots, slots, key_padding_mask=key_padding_mask
        )
        if self.output_dim is not None:
            attn_output = self.slot2output(attn_output)
        return attn_output, attn_score


class DiffuseSlot(nn.Module):
    def __init__(
        self,
        num_slots=32,
        slot_dim=4,
        num_samplers=32,
        sampler_dim=4,
        enable_nest=False,
        enable_nest_after=-1,
        nest_dist="geometric",
        nest_rho=0.03,
        cond_method="adaln",
        use_encoder_rgb=False,
        encoder="vit_base_patch16",
        head_type="early_return",
        drop_path_rate=0.1,
        enc_img_size=256,
        dit_model="DiT-B-4",
        vae="stabilityai/sd-vae-ft-ema",
        use_classifier=False,
        num_classes=10,  # 10 for imagewoof
        use_vq=False,
        use_lfq=False,
        codebook_size=65536,
        code_dim=256,
        code_beta=0.25,
        vq_norm=True,
        pretrained_dit=None,
        pretrained_encoder=None,
        freeze_dit=False,
        **kwargs,
    ):
        super().__init__()

        # DiT part
        self.diffusion = create_diffusion(timestep_respacing="")
        self.ddim_diffusion = create_diffusion(timestep_respacing="100")
        self.dit = DiT_with_autoenc_cond_models[dit_model](
            input_size=enc_img_size // 8,
            num_autoenc=num_samplers,
            autoenc_dim=sampler_dim,
            cond_method=cond_method,
        )
        self.pretrained_dit = pretrained_dit
        if pretrained_dit is not None:
            # now we load some pretrained model
            dit_ckpt = torch.load(pretrained_dit, map_location="cpu")
            msg = self.dit.load_state_dict(dit_ckpt, strict=False)
            print("Load DiT from ckpt")
            print(msg)
        self.freeze_dit = freeze_dit
        if freeze_dit:
            assert pretrained_dit is not None, "pretrained_dit must be provided"
            for param in self.dit.parameters():
                param.requires_grad = False

        self.vae = AutoencoderKL.from_pretrained(vae).cuda().requires_grad_(False)
        self.scaling_factor = self.vae.scaling_factor

        # image encoder part
        # TODO: make a vit base model with image size being 256
        import paintmind.stage1.vision_transformers as vision_transformer

        self.use_encoder_rgb = use_encoder_rgb
        # import ipdb; ipdb.set_trace()
        self.enc_img_size = enc_img_size
        if use_encoder_rgb:
            encoder_fn = vision_transformer.__dict__[encoder]

            self.encoder = encoder_fn(
                img_size=enc_img_size,
                head_type=head_type,
                drop_path_rate=drop_path_rate,
            )
            self.num_channels = self.encoder.num_features
            self.pretrained_encoder = pretrained_encoder
            if pretrained_encoder is not None:
                # __import__("ipdb").set_trace()
                encoder_ckpt = torch.load(pretrained_encoder, map_location="cpu")
                # drop pos_embed from ckpt
                encoder_ckpt = {
                    k.replace("blocks.", "blocks.0."): v
                    for k, v in encoder_ckpt.items()
                    if not k.startswith("pos_embed")
                }
                msg = self.encoder.load_state_dict(encoder_ckpt, strict=False)
                print("Load encoder from ckpt")
                print(msg)
                self.encoder2slot = nn.Linear(self.num_channels, slot_dim)
            else:
                self.encoder2slot = nn.Conv2d(self.num_channels, slot_dim, 1)

        self.grouping = CausalSemanticGrouping(num_slots, slot_dim)
        self.sampler_latents = nn.Parameter(torch.randn(num_samplers, sampler_dim))
        self.nested_sampler = NestedAttention(
            slot_dim,
            sampler_dim,
            num_slots,
            enable=enable_nest,
            rho=nest_rho,
            nest_dist=nest_dist,
        )
        self.enable_nest_after = enable_nest_after

        self.use_vq = use_vq
        if use_vq:
            print("Using VQ")
            self.use_lfq = use_lfq
            if self.use_lfq:
                self.slot2vq = nn.Linear(slot_dim, code_dim)
                self.vq = LFQ(
                    # codebook_size,
                    # code_dim,
                    # code_beta,
                    codebook_size=codebook_size,  # codebook size, must be a power of 2
                    dim=code_dim,  # this is the input feature dimension, defaults to log2(codebook_size) if not defined
                    entropy_loss_weight=code_beta,  # how much weight to place on entropy loss
                    diversity_gamma=1.0,  # within entropy loss, how much weight to give to diversity of codes, taken from https://arxiv.org/abs/1911.05894
                    experimental_softplus_entropy_loss=True
                )
                self.vq2sample = nn.Linear(code_dim, sampler_dim)
            else:
                assert False, "this does not work!!!"
                self.slot2vq = nn.Linear(slot_dim, code_dim)
                self.vq = VectorQuantizer(
                    codebook_size, code_dim, code_beta, use_norm=vq_norm
                )
                self.vq2sample = nn.Linear(code_dim, sampler_dim)

        if use_classifier:
            # TODO: add classifier
            self.use_classifier = True
            self.classifier = nn.Linear(slot_dim, num_classes)

    @torch.no_grad()
    def vae_encode(self, x):
        return self.vae.encode(x).latent_dist.sample().mul_(self.scaling_factor)

    @torch.no_grad()
    def vae_decode(self, z):
        return self.vae.decode(z / self.scaling_factor).sample

    # currently only use it for inference
    def encode_slots(self, x):
        if self.use_encoder_rgb:
            x_enc = self.encoder(x)
            x_enc = self.encoder2slot(x_enc)
        else:
            x_enc = self.vae_encode(x)

        slots, dots = self.grouping(x_enc)  # (n, num_slots, slot_dim)
        return slots, dots

    def forward(self, x, targets, latents=None, sample=False, epoch=None, inference_with_n_slots=-1):
        losses = {}
        batch_size = x.shape[0]
        device = x.device
        # __import__('ipdb').set_trace()

        # TODO: generate slots from input
        # __import__("ipdb").set_trace()
        if latents is None:
            x_vae = self.vae_encode(x)  # (N, C, H, W)
        else:
            x_vae = latents

        if self.use_encoder_rgb:
            # __import__("ipdb").set_trace()
            if self.pretrained_encoder is not None:
                x = F.interpolate(x, size=224)
            x_enc = self.encoder(x)
            x_enc = self.encoder2slot(x_enc)
        else:
            x_enc = x_vae
        if (
            epoch is not None
            and epoch > self.enable_nest_after
            and self.enable_nest_after != -1
            and not self.nested_sampler.enable
        ):
            self.nested_sampler.set_enable(True)

        slots, dots = self.grouping(x_enc)  # (n, num_slots, slot_dim)
        # __import__("ipdb").set_trace()
        if self.use_vq:
            slots_vq = self.slot2vq(slots)
            # slots_vq, vq_loss, _ = self.vq(slots_vq.view(-1, self.vq.e_dim))
            slots_vq, indices, vq_loss = self.vq(slots_vq)
            # slots_vq = slots_vq.view(slots.shape)
            slots = self.vq2sample(slots_vq)
            losses["vq_loss"] = vq_loss
        # make the slots as gaussian, reparameterize the slots
        # and also add a kl loss to the slots to make them standard gaussian
        # for example,
        # def kl_loss(mean, logvar):
            # # Compute KL divergence
            # kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            # # Normalize by batch size
            # kl /= mean.size(0)
            # return kl
        # mean, logvar = self.slots2gaussian(slots)
        # slots = mean + torch.exp(logvar * .5) * torch.randn_like(logvar)
        # losses['kl_slots'] = kl_loss(mean, logvar)
        sampler_latents = repeat(self.sampler_latents, "s d -> b s d", b=batch_size)
        sampled_slots, attn_score = self.nested_sampler(
            slots, sampler_latents, inference_with_n_slots=inference_with_n_slots
        )

        if sample:
            return self.sample(x_vae, sampled_slots, device)

        # TODO: here we want larger t for when num_slots is smaller,
        #       and smaller t for when num_slots is larger
        #       to make that the earlier slots to learn semantics
        #       and the later slots to learn details
        #       the intuition is that, for larger steps, the learning is more on semantics,
        #       and for smaller steps, the learning is more on details
        #       consider this from the score function perspective
        #       when we start with pure noise, the score function will point to the region of the category distribution
        #       and when the noise is smaller, the score function will point to the region of more details
        t = torch.randint(
            0, self.diffusion.num_timesteps, (x_vae.shape[0],), device=device
        )
        model_kwargs = dict(autoenc_cond=sampled_slots)
        loss_dict = self.diffusion.training_losses(self.dit, x_vae, t, model_kwargs)
        diff_loss = loss_dict["loss"].mean()

        losses["diff_loss"] = diff_loss
        if diff_loss.item() > 1.25:
            raise ValueError(f"diff_loss is too high: {diff_loss.item()}")

        return losses

    @torch.no_grad()
    def sample_from_slots(self, slots, device, inference_with_n_slots=-1, ddim=False):
        batch_size = slots.shape[0]
        sampler_latents = repeat(self.sampler_latents, "s d -> b s d", b=batch_size)
        sampled_slots, attn_score = self.nested_sampler(
            slots, sampler_latents, inference_with_n_slots=inference_with_n_slots
        )
        return self.sample(
            torch.randn(
                (batch_size, 4, self.enc_img_size // 8, self.enc_img_size // 8)
            ),
            sampled_slots,
            device,
            ddim=ddim,
        )

    @torch.no_grad()
    def sample(self, x, sampled_slots, device, ddim=False):
        # Sample per epoch
        # NOTE: get cond

        z = torch.randn_like(x, device=device)

        # Setup classifier-free guidance:
        # null_cond = repeat(self.dit.null_cond, 'S D -> B S D', B=z.shape[0])
        # z = torch.cat([z, z], 0)
        # cond = torch.cat([sampled_slots, null_cond], 0)
        # model_kwargs = dict(autoenc_cond=cond)
        model_kwargs = dict(autoenc_cond=sampled_slots)

        # Sample images:
        if ddim:
            samples = self.ddim_diffusion.ddim_sample_loop(
                self.dit.forward,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True,
                device=device,
            )
        else:
            samples = self.diffusion.p_sample_loop(
                self.dit.forward,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True,
                device=device,
            )

        # samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = self.vae.decode(samples / self.scaling_factor).sample
        return samples

    @torch.no_grad()
    def encode_to_sampler(self, x, n_slots=32):
        batch_size = x.shape[0]
        x_enc = self.encoder(x)
        x_enc = self.encoder2slot(x_enc)
        slots, dots = self.grouping(x_enc)  # (n, num_slots, slot_dim)
        sampled_slots = slots[:, :n_slots, :]
        # sampler_latents = repeat(self.sampler_latents, 's d -> b s d', b=batch_size)
        # sampled_slots, attn_score = self.nested_sampler(slots, sampler_latents,
        #                                                 inference_with_n_slots=n_slots)
        return sampled_slots

    # def patchify(self, imgs):
    #     """
    #     imgs: (N, 3, H, W)
    #     x: (N, L, patch_size**2 *3)
    #     """
    #     p = self.encoder.patch_embed.patch_size[0]
    #     assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    #     h = w = imgs.shape[2] // p
    #     x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
    #     x = torch.einsum('nchpwq->nhwpqc', x)
    #     x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
    #     return x

    # def unpatchify(self, x):
    #     """
    #     x: (N, L, patch_size**2 *3)
    #     imgs: (N, 3, H, W)
    #     """
    #     p = self.encoder.patch_embed.patch_size[0]
    #     h = w = int(x.shape[1]**.5)
    #     assert h * w == x.shape[1]

    #     x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
    #     x = torch.einsum('nhwpqc->nchpwq', x)
    #     imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
    #     return imgs


class DiffuseSlotPixArt(nn.Module):
    def __init__(
        self,
        num_slots=32,
        slot_dim=4,
        num_samplers=32,
        sampler_dim=4,
        enable_nest=False,
        enable_nest_after=-1,
        nest_dist="geometric",
        nest_rho=0.03,
        pixart_path="",
        use_encoder_rgb=False,
        encoder="vit_base_patch16",
        head_type="early_return",
        drop_path_rate=0.1,
        enc_img_size=256,
        pretrained_encoder=None,
        reset_xattn=False,
        kl_loss=False,
        kl_loss_weight=0.01,
        **kwargs,
    ):
        # __import__("ipdb").set_trace()
        super().__init__()

        # DiT part
        self.diffusion = create_diffusion(timestep_respacing="")
        self.ddim_diffusion = create_diffusion(timestep_respacing="100")

        # loading PixArt
        import safetensors
        from diffusers import PixArtTransformer2DModel

        self.pixart_dit_config = json.load(
            open(os.path.join(pixart_path, "config.json"), "r"))
        self.pixart_dit = PixArtTransformer2DModel(**self.pixart_dit_config)
        ckpt = {}
        with safetensors.safe_open(os.path.join(pixart_path, "pixart_alpha256XL_2.safetensors"),
                                   framework='pt',
                                   device='cpu') as f:
            for k in f.keys():
                if reset_xattn and 'attn2' in k:
                    continue
                ckpt[k] = f.get_tensor(k)

        print("Loading PixArt ckpt")
        msg = self.pixart_dit.load_state_dict(ckpt, strict=False)
        print(msg)

        self.vae_config = json.load(
            open(os.path.join(pixart_path, "vae_config.json"), "r"))
        self.vae = AutoencoderKL(**self.vae_config)
        vae_ckpt = {}
        with safetensors.safe_open(os.path.join(pixart_path, "pixart_vae.safetensors"),
                                   framework='pt',
                                   device='cpu') as f:
            for k in f.keys():
                vae_ckpt[k] = f.get_tensor(k)
        print("Loading VAE ckpt")
        msg = self.vae.load_state_dict(vae_ckpt, strict=False)
        print(msg)
        self.scaling_factor = self.vae.config.scaling_factor
        self.vae.requires_grad_(False)

        # image encoder part
        # TODO: make a vit base model with image size being 256
        import paintmind.stage1.vision_transformers as vision_transformer

        self.use_encoder_rgb = use_encoder_rgb
        # import ipdb; ipdb.set_trace()
        self.enc_img_size = enc_img_size
        if use_encoder_rgb:
            encoder_fn = vision_transformer.__dict__[encoder]

            self.encoder = encoder_fn(
                img_size=enc_img_size,
                head_type=head_type,
                drop_path_rate=drop_path_rate,
            )
            self.num_channels = self.encoder.num_features
            self.pretrained_encoder = pretrained_encoder
            if pretrained_encoder is not None:
                # __import__("ipdb").set_trace()
                encoder_ckpt = torch.load(pretrained_encoder, map_location="cpu")
                # drop pos_embed from ckpt
                encoder_ckpt = {
                    k.replace("blocks.", "blocks.0."): v
                    for k, v in encoder_ckpt.items()
                    if not k.startswith("pos_embed")
                }
                msg = self.encoder.load_state_dict(encoder_ckpt, strict=False)
                print("Load encoder from ckpt")
                print(msg)
                self.encoder2slot = nn.Linear(self.num_channels, slot_dim)
            else:
                self.encoder2slot = nn.Conv2d(self.num_channels, slot_dim, 1)

        self.grouping = CausalSemanticGrouping(num_slots, slot_dim)
        self.sampler_latents = nn.Parameter(torch.randn(num_samplers, sampler_dim))
        self.nested_sampler = NestedAttention(
            slot_dim,
            sampler_dim,
            num_slots,
            enable=enable_nest,
            rho=nest_rho,
            nest_dist=nest_dist,
            # map the slots to the caption space.
            output_dim=self.pixart_dit.config.caption_channels,
        )
        self.enable_nest_after = enable_nest_after
        self.use_kl = kl_loss
        self.kl_loss_weight = kl_loss_weight
        if self.use_kl:
            self.slots2gaussian = nn.Linear(slot_dim, slot_dim * 2)

        # make the slots as gaussian, reparameterize the slots
        # and also add a kl loss to the slots to make them standard gaussian
        # for example,
        # def kl_loss(mean, logvar):
            # # Compute KL divergence
            # kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
            # # Normalize by batch size
            # kl /= mean.size(0)
            # return kl
        # mean, var = self.slots2gaussian(slots)
        # slots = mean + var * torch.randn_like(var)
        # losses['kl_slots'] = kl_loss(mean, var)
    def kl_loss(self, mean, logvar):
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        kl /= mean.size(0)
        return kl
    
    def reparameterise(self, mean, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mean)
        else:
            return mean

    @torch.no_grad()
    def vae_encode(self, x):
        return self.vae.encode(x).latent_dist.sample().mul_(self.scaling_factor)

    @torch.no_grad()
    def vae_decode(self, z):
        return self.vae.decode(z / self.scaling_factor).sample

    def pixart_forward(self, x, t, xattn_enc):
        # __import__("ipdb").set_trace()
        added_cond_kwargs = {
            'resolution': None,
            'aspect_ratio': None,
        }
        return self.pixart_dit(
            x, xattn_enc, t, added_cond_kwargs
            ).sample #.chunk(2, dim=1)[0]

    # currently only use it for inference
    def encode_slots(self, x):
        if self.use_encoder_rgb:
            x_enc = self.encoder(x)
            x_enc = self.encoder2slot(x_enc)
        else:
            x_enc = self.vae_encode(x)

        slots, dots = self.grouping(x_enc)  # (n, num_slots, slot_dim)
        return slots, dots

    @torch.no_grad()
    def sample_from_slots(self, slots, device, inference_with_n_slots=-1, ddim=False):
        batch_size = slots.shape[0]
        if self.use_kl:
            mean, logvar = self.slots2gaussian(slots)
            slots = self.reparameterise(mean, logvar)
        sampler_latents = repeat(self.sampler_latents, "s d -> b s d", b=batch_size)
        sampled_slots, attn_score = self.nested_sampler(
            slots, sampler_latents, inference_with_n_slots=inference_with_n_slots
        )
        return self.sample(
            torch.randn(
                (batch_size, 4, self.enc_img_size // 8, self.enc_img_size // 8)
            ),
            sampled_slots,
            device,
            ddim=ddim,
        )

    @torch.no_grad()
    def sample(self, x, sampled_slots, device, ddim=False):
        # Sample per epoch
        # NOTE: get cond

        z = torch.randn_like(x, device=device)

        # Setup classifier-free guidance:
        # null_cond = repeat(self.dit.null_cond, 'S D -> B S D', B=z.shape[0])
        # z = torch.cat([z, z], 0)
        # cond = torch.cat([sampled_slots, null_cond], 0)
        # model_kwargs = dict(autoenc_cond=cond)
        model_kwargs = dict(xattn_enc=sampled_slots)

        # Sample images:
        if ddim:
            samples = self.ddim_diffusion.ddim_sample_loop(
                # self.pixart_dit.forward,
                self.pixart_forward,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True,
                device=device,
            )
        else:
            samples = self.diffusion.p_sample_loop(
                # self.pixart_dit.forward,
                self.pixart_forward,
                z.shape,
                z,
                clip_denoised=False,
                model_kwargs=model_kwargs,
                progress=True,
                device=device,
            )

        # samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        samples = self.vae.decode(samples / self.scaling_factor).sample
        return samples

    def forward(self, x, targets, latents=None, sample=False, epoch=None, inference_with_n_slots=-1):
        losses = {}
        batch_size = x.shape[0]
        device = x.device
        # __import__('ipdb').set_trace()
        if latents is None:
            x_vae = self.vae_encode(x)  # (N, C, H, W)
        else:
            x_vae = latents

        if self.use_encoder_rgb:
            # __import__("ipdb").set_trace()
            if self.pretrained_encoder is not None:
                x = F.interpolate(x, size=224)
            x_enc = self.encoder(x)
            x_enc = self.encoder2slot(x_enc)
        else:
            x_enc = x_vae
        if (
            epoch is not None
            and epoch > self.enable_nest_after
            and self.enable_nest_after != -1
            and not self.nested_sampler.enable
        ):
            self.nested_sampler.set_enable(True)

        slots, dots = self.grouping(x_enc)  # (n, num_slots, slot_dim)
        # __import__("ipdb").set_trace()
        if self.use_kl:
            # __import__("ipdb").set_trace()
            mean, logvar = self.slots2gaussian(slots).chunk(2, dim=-1)
            slots = self.reparameterise(mean, logvar)

        sampler_latents = repeat(self.sampler_latents, "s d -> b s d", b=batch_size)
        sampled_slots, attn_score = self.nested_sampler(
            slots, sampler_latents, inference_with_n_slots=inference_with_n_slots
        )

        if sample:
            return self.sample(x_vae, sampled_slots, device)

        t = torch.randint(
            0, self.diffusion.num_timesteps, (x_vae.shape[0],), device=device
        )
        model_kwargs = dict(xattn_enc=sampled_slots)
        loss_dict = self.diffusion.training_losses(self.pixart_forward, x_vae, t, model_kwargs)
        diff_loss = loss_dict["loss"].mean()

        losses["diff_loss"] = diff_loss
        if diff_loss.item() > 1.25:
            raise ValueError(f"diff_loss is too high: {diff_loss.item()}")
        if self.use_kl:
            losses["kl_loss"] = self.kl_loss_weight * self.kl_loss(mean, logvar)
        
        return losses

class DiffuseSlotPixArtHybridLoRA(DiffuseSlotPixArt):
    def __init__(
        self,
        num_slots=32,
        slot_dim=4,
        num_samplers=32,
        sampler_dim=4,
        enable_nest=False,
        enable_nest_after=-1,
        nest_dist="geometric",
        nest_rho=0.03,
        pixart_path="",
        use_encoder_rgb=False,
        encoder="vit_base_patch16",
        head_type="early_return",
        drop_path_rate=0.1,
        enc_img_size=256,
        pretrained_encoder=None,
        reset_xattn=False,
        lora_r=64,
        lora_alpha=32,
        target_modules='qkv',
        kl_loss=False,
        kl_loss_weight=0.01,
        **kwargs,
    ):
        # __import__("ipdb").set_trace()
        nn.Module.__init__(self)

        # DiT part
        self.diffusion = create_diffusion(timestep_respacing="")
        self.ddim_diffusion = create_diffusion(timestep_respacing="100")

        # loading PixArt
        import safetensors
        from diffusers import PixArtTransformer2DModel
        from peft import LoraConfig, get_peft_model

        self.pixart_dit_config = json.load(
            open(os.path.join(pixart_path, "config.json"), "r"))
        self.pixart_dit = PixArtTransformer2DModel(**self.pixart_dit_config)

        self.vae_config = json.load(
            open(os.path.join(pixart_path, "vae_config.json"), "r"))
        self.vae = AutoencoderKL(**self.vae_config)
        vae_ckpt = {}
        with safetensors.safe_open(os.path.join(pixart_path, "pixart_vae.safetensors"),
                                   framework='pt',
                                   device='cpu') as f:
            for k in f.keys():
                vae_ckpt[k] = f.get_tensor(k)
        print("Loading VAE ckpt")
        msg = self.vae.load_state_dict(vae_ckpt, strict=False)
        print(msg)
        self.scaling_factor = self.vae.config.scaling_factor
        self.vae.requires_grad_(False)

        # image encoder part
        # TODO: make a vit base model with image size being 256
        import paintmind.stage1.vision_transformers as vision_transformer

        self.use_encoder_rgb = use_encoder_rgb
        # import ipdb; ipdb.set_trace()
        self.enc_img_size = enc_img_size
        if use_encoder_rgb:
            encoder_fn = vision_transformer.__dict__[encoder]

            self.encoder = encoder_fn(
                img_size=enc_img_size,
                head_type=head_type,
                drop_path_rate=drop_path_rate,
            )
            self.num_channels = self.encoder.num_features
            self.pretrained_encoder = pretrained_encoder
            if pretrained_encoder is not None:
                # __import__("ipdb").set_trace()
                encoder_ckpt = torch.load(pretrained_encoder, map_location="cpu")
                # drop pos_embed from ckpt
                encoder_ckpt = {
                    k.replace("blocks.", "blocks.0."): v
                    for k, v in encoder_ckpt.items()
                    if not k.startswith("pos_embed")
                }
                msg = self.encoder.load_state_dict(encoder_ckpt, strict=False)
                print("Load encoder from ckpt")
                print(msg)
                self.encoder2slot = nn.Linear(self.num_channels, slot_dim)
            else:
                self.encoder2slot = nn.Conv2d(self.num_channels, slot_dim, 1)

        self.grouping = CausalSemanticGrouping(num_slots, slot_dim)
        self.sampler_latents = nn.Parameter(torch.randn(num_samplers, sampler_dim))
        self.nested_sampler = NestedAttention(
            slot_dim,
            sampler_dim,
            num_slots,
            enable=enable_nest,
            rho=nest_rho,
            nest_dist=nest_dist,
            # map the slots to the caption space.
            output_dim=self.pixart_dit.config.caption_channels,
        )
        self.enable_nest_after = enable_nest_after
        self.use_kl = kl_loss
        self.kl_loss_weight = kl_loss_weight
        if self.use_kl:
            self.slots2gaussian = nn.Linear(slot_dim, slot_dim * 2)

        ckpt = {}
        with safetensors.safe_open(os.path.join(pixart_path, "pixart_alpha256XL_2_resetx_hybrid.safetensors"),
                                   framework='pt',
                                   device='cpu') as f:
            for k in f.keys():
                ckpt[k] = f.get_tensor(k)

        print("Loading ckpt")
        msg = self.load_state_dict(ckpt, strict=False)
        print(msg)
        if target_modules == 'qv':
            target_modules = ['to_q', 'to_v', ]
        elif target_modules == 'qkv':
            target_modules = ['to_q', 'to_k', 'to_v', ]
        elif target_modules == 'all-linear':
            target_modules = ['attn.proj', 'attn.qkv', 'mlp.fc1', 'mlp.fc2', 'patch_embed.proj', 
                              'encoder2slot', 'nested_sampler.attn', 'slot2attn', 'slot2output',
                              'linear_1', 'linear_2', 'pos_embed.proj', 'pixart_dit.proj_out',
                              'attn1.to_q', 'attn1.to_k', 'attn1.to_v', 'attn1.to_out.0',
                              'attn2.to_q', 'attn2.to_k', 'attn2.to_v', 'attn2.to_out.0',
                              'ff.net.0.proj', 'ff.net.2']
        else:
            raise ValueError(f"target_modules must be 'qv' or 'qkv', got {target_modules}")

        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
        )
        self.pixart_dit = get_peft_model(self.pixart_dit, self.lora_config)
        print("Using LoRA")
        self.pixart_dit.print_trainable_parameters()
    

from copy import deepcopy

class ControlT2IDitBlockHalf(nn.Module):
    def __init__(self, base_block, block_index: 0, hidden_size) -> None:
        super().__init__()
        self.copied_block = deepcopy(base_block)
        self.block_index = block_index

        for p in self.copied_block.parameters():
            p.requires_grad_(True)

        self.copied_block.load_state_dict(base_block.state_dict())
        self.copied_block.train()
        
        self.hidden_size = hidden_size
        if self.block_index == 0:
            self.before_proj = nn.Linear(hidden_size, hidden_size)
            nn.init.zeros_(self.before_proj.weight)
            nn.init.zeros_(self.before_proj.bias)
        self.after_proj = nn.Linear(hidden_size, hidden_size) 
        nn.init.zeros_(self.after_proj.weight)
        nn.init.zeros_(self.after_proj.bias)

    def forward(self, x, y, t, mask=None, c=None):
        
        if self.block_index == 0:
            # the first block
            c = self.before_proj(c)
            c = self.copied_block(
                    x + c, 
                    mask,
                    y, 
                    None,
                    t, 
                )
            c_skip = self.after_proj(c)
        else:
            # load from previous c and produce the c for skip connection
            c = self.copied_block(
                    c, 
                    mask,
                    y, 
                    None,
                    t, 
                )
            c_skip = self.after_proj(c)
        
        return c, c_skip
    
from typing import Any, Dict, Optional, Union
from diffusers.utils import is_torch_version
from diffusers.models.modeling_outputs import Transformer2DModelOutput

class DiffuseSlotPixArtControlNet(DiffuseSlotPixArt):
    def __init__(
        self,
        num_slots=32,
        slot_dim=4,
        num_samplers=32,
        sampler_dim=4,
        enable_nest=False,
        enable_nest_after=-1,
        nest_dist="geometric",
        nest_rho=0.03,
        pixart_path="",
        use_encoder_rgb=False,
        encoder="vit_base_patch16",
        head_type="early_return",
        drop_path_rate=0.1,
        enc_img_size=256,
        pretrained_encoder=None,
        reset_xattn=False,
        control_blocks=13,
        **kwargs,
    ):
        super().__init__(
            num_slots=num_slots,
            slot_dim=slot_dim,
            num_samplers=num_samplers,
            sampler_dim=sampler_dim,
            enable_nest=enable_nest,
            enable_nest_after=enable_nest_after,
            nest_dist=nest_dist,
            nest_rho=nest_rho,
            pixart_path=pixart_path,
            use_encoder_rgb=use_encoder_rgb,
            encoder=encoder,
            head_type=head_type,
            drop_path_rate=drop_path_rate,
            enc_img_size=enc_img_size,
            pretrained_encoder=pretrained_encoder,
            reset_xattn=reset_xattn,
            **kwargs,
        )
        self.control_blocks = control_blocks
        self.total_blocks = len(self.pixart_dit.transformer_blocks)
        for p in self.pixart_dit.parameters():
            p.requires_grad_(False)
        if reset_xattn:
            for n, p in self.pixart_dit.named_parameters():
                if 'attn2' in n:
                    p.requires_grad_(True)
        
        self.controlnet = []
        for i in range(self.control_blocks):
            self.controlnet.append(
                ControlT2IDitBlockHalf(
                    self.pixart_dit.transformer_blocks[i],
                    i,
                    self.pixart_dit.config.cross_attention_dim
                )
            )
        self.controlnet = nn.ModuleList(self.controlnet)

    def own_pixart_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        """
        The [`PixArtTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence len, embed dims)`, *optional*):
                Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                self-attention.
            timestep (`torch.LongTensor`, *optional*):
                Used to indicate denoising step. Optional timestep to be applied as an embedding in `AdaLayerNorm`.
            added_cond_kwargs: (`Dict[str, Any]`, *optional*): Additional conditions to be used as inputs.
            cross_attention_kwargs ( `Dict[str, Any]`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            attention_mask ( `torch.Tensor`, *optional*):
                An attention mask of shape `(batch, key_tokens)` is applied to `encoder_hidden_states`. If `1` the mask
                is kept, otherwise if `0` it is discarded. Mask will be converted into a bias, which adds large
                negative values to the attention scores corresponding to "discard" tokens.
            encoder_attention_mask ( `torch.Tensor`, *optional*):
                Cross-attention mask applied to `encoder_hidden_states`. Two formats supported:

                    * Mask `(batch, sequence_length)` True = keep, False = discard.
                    * Bias `(batch, 1, sequence_length)` 0 = keep, -10000 = discard.

                If `ndim == 2`: will be interpreted as a mask, then converted into a bias consistent with the format
                above. This bias will be added to the cross-attention scores.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.unets.unet_2d_condition.UNet2DConditionOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """
        if self.pixart_dit.use_additional_conditions and added_cond_kwargs is None:
            raise ValueError("`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`.")

        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet2DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)
        if attention_mask is not None and attention_mask.ndim == 2:
            # assume that mask is expressed as:
            #   (1 = keep,      0 = discard)
            # convert mask into a bias that can be added to attention scores:
            #       (keep = +0,     discard = -10000.0)
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        batch_size = hidden_states.shape[0]
        height, width = (
            hidden_states.shape[-2] // self.pixart_dit.config.patch_size,
            hidden_states.shape[-1] // self.pixart_dit.config.patch_size,
        )
        hidden_states = self.pixart_dit.pos_embed(hidden_states)

        timestep, embedded_timestep = self.pixart_dit.adaln_single(
            timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
        )

        if self.pixart_dit.caption_projection is not None:
            encoder_hidden_states = self.pixart_dit.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        # 2. Blocks
        for block in self.pixart_dit.transformer_blocks:
            if self.pixart_dit.training and self.pixart_dit.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    cross_attention_kwargs,
                    None,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=None,
                )

        # 3. Output
        shift, scale = (
            self.pixart_dit.scale_shift_table[None] + embedded_timestep[:, None].to(self.pixart_dit.scale_shift_table.device)
        ).chunk(2, dim=1)
        hidden_states = self.pixart_dit.norm_out(hidden_states)
        # Modulation
        hidden_states = hidden_states * (1 + scale.to(hidden_states.device)) + shift.to(hidden_states.device)
        hidden_states = self.pixart_dit.proj_out(hidden_states)
        hidden_states = hidden_states.squeeze(1)

        # unpatchify
        hidden_states = hidden_states.reshape(
            shape=(-1, height, width, self.pixart_dit.config.patch_size, self.pixart_dit.config.patch_size, self.pixart_dit.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, self.pixart_dit.out_channels, height * self.pixart_dit.config.patch_size, width * self.pixart_dit.config.patch_size)
        )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def controlnet_forward(
            self, 
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            timestep: Optional[torch.LongTensor] = None,
            added_cond_kwargs: Dict[str, torch.Tensor] = None,
            cross_attention_kwargs: Dict[str, Any] = None,
            attention_mask: Optional[torch.Tensor] = None,
            encoder_attention_mask: Optional[torch.Tensor] = None,
            return_dict: bool = True,
        ):
        # y is also xattn_enc

        if attention_mask is not None and attention_mask.ndim == 2:
            attention_mask = (1 - attention_mask.to(hidden_states.dtype)) * -10000.0
            attention_mask = attention_mask.unsqueeze(1)

        # convert encoder_attention_mask to a bias the same way we do for attention_mask
        if encoder_attention_mask is not None and encoder_attention_mask.ndim == 2:
            encoder_attention_mask = (1 - encoder_attention_mask.to(hidden_states.dtype)) * -10000.0
            encoder_attention_mask = encoder_attention_mask.unsqueeze(1)

        # 1. Input
        batch_size = hidden_states.shape[0]
        height, width = (
            hidden_states.shape[-2] // self.pixart_dit.config.patch_size,
            hidden_states.shape[-1] // self.pixart_dit.config.patch_size,
        )
        hidden_states = self.pixart_dit.pos_embed(hidden_states)

        timestep, embedded_timestep = self.pixart_dit.adaln_single(
            timestep, added_cond_kwargs, batch_size=batch_size, hidden_dtype=hidden_states.dtype
        )

        if self.pixart_dit.caption_projection is not None:
            encoder_hidden_states = self.pixart_dit.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(batch_size, -1, hidden_states.shape[-1])

        # 2. Blocks
        # for block in self.pixart_dit.transformer_blocks:
        # process first block
        # __import__('ipdb').set_trace()
        hidden_states = self.pixart_dit.transformer_blocks[0](
            hidden_states,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            timestep=timestep,
            cross_attention_kwargs=cross_attention_kwargs,
            class_labels=None,
        )
        # c for controlnet, make it addable to the hidden_states
        c = F.interpolate(encoder_hidden_states.transpose(1, 2), hidden_states.shape[1])
        c = c.transpose(1, 2)
        for i in range(1, self.control_blocks + 1):
            block = self.pixart_dit.transformer_blocks[i]
            controlnet_block = self.controlnet[i - 1]
            if self.pixart_dit.training and self.pixart_dit.gradient_checkpointing:
                raise NotImplementedError("Not implemented yet")
            else:
                c, c_skip = controlnet_block(
                    hidden_states,
                    encoder_hidden_states,
                    timestep,
                    attention_mask,
                    c
                )
                hidden_states = block(
                    hidden_states + c_skip,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=None,
                )

        for i in range(self.control_blocks + 1, self.total_blocks):
            block = self.pixart_dit.transformer_blocks[i]
            hidden_states = block(
                hidden_states,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timestep,
                cross_attention_kwargs=cross_attention_kwargs,
                class_labels=None,
            )
        
        # 3. Output
        shift, scale = (
            self.pixart_dit.scale_shift_table[None] + embedded_timestep[:, None].to(self.pixart_dit.scale_shift_table.device)
        ).chunk(2, dim=1)
        hidden_states = self.pixart_dit.norm_out(hidden_states)
        # Modulation
        hidden_states = hidden_states * (1 + scale.to(hidden_states.device)) + shift.to(hidden_states.device)
        hidden_states = self.pixart_dit.proj_out(hidden_states)
        hidden_states = hidden_states.squeeze(1)

        # unpatchify
        hidden_states = hidden_states.reshape(
            shape=(-1, height, width, self.pixart_dit.config.patch_size, self.pixart_dit.config.patch_size, self.pixart_dit.out_channels)
        )
        hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
        output = hidden_states.reshape(
            shape=(-1, self.pixart_dit.out_channels, height * self.pixart_dit.config.patch_size, width * self.pixart_dit.config.patch_size)
        )

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)

    def pixart_forward(self, x, t, xattn_enc):
        added_cond_kwargs = {
            'resolution': None,
            'aspect_ratio': None,
        }
        return self.controlnet_forward(
            x, xattn_enc, t, added_cond_kwargs
            ).sample


    def forward(self, x, targets, latents=None, sample=False, epoch=None, inference_with_n_slots=-1):
        losses = {}
        batch_size = x.shape[0]
        device = x.device
        # __import__('ipdb').set_trace()
        if latents is None:
            x_vae = self.vae_encode(x)  # (N, C, H, W)
        else:
            x_vae = latents

        if self.use_encoder_rgb:
            # __import__("ipdb").set_trace()
            if self.pretrained_encoder is not None:
                x = F.interpolate(x, size=224)
            x_enc = self.encoder(x)
            x_enc = self.encoder2slot(x_enc)
        else:
            x_enc = x_vae
        if (
            epoch is not None
            and epoch > self.enable_nest_after
            and self.enable_nest_after != -1
            and not self.nested_sampler.enable
        ):
            self.nested_sampler.set_enable(True)

        slots, dots = self.grouping(x_enc)  # (n, num_slots, slot_dim)
        # __import__("ipdb").set_trace()

        sampler_latents = repeat(self.sampler_latents, "s d -> b s d", b=batch_size)
        sampled_slots, attn_score = self.nested_sampler(
            slots, sampler_latents, inference_with_n_slots=inference_with_n_slots
        )

        if sample:
            return self.sample(x_vae, sampled_slots, device)

        t = torch.randint(
            0, self.diffusion.num_timesteps, (x_vae.shape[0],), device=device
        )
        model_kwargs = dict(xattn_enc=sampled_slots)
        loss_dict = self.diffusion.training_losses(self.pixart_forward, x_vae, t, model_kwargs)
        diff_loss = loss_dict["loss"].mean()

        losses["diff_loss"] = diff_loss
        if diff_loss.item() > 1.25:
            raise ValueError(f"diff_loss is too high: {diff_loss.item()}")
        
        return losses
    
