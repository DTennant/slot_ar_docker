# Adopted from LDM's KL-VAE: https://github.com/CompVis/latent-diffusion
import torch
import torch.nn as nn

import numpy as np


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels, num_groups=32):
    return torch.nn.GroupNorm(
        num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True
    )


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(
                in_channels, in_channels, kernel_size=3, stride=2, padding=0
            )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(
        self,
        *,
        in_channels,
        out_channels=None,
        conv_shortcut=False,
        dropout,
        temb_channels=512,
    ):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels, out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1
        )
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=1, padding=1
                )
            else:
                self.nin_shortcut = torch.nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=1, padding=0
                )

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.k = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.v = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )
        self.proj_out = torch.nn.Conv2d(
            in_channels, in_channels, kernel_size=1, stride=1, padding=0
        )

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        out_ch=3,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(16,),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=256,
        z_channels=16,
        double_z=True,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # downsampling
        self.conv_in = torch.nn.Conv2d(
            in_channels, self.ch, kernel_size=3, stride=1, padding=1
        )

        curr_res = resolution
        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in,
            2 * z_channels if double_z else z_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, x):
        # assert x.shape[2] == x.shape[3] == self.resolution, "{}, {}, {}".format(x.shape[2], x.shape[3], self.resolution)

        # timestep embedding
        temb = None

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class Decoder(nn.Module):
    def __init__(
        self,
        *,
        ch=128,
        out_ch=3,
        ch_mult=(1, 1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(),
        dropout=0.0,
        resamp_with_conv=True,
        in_channels=3,
        resolution=256,
        z_channels=16,
        give_pre_end=False,
        **ignore_kwargs,
    ):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]
        curr_res = resolution // 2 ** (self.num_resolutions - 1)
        self.z_shape = (1, z_channels, curr_res, curr_res)
        print(
            "Working with z of shape {} = {} dimensions.".format(
                self.z_shape, np.prod(self.z_shape)
            )
        )

        # z to block_in
        self.conv_in = torch.nn.Conv2d(
            z_channels, block_in, kernel_size=3, stride=1, padding=1
        )

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(
            in_channels=block_in,
            out_channels=block_in,
            temb_channels=self.temb_ch,
            dropout=dropout,
        )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks + 1):
                block.append(
                    ResnetBlock(
                        in_channels=block_in,
                        out_channels=block_out,
                        temb_channels=self.temb_ch,
                        dropout=dropout,
                    )
                )
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(
            block_in, out_ch, kernel_size=3, stride=1, padding=1
        )

    def forward(self, z):
        # assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(
                device=self.parameters.device
            )

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(
            device=self.parameters.device
        )
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample, dims=[1, 2, 3]):
        if self.deterministic:
            return torch.Tensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self):
        return self.mean


class AutoencoderKL(nn.Module):
    def __init__(self, embed_dim, ch_mult, logvar_init=0.0, use_variational=True, ckpt_path=None):
        super().__init__()
        self.encoder = Encoder(ch_mult=ch_mult, z_channels=embed_dim)
        self.decoder = Decoder(ch_mult=ch_mult, z_channels=embed_dim)
        self.use_variational = use_variational
        mult = 2 if self.use_variational else 1
        self.quant_conv = torch.nn.Conv2d(2 * embed_dim, mult * embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, embed_dim, 1)
        self.embed_dim = embed_dim
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path)
            
        # output log variance
        self.logvar = nn.Parameter(torch.ones(size=()) * logvar_init)

    def init_from_ckpt(self, path):
        sd = torch.load(path, map_location="cpu")["model"]
        msg = self.load_state_dict(sd, strict=False)
        print("Loading pre-trained KL-VAE")
        print("Missing keys:")
        print(msg.missing_keys)
        print("Unexpected keys:")
        print(msg.unexpected_keys)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        if not self.use_variational:
            moments = torch.cat((moments, torch.ones_like(moments)), 1)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, inputs, sample_posterior=False):
        losses = {}
        posterior = self.encode(inputs)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        # NOTE: kl loss for the posterior
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        losses['kl_loss'] = kl_loss
        return dec, losses, self.logvar
        # return dec, posterior

    # def _forward(self, inputs, disable=True, train=True, optimizer_idx=0):
    #     if train:
    #         return self.training_step(inputs, disable, optimizer_idx)
    #     else:
    #         return self.validation_step(inputs, disable)
        
        
    # def training_step(self, inputs, batch_idx, optimizer_idx):
    #     # inputs = self.get_input(batch, self.image_key)
    #     reconstructions, posterior = self(inputs)

    #     if optimizer_idx == 0:
    #         # train encoder+decoder+logvar
    #         aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
    #                                         last_layer=self.get_last_layer(), split="train")
    #         self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
    #         self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
    #         return aeloss

    #     if optimizer_idx == 1:
    #         # train the discriminator
    #         discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, optimizer_idx, self.global_step,
    #                                             last_layer=self.get_last_layer(), split="train")

    #         self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
    #         self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
    #         return discloss

    # def validation_step(self, batch, batch_idx):
    #     inputs = self.get_input(batch, self.image_key)
    #     reconstructions, posterior = self(inputs)
    #     aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
    #                                     last_layer=self.get_last_layer(), split="val")

    #     discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
    #                                         last_layer=self.get_last_layer(), split="val")

    #     self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
    #     self.log_dict(log_dict_ae)
    #     self.log_dict(log_dict_disc)
    #     return self.log_dict
    
    
from torch.distributions import Geometric

class NestedDropout(nn.Module):
    def __init__(self, num_units, rho):
        super().__init__()
        self.num_units = num_units
        self.rho = rho
        self.geo_dist = Geometric(rho)
        
    def geometric_sample(self, num):
        return self.geo_dist.sample([num]) + 1

    def forward(self, x):
        if self.training:
            # __import__("ipdb").set_trace()
            batch_size = x.shape[0]
            # Sample b values for the entire batch at once
            b = self.geometric_sample(batch_size)
            b = torch.clamp(b, max=self.num_units).to(x.device)
            
            # Create an arange tensor for comparison
            arange = torch.arange(self.num_units).expand(batch_size, -1).to(x.device)
            
            # Create the mask using broadcasting
            mask = (arange < b.unsqueeze(1)).float().unsqueeze(-1)
            return x * mask.to(x.device)
        return x
    
from paintmind.stage1.slotcoder import CausalSemanticGrouping


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

class AutoencoderNested(AutoencoderKL):
    def __init__(self, embed_dim, ch_mult, num_slot, dim_slot, img_size, use_ramp=True, ramp_method='sigmoid', rho=0.01, logvar_init=0.0, use_variational=True, load_ckpt_path=None, **kwargs):
        super(AutoencoderNested, self).__init__(embed_dim, ch_mult, logvar_init, use_variational, load_ckpt_path)

        self.proj_slot = torch.nn.Conv2d(embed_dim, dim_slot, 1)
        self.grouping = CausalSemanticGrouping(num_slot, dim_slot)
        self.proj_back = torch.nn.Linear(dim_slot, embed_dim)
        
        self.img_size = img_size
        # patch_size 16
        self.reorg_slots = torch.nn.Linear(num_slot, (self.img_size // 16) ** 2)
        
        self.nested_dropout = NestedDropout(num_slot, rho)
        
        self.use_ramp = use_ramp
        if ramp_method == 'sigmoid':
            self.ramp_method = sigmoid_rampup
        elif ramp_method == 'linear':
            self.ramp_method = linear_rampup
        else:
            raise ValueError(f"Ramp method {ramp_method} not supported")


    def forward(self, inputs, sample_posterior=False, ramp_step=None, ramp_length=None):
        # __import__("ipdb").set_trace()
        losses = {}
        posterior = self.encode(inputs)
        if sample_posterior:
            enc = posterior.sample()
        else:
            enc = posterior.mode()

        z = self.proj_slot(enc)
        z, dots = self.grouping(z)
        # TODO: check the implementation of nested dropout
        # __import__("ipdb").set_trace()
        z = self.nested_dropout(z)
        z = self.proj_back(z)
        z = z.permute(0, 2, 1)
        z = self.reorg_slots(z)

        # TODO: need to reshape the slots to the original shape
        z = z.view(-1, self.embed_dim, self.img_size // 16, self.img_size // 16)
        
        # annealing
        if ramp_step is not None and self.use_ramp:
            ramp_step = max(0, ramp_step - 1)
            ramp_step = min(ramp_step, ramp_length)
            ramp_factor = self.ramp_method(ramp_step, ramp_length)
            z = ramp_factor * z + (1 - ramp_factor) * enc

        dec = self.decode(z)

        # NOTE: kl loss for the posterior
        kl_loss = posterior.kl()
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        losses['kl_loss'] = kl_loss
        return dec, losses, self.logvar
        

        

