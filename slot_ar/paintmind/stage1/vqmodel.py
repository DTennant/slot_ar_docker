import torch, pdb
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDPMScheduler
from .layers import Encoder, Decoder
from .quantize import VectorQuantizer
from .slot_quantize import SlotVectorQuantizer
from ..engine.util import instantiate_from_config
from ..modules.ocl.perceptual_grouping import build_mlp

class GaussianDiffusion:
    def __init__(self, num_timesteps=1000, beta_start=0.0001, beta_end=0.02):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.sqrt_betas = torch.sqrt(self.betas)

        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise):
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        x_noisy = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
        return x_noisy.detach()

    def p_sample(self, model, x, t):
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_betas_t = self._extract(self.sqrt_betas, t, x.shape)
        sqrt_alphas_t = self._extract(self.sqrt_alphas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        
        pred_noise = model(x, t)
        mean = 1. / sqrt_alphas_t * (x - betas_t / sqrt_one_minus_alphas_cumprod_t * pred_noise)
        noise = torch.randn_like(x) if t[0] > 0 else torch.zeros_like(x)
        x_prev = mean + sqrt_betas_t * noise
        return x_prev.clamp(-1, 1)

    def _extract(self, a, t, shape):
        batch_size = t.shape[0]
        out = a.to(t.device).gather(-1, t)
        return out.view(batch_size, *((1,) * (len(shape) - 1)))

class SimpleMLP(nn.Module):
    def __init__(self, depth, width, time_embed_dim=128):
        super(SimpleMLP, self).__init__()
        self.time_embedding = nn.Embedding(1000, time_embed_dim)  # Assuming 1000 timesteps
        self.fc_time = nn.Linear(time_embed_dim, width)
        
        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(width, width)

    def forward(self, x, t):
        # Get time step embedding
        t_embed = self.time_embedding(t)
        t_embed = self.fc_time(t_embed)
        
        # Merge x and t_embed
        x = x + t_embed
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)

class CondMLP(nn.Module):
    def __init__(self, depth, width, cond_dim, time_embed_dim=128):
        super(CondMLP, self).__init__()
        self.time_embedding = nn.Embedding(1000, time_embed_dim)  # Assuming 1000 timesteps
        self.fc_time = nn.Linear(time_embed_dim, width)

        self.cond_proj = nn.Linear(cond_dim, width)
        
        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(width, width)

    def forward(self, x, c, t):
        # Get time step embedding
        t_embed = self.time_embedding(t)
        t_embed = self.fc_time(t_embed)

        # Get condition embedding
        c_embed = self.cond_proj(c)
        
        # Merge x, c, and t_embed
        x = x + c + t_embed
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)


class Diffusion(nn.Module):

    def __init__(self, depth, width):
        # SimpleMLP takes in x_t, timestep, and condition, and outputs predicted noise.
        self.net = SimpleMLP(depth, width)

        # GaussianDiffusion offers forward and backward functions q_sample and p_sample.
        self.diffusion = GaussianDiffusion()

    # Given ground truth token x, compute loss
    def loss(self, x):
        # sample random noise and timestep
        noise = torch.randn_like(x.shape)
        timestep = torch.randint(0, self.diffusion.num_timesteps, x.size(0))

        # sample x_t from x
        x_t = self.diffusion.q_sample(x, timestep, noise)

        # predict noise from x_t
        noise_pred = self.net(x_t, timestep)

        # L2 loss
        loss = ((noise_pred - noise) ** 2).mean()

        # optional: loss += loss_vlb

        return loss

    # Given noise, sample x using reverse diffusion process
    def sample(self, noise):
        x = noise
        for t in list(range(self.diffusion.num_timesteps))[::-1]:
            timestep = torch.tensor([t]).repeat(x.size(0))
            x = self.diffusion.p_sample(self.net, x, timestep)
        return x


class VQModel(nn.Module):
    def __init__(self, n_embed, embed_dim, beta, encoder_config, decoder_config, 
                 n_slots, slot_dim, use_diffusion=False, **kwargs):
        super().__init__()

        self.num_slots = n_slots
        self.slots = nn.Embedding(n_slots, slot_dim)
        self.encoder = instantiate_from_config(encoder_config)
        self.decoder = instantiate_from_config(decoder_config)
        self.use_diffusion = use_diffusion
        if self.use_diffusion:
            # TODO: implement the diffusion loss part
            # self.diffusion = Diffusion(depth, width)
            # do we consider LDM(reconstruct the latents) or DDPM(images)?
            # NOTE: hard code for now, will change in the next commit
            self.num_timesteps = 1000
            self.diffuse_sched = DDPMScheduler(num_train_timesteps=self.num_timesteps, beta_schedule='squaredcos_cap_v2')
            self.diffuse_mlp = CondMLP(depth=3, width=512, cond_dim=slot_dim)
            
        self.attn_type = encoder_config.params.get('attn_type', 'normal')

        self.query_quantisation = VectorQuantizer(2 * n_embed, embed_dim, beta)

        self.prev_quant = nn.Linear(encoder_config.params.dim, embed_dim)
        self.post_quant = nn.Linear(embed_dim, decoder_config.params.dim)  
  
    def freeze(self):
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def generate_attention_mask(self, n, m):
        # Create an initial mask of size (n + m) x (n + m) filled with zeros
        mask = np.zeros((n + m, n + m))

        # First n tokens can see themselves only
        mask[:n, :n] = 1

        # Second m tokens can see all first n tokens and themselves causally
        mask[n:, :n] = 1
        mask[n:, n:] = np.tril(np.ones((m, m)))

        return torch.from_numpy(mask)#.to(self.device)
    
    def encode(self, x):
        
        bs = x.size(0)
        slots = self.slots.weight.unsqueeze(0).expand(bs, -1, -1)
        if self.attn_type == 'causal':
            x, slots = self.encoder(x, slots)
        else:
            x = self.encoder(x, slots)
            slots = x[:, -self.num_slots:]
        
        slots = self.prev_quant(slots)
        queries, q_loss, q_indices = self.query_quantisation(slots)

        return queries, q_loss, q_indices
    
    def decode(self, slots):

        x = self.decoder(slots)
        return x.clamp(-1.0, 1.0)

    def diffuse(self, slots, gt):
        # slots are the condition
        # gt is the ground truth image components
        batch_size = slots.size(0)
        device = slots.device

        noise = torch.rand_like(gt)
        timesteps = torch.randint(0, self.num_timesteps, (batch_size,)).long().to(device)
        
        noisy_gt = self.diffuse_sched.add_noise(gt, noise, timesteps)
        pred = self.diffuse_mlp(noisy_gt, slots, timesteps)
        loss = F.mse_loss(pred, noise)

    
    def forward(self, img, targets):
        # __import__("ipdb").set_trace()
        queries, q_loss, q_indices = self.encode(img)

        slots = self.post_quant(queries)

        if self.use_diffusion:
            # TODO: get gt image components here
            diffuse_loss = self.diffuse(slots)

        rec = self.decode(slots)

        results = {}
        if self.training:
            results.update({'q_loss': q_loss})
        

        return rec, results
    
    def decode_from_indice(self, indice):

        z_q = self.quantize.decode_from_indice(indice)
        img = self.decode(z_q)
        return img
    
    def from_pretrained(self, path):

        return self.load_state_dict(torch.load(path, map_location='cpu'))


        
