import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from diffusers import DDPMScheduler
from .quantize import VectorQuantizer
from .maskgit_vqgan import Decoder as Pixel_Decoder
from .maskgit_vqgan import VectorQuantizer as Pixel_Quantizer
from ..engine.util import instantiate_from_config
from omegaconf import OmegaConf

class TiTok(nn.Module):
    def __init__(self, n_embed, embed_dim, beta, encoder_config, decoder_config, 
                 n_slots, slot_dim, use_diffusion=False, decoder_ckpt_path=None, **kwargs):
        super().__init__()

        # __import__("ipdb").set_trace()
        self.num_slots = n_slots
        self.slots = nn.Embedding(n_slots, slot_dim)
        # I have not checked the encoder part, remaining the same as VQModel
        self.encoder = instantiate_from_config(encoder_config)
        self.decoder = instantiate_from_config(decoder_config)
            
        self.attn_type = encoder_config.params.get('attn_type', 'normal')

        self.query_quantisation = VectorQuantizer(2 * n_embed, embed_dim, beta)

        self.prev_quant = nn.Linear(encoder_config.params.dim, embed_dim)
        # self.post_quant = nn.Linear(embed_dim, decoder_config.params.dim)  
        self.post_quant = nn.Linear(embed_dim, decoder_config.params.token_size)

        self.pixel_quantize = Pixel_Quantizer(
            num_embeddings=1024, embedding_dim=256, commitment_cost=0.25)
        
        self.pixel_decoder = Pixel_Decoder(OmegaConf.create(
            {"channel_mult": [1, 1, 2, 2, 4],
            # {"channel_mult": [1, 2, 2, 4, 4],
            #  "num_resolutions": 4, # 256x256
             "num_resolutions": 5, # 256x256
             "dropout": 0.0,
             "hidden_channels": 128,
             "num_channels": 3,
             "num_res_blocks": 2,
             "resolution": 256,
             "z_channels": 256}))
            
        if decoder_ckpt_path is not None:
            ckpt = torch.load(decoder_ckpt_path, map_location='cpu')
            ckpt = {k: v for k, v in ckpt.items() if 'encoder' not in k} # only load decoders
            msg = self.load_state_dict(ckpt, strict=False)
            print(msg)

  
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
        return x#.clamp(-1.0, 1.0)

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
        raise NotImplementedError
    
    def forward(self, img, targets):
        # __import__("ipdb").set_trace()
        queries, q_loss, q_indices = self.encode(img)

        slots = self.post_quant(queries)
        slots = rearrange(slots, 'b n d -> b d 1 n')
        decoded_latent = self.decode(slots)
        quantized_states = torch.einsum(
            'nchw,cd->ndhw', decoded_latent.softmax(1),
            self.pixel_quantize.embedding.weight)
        decoded = self.pixel_decoder(quantized_states)

        results = {}
        if self.training:
            results.update({'q_loss': q_loss})
        return decoded, results
    
    def decode_from_indice(self, indice):
        z_q = self.quantize.decode_from_indice(indice)
        img = self.decode(z_q)
        return img
    
    def from_pretrained(self, path):
        return self.load_state_dict(torch.load(path, map_location='cpu'))


        
