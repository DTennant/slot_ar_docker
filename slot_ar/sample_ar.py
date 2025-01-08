from omegaconf import OmegaConf
from paintmind.engine.util import instantiate_from_config
import torch

cfg = 'configs/dit_enc_rgb_woof_ar_exp_large.yaml'
cfg = OmegaConf.load(cfg)
cfg['trainer']['params']['model']['params']['ckpt_path'] = \
'output/dit_woof_3460ep_enable_after_3400ep_noln/models/step485000/'
trainer = instantiate_from_config(cfg.trainer)

from mar_test import Mar, DiffLoss
import torch

mar_dict = torch.load('mar_model.pth', map_location='cpu')
mar_loss_dict = torch.load('mar_loss_fn.pth', map_location='cpu')

model = Mar().cuda()
loss_fn = DiffLoss(
        target_channels=8, 
        z_channels=512, 
        depth=12, 
        width=1536, 
        num_sampling_steps="100"
    ).cuda()

model.load_state_dict(mar_dict)
loss_fn.load_state_dict(mar_loss_dict)

__import__('ipdb').set_trace()
bos_start = model.bos_embed.data.unsqueeze(0)
bos_sampled = loss_fn.sample(bos_start)