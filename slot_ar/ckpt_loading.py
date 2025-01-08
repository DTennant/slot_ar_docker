from diffusers import UnCLIPImageVariationPipeline
import torch, numpy as np
from PIL import Image
import torchvision
from glob import glob
from sklearn.decomposition import PCA
import textwrap
import os, sys

from paintmind.stage1.titok import TiTok

from omegaconf import OmegaConf
config = OmegaConf.load('configs/vit_titok.yaml')
model = TiTok(**config['trainer']['params']['model']['params'])
__import__("ipdb").set_trace()
output = model(torch.randn(1, 3, 256, 256), None)
print(output[0].shape)


ckpt = torch.load('tokenizer_titok_l32.bin', map_location='cpu')
ckpt = {k: v for k, v in ckpt.items() if 'encoder' not in k}
msg = model.load_state_dict(ckpt, strict=False)
print(msg)