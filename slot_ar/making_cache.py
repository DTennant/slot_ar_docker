import numpy as np
import os, pdb, time
import torch_fidelity
import tqdm
import torch
import os.path as osp
import argparse
from omegaconf import OmegaConf
from paintmind.engine.util import instantiate_from_config


@torch.no_grad()
def caching():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/vit_vqgan.yaml')
    args = parser.parse_args()

    cfg_file = args.cfg
    assert osp.exists(cfg_file)
    config = OmegaConf.load(cfg_file)
    dataset = instantiate_from_config(config.trainer.params.dataset)
    model = instantiate_from_config(config.trainer.params.model)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=config.trainer.params.batch_size, 
        shuffle=False, 
        num_workers=config.trainer.params.num_workers,
    )
    # Each batch will give us a (N, C, H, W) tensor of images
    # We need to cache them and save them to a pth file
    cache_save_file = config.trainer.params.latent_cache_file
    cache = []
    # import ipdb; ipdb.set_trace()
    model.cuda()
    model.eval()
    for idx, batch in enumerate(tqdm.tqdm(dataloader)):
        batch = batch[0].cuda()
        latent = model.vae_encode(batch)
        cache.append(latent.cpu())
    cache = torch.cat(cache, dim=0)
    torch.save(cache, cache_save_file)

if __name__ == '__main__':

    caching()

