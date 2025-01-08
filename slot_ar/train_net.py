import numpy as np
import os, pdb, time
import torch_fidelity
import os.path as osp
import argparse
from omegaconf import OmegaConf
from paintmind.engine.util import instantiate_from_config

def train_on_coco():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='configs/vit_vqgan.yaml')
    args = parser.parse_args()

    cfg_file = args.cfg
    assert osp.exists(cfg_file)
    config = OmegaConf.load(cfg_file)
    trainer = instantiate_from_config(config.trainer)
    trainer.train(args.cfg)

if __name__ == '__main__':

    train_on_coco()
