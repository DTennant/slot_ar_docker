import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from paintmind.engine.util import instantiate_from_config
import torch

from paintmind.stage1.diffuse_slot import DiffuseSlot
from safetensors import safe_open

ckpt = {}
# with safe_open('/mnt/ceph_rbd/wx/work/slot_ar/output/'
#                'dit_ffhq256_2500ep_nest_after1500_32bs/models/step600000/model.safetensors',
# with safe_open('/mnt/ceph_rbd/zbc/slot_ar/output/dit_clevr4_ep400/models/step312400/model.safetensors',
#                framework='pt', 
#                device='cpu') as f:
with safe_open('/mnt/ceph_rbd/zbc/slot_ar/output/imagenet820k.safetensors',
               framework='pt', 
               device='cpu') as f:
    for key in f.keys():
        ckpt[key] = f.get_tensor(key)

# print(ckpt.keys())

# cfg = OmegaConf.load('/mnt/ceph_rbd/wx/work/slot_ar/configs/dit_ffhq256_2000ep.yaml')
# cfg = OmegaConf.load('/mnt/ceph_rbd/zbc/slot_ar/configs/dit_clevr4.yaml')

cfg = OmegaConf.load('/mnt/ceph_rbd/zbc/slot_ar/configs/imagenet_pretrained_1664.yaml')
# model = DiffuseSlot(**cfg['trainer']['params']['model']['params'])

# msg = model.load_state_dict(ckpt, strict=False)

# print(msg)

# model = model.cuda()
# model = torch.nn.DataParallel(model)

from paintmind.data.coco import vae_transforms
from PIL import Image

transform = vae_transforms('test')


def norm_ip(img, low, high):
    img.clamp_(min=low, max=high)
    img.sub_(low).div_(max(high - low, 1e-5))

def norm_range(t, value_range):
    if value_range is not None:
        norm_ip(t, value_range[0], value_range[1])
    else:
        norm_ip(t, float(t.min()), float(t.max()))

from PIL import Image
def convert_np(img):
    ndarr = img.mul(255).add_(0.5).clamp_(0, 255)\
            .permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    return ndarr
def convert_PIL(img):
    ndarr = img.mul(255).add_(0.5).clamp_(0, 255)\
            .permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    img = Image.fromarray(ndarr)
    return img

from glob import glob
class clevr4:
    def __init__(self):
        # self.img_list = glob('/mnt/ceph_rbd/zbc/ffhq-dataset/images256x256/*.png')
        # self.img_list = glob('/mnt/ceph_rbd/zbc/data/clevr-4/images/*.png')
        self.img_list = glob('/mnt/ceph_rbd/zbc/data/imagenet100/train/*/*')
        self.img_list = sorted(self.img_list)[:10000]
        self.transform = transform

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        img, _ = self.transform(img, None)
        return img

ds = clevr4()


import os
import torch
from torch.utils.data import DataLoader, Subset
from multiprocessing import Pool, cpu_count, set_start_method
from tqdm import tqdm
from PIL import Image
set_start_method('spawn', force=True)

# Define the number of processes
num_processes = min(cpu_count(), torch.cuda.device_count())
gpu_indices = list(range(num_processes))

# Define the output directory
output_dir = '/mnt/ceph_rbd/zbc/imagenet100_fid_test'
os.makedirs(output_dir, exist_ok=True)

def process_batch(batch, model, gpu_index):
    """Process a batch of images and return the output."""
    # __import__('ipdb').set_trace()
    model = model.to(f'cuda:{gpu_index}')
    model_output = model(batch.to(f'cuda:{gpu_index}'), None, sample=True).cpu()
    return model_output

def save_outputs(outputs, start_idx):
    """Save the outputs to disk."""
    # __import__('ipdb').set_trace()
    for i, img in enumerate(outputs):
        img_pil = convert_PIL(img)
        img_pil.save(os.path.join(output_dir, f'output_{start_idx + i}.png'))

def worker(args):
    """Worker function to process a subset of the dataset."""
    subset_indices, gpu_index = args
    subset = Subset(ds, subset_indices)
    dl = DataLoader(subset, batch_size=32, shuffle=False, num_workers=4)
    
    # Load the model on the specific GPU
    model = DiffuseSlot(**cfg['trainer']['params']['model']['params'])
    model.load_state_dict(ckpt, strict=False)
    model = model.to(f'cuda:{gpu_index}')
    
    all_outputs = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dl, desc=f"Process {os.getpid()} on GPU {gpu_index}")):
            start_idx = subset_indices[0] + batch_idx * 128
            outputs = process_batch(batch, model, gpu_index)
            all_outputs.extend(outputs)
    
    # Save all outputs after processing the entire subset
    save_outputs(all_outputs, subset_indices[0])

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_processes', type=int, default=4)
    parser.add_argument('--pid', type=int, default=0)
    args = parser.parse_args()

    dataset_size = len(ds)
    indices = list(range(dataset_size))
    chunk_size = dataset_size // num_processes
    chunks = [indices[i:i + chunk_size] for i in range(0, dataset_size, chunk_size)]
    chunk = chunks[args.pid]
    worker((chunk, gpu_indices[args.pid]))

