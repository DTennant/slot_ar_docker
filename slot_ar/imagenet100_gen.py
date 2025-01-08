import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from paintmind.engine.util import instantiate_from_config
import torch

from paintmind.stage1.diffuse_slot import DiffuseSlot
from safetensors import safe_open
from paintmind.data.coco import vae_transforms
from PIL import Image
import urllib
import json
from torchvision.datasets import ImageNet
import torchvision as tv

val_transform = tv.transforms.Compose([
    tv.transforms.Resize(256),
    tv.transforms.CenterCrop(224),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


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

ckpt = {}
# with safe_open('/mnt/ceph_rbd/wx/work/slot_ar/output/'
#                'dit_ffhq256_2500ep_nest_after1500_32bs/models/step600000/model.safetensors',
# with safe_open('/mnt/ceph_rbd/zbc/slot_ar/output/imagenet820k.safetensors',
with safe_open('/disk/scratch_big/bingchen/dataset/semanticist_ckpts/imagenet100_990k.safetensors',
# with safe_open('/disk/scratch_big/bingchen/dataset/semanticist_ckpts/ffhq600k.safetensors',
               framework='pt', 
               device='cpu') as f:
    for key in f.keys():
        ckpt[key] = f.get_tensor(key)


cfg = OmegaConf.load('configs/imagenet100_cache.yaml')
# cfg = OmegaConf.load('/mnt/ceph_rbd/zbc/slot_ar/configs/dit_clevr4.yaml')
# cfg = OmegaConf.load('configs/ffhq1264.yaml')

transform = vae_transforms('test')

model = DiffuseSlot(**cfg['trainer']['params']['model']['params'])
msg = model.load_state_dict(ckpt, strict=False)
model.cuda();
model.eval();


@torch.no_grad()
def get_model_pred(img, n_slots=64, ddim=True):
    slots, _ = model.encode_slots(img.to('cuda'))
    sampled_from_slots = model.sample_from_slots(
        slots, 
        slots.device, 
        inference_with_n_slots=n_slots, 
        ddim=True
    )
    return sampled_from_slots, slots

imagenet100_classes = open('/disk/scratch_big/bingchen/dataset/imagenet100_slot.txt').read().splitlines()
from glob import glob
class imagenet100_dataset:
    def __init__(self, trm=None):
        self.classes = imagenet100_classes
        self.imgs = []
        self.labels = []
        for c in self.classes:
            self.imgs += sorted(glob(f'/disk/scratch_fast/datasets/ImageNet1k/val/{c}/*.JPEG'))
            self.labels += [self.classes.index(c)] \
                * len(glob(f'/disk/scratch_fast/datasets/ImageNet1k/val/{c}/*.JPEG'))
        self.trm = trm

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = Image.open(self.imgs[idx]).convert('RGB')
        # if self.trm is None:
        n_img, _ = transform(img, None)
        v_img = val_transform(img)
        # else:
            # img = self.trm(img)
        return n_img, self.labels[idx], v_img
        
imagenet_ds = ImageNet('/disk/scratch_fast/datasets/ImageNet1k/', split='val', transform=val_transform)

from tqdm import tqdm
imagenet100_ds = imagenet100_dataset(trm=val_transform)
imagenet100_dl = torch.utils.data.DataLoader(imagenet100_ds, batch_size=128, shuffle=False, num_workers=8)

imagenet100_map2imagenet1k = {i: imagenet_ds.wnids.index(k) for i, k in enumerate(imagenet100_ds.classes)}
# pred_slots = [8, 16, 32, 64, 128]
pred_slots = [8]
    
from torch.utils.data import Subset
import os
def split_dataset(dataset, world_size, rank):
    """Split a dataset into `world_size` parts and return the subset for the current `rank`."""
    total_size = len(dataset)
    indices = list(range(total_size))
    per_thread = total_size // world_size
    remainder = total_size % world_size
    
    # Determine start and end indices for this rank
    start_idx = rank * per_thread + min(rank, remainder)
    end_idx = start_idx + per_thread + (1 if rank < remainder else 0)
    return Subset(dataset, indices[start_idx:end_idx])


for n_slots in pred_slots:
    pred = []
    with torch.no_grad():

        slot_save_path = f'/disk/scratch_big/bingchen/dataset/imagenet100_slot_{n_slots}/'

        pred_imgs = []
        for slot_img, label, img in tqdm(imagenet100_dl, desc=f"Prediction with {n_slots} slots"):
            slot_img = slot_img.cuda()
            pred_img, _ = get_model_pred(slot_img, n_slots=n_slots)
            pred_imgs.append(pred_img.cpu())
        pred_imgs = torch.cat(pred_imgs, dim=0)
        os.makedirs(slot_save_path, exist_ok=True)
        for i, img in enumerate(tqdm(pred_imgs, desc=f"Saving images with {n_slots} slots")):
            img = convert_PIL(img)
            img.save(f'{slot_save_path}/{str(i).zfill(5)}.JPEG')




