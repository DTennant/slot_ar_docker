import matplotlib.pyplot as plt
from omegaconf import OmegaConf
from paintmind.engine.util import instantiate_from_config
import torch

from paintmind.stage1.diffuse_slot import DiffuseSlot
from safetensors import safe_open

ckpt = {}
# with safe_open('/mnt/ceph_rbd/wx/work/slot_ar/output/'
#                'dit_ffhq256_2500ep_nest_after1500_32bs/models/step600000/model.safetensors',
# with safe_open('/mnt/ceph_rbd/zbc/slot_ar/output/imagenet820k.safetensors',
# with safe_open('/disk/scratch_big/bingchen/dataset/semanticist_ckpts/imagenet100_990k.safetensors',
with safe_open('/disk/scratch_big/bingchen/dataset/semanticist_ckpts/ffhq600k.safetensors',
               framework='pt', 
               device='cpu') as f:
    for key in f.keys():
        ckpt[key] = f.get_tensor(key)


# cfg = OmegaConf.load('configs/imagenet100_cache.yaml')
# cfg = OmegaConf.load('/mnt/ceph_rbd/zbc/slot_ar/configs/dit_clevr4.yaml')
cfg = OmegaConf.load('configs/ffhq1264.yaml')

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


model = DiffuseSlot(**cfg['trainer']['params']['model']['params'])
msg = model.load_state_dict(ckpt, strict=False)
model.cuda();

print(msg)
model.eval();
@torch.no_grad()
def get_model_pred(img, n_slots=64, ddim=True):
    slots, _ = model.encode_slots(img.to('cuda'))
    sampled_from_slots = model.sample_from_slots(
        slots, 
        slots.device, 
        inference_with_n_slots=n_slots, 
        ddim=ddim
    )
    return sampled_from_slots, slots

@torch.no_grad()
def get_slots(img):
    slots, _ = model.encode_slots(img.to('cuda'))
    return slots

from glob import glob
class ffhq_dataset:
    def __init__(self):
        self.data_path = sorted(glob('/disk/scratch_big/bingchen/dataset/ffhq/images256x256/*.png'))
    def __len__(self):
        return len(self.data_path)
    def __getitem__(self, idx):
        img = Image.open(self.data_path[idx])
        img, _ = transform(img, None)
        return img


import facer
face_detector = facer.face_detector("retinaface/mobilenet", device='cuda')
face_attr = facer.face_attr("farl/celeba/224", device='cuda')

labels = face_attr.labels

print(labels)

def get_facer_attr(img_path):
    image = facer.hwc2bchw(
        facer.read_hwc(img_path)
    ).to('cuda')
    with torch.inference_mode():
        faces = face_detector(image)
        faces = face_attr(image, faces)
    attr_dict = {label: prob.item() for label, prob in zip(labels, faces['attrs'][0])}
    return attr_dict

def get_facer_attr_from_tensor(img):
    # img is already in bchw format, that is, 1, 3, 224, 224
    image = img.to('cuda')
    with torch.inference_mode():
        faces = face_detector(image)
        faces = face_attr(image, faces)
    attr_dict = {label: prob.item() for label, prob in zip(labels, faces['attrs'][0])}
    return attr_dict

ori_images = ffhq_dataset()
ori_dl = torch.utils.data.DataLoader(ori_images, batch_size=64, shuffle=False)

from tqdm import tqdm
import json
# ori_attr_dicts = []
# for img in tqdm(ori_images.data_path):
#     attr_dict = get_facer_attr(img)
#     ori_attr_dicts.append(attr_dict)
# json.dump(ori_attr_dicts, open('ori_attr.json', 'w'))

err_cnt = 0

for num_slots in [1, 2, 4, 8, 16, 32, 64]:
    print(f'num_slots: {num_slots}')
    slot_attr_dicts = []
    for i, img in enumerate(tqdm(ori_dl)):
        sampled_from_slots, slots = get_model_pred(img, n_slots=num_slots)
        sampled_from_slots = sampled_from_slots * 255
        for per_img in sampled_from_slots:
            try:
                attr_dict = get_facer_attr_from_tensor(per_img.unsqueeze(0))
                slot_attr_dicts.append(attr_dict)
            except:
                slot_attr_dicts.append({'error': 'no face'})
                err_cnt += 1
    json.dump(slot_attr_dicts, open(f'slot_attr_{num_slots}.json', 'w'))

print(f'error count: {err_cnt}')

