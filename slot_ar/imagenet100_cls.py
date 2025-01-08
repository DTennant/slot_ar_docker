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
torch.multiprocessing.set_sharing_strategy('file_system')

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
    def __init__(self, trm=None, split='val'):
        self.classes = imagenet100_classes
        self.imgs = []
        self.labels = []
        for c in self.classes:
            self.imgs += sorted(glob(f'/disk/scratch_fast/datasets/ImageNet1k/{split}/{c}/*.JPEG'))
            self.labels += [self.classes.index(c)] \
                * len(glob(f'/disk/scratch_fast/datasets/ImageNet1k/{split}/{c}/*.JPEG'))
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

train_imagenet100_ds = imagenet100_dataset(trm=val_transform, split='train')
train_imagenet100_dl = torch.utils.data.DataLoader(train_imagenet100_ds, batch_size=128, shuffle=False, num_workers=8)

imagenet100_map2imagenet1k = {i: imagenet_ds.wnids.index(k) for i, k in enumerate(imagenet100_ds.classes)}
pred_slots = [8, 16, 32, 64, 128]

@torch.no_grad()
def get_slots(img):
    slots, _ = model.encode_slots(img.to('cuda'))
    return slots

import os

if os.path.exists('imagenet100_slot_data_cls.pt'):
    data = torch.load('imagenet100_slot_data_cls.pt')
    X_train, y_train, X_test, y_test = data['X_train'], data['y_train'], data['X_test'], data['y_test']
    print('Loaded data')
else:

    pred = []
    with torch.no_grad():
        pred_imgs = []
        labels = []
        for slot_img, label, img in tqdm(imagenet100_dl):
            slot_img = slot_img.cuda()
            slots = get_slots(slot_img)
            pred_imgs.append(slots.cpu())
            labels.append(label)
        pred_imgs = torch.cat(pred_imgs, dim=0)
        labels = torch.cat(labels)
    X_test, y_test = pred_imgs, labels

    pred = []
    with torch.no_grad():
        pred_imgs = []
        labels = []
        for slot_img, label, img in tqdm(train_imagenet100_dl):
            slot_img = slot_img.cuda()
            slots = get_slots(slot_img)
            pred_imgs.append(slots.cpu())
            labels.append(label)
        pred_imgs = torch.cat(pred_imgs, dim=0)
        labels = torch.cat(labels)
    X_train, y_train = pred_imgs, labels

    torch.save({'X_train': X_train, 'y_train': y_train, 'X_test': X_test, 'y_test': y_test}, 'imagenet100_slot_data_cls.pt')


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# import ipdb; ipdb.set_trace()
# X_train, X_test, y_train, y_test = train_test_split(pred_imgs, labels, test_size=0.2, random_state=42)

slots_score = {}
for n_slots in pred_slots:
    # Initialize the logistic regression model
    clf = LogisticRegression(max_iter=5000, n_jobs=32)

    # Fit the model on the training set
    clf.fit(X_train[:, :n_slots].reshape(X_train.shape[0], -1), y_train)

    # Evaluate on the test set
    y_pred = clf.predict(X_test[:, :n_slots].reshape(X_test.shape[0], -1))
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Linear probing accuracy with {n_slots} slots: {accuracy:.4f}")
    slots_score[n_slots] = accuracy

for k, v in slots_score.items():
    print(f'with {k} slots: {v:.4f}')




