import os, pdb
import io
import torchvision
import numpy as np
import pandas as pd
import os.path as osp
from glob import glob
from PIL import Image
import torch, zipfile
# from datasets import load_dataset
from pycocotools.coco import COCO
from ..data.coco import coco_panoptic_transforms, vae_transforms, vae_cache_transforms
from ..data.coco_panoptic import build_coco_panotic_dataset

def unzip_file(zip_src, tgt_dir):
    if zipfile.is_zipfile(zip_src):
        fz = zipfile.ZipFile(zip_src, 'r')
        for file in fz.namelist():
            fz.extract(file, tgt_dir)       
    else:
        raise RuntimeError("This is not zip file.")

        
class Laion:
    def __init__(self, metadata_path, folder_path, fid='folder', key='key', caption_col='caption', transform=None):
        self.df = pd.read_parquet(metadata_path)
        self.fpath = folder_path
        self.fid = fid
        self.key = key
        self.caption_col = caption_col
        self.transform = transform
        
    def __getitem__(self, idx):
        fid = self.df[self.fid][idx]
        key = self.df[self.key][idx]
        img_path = f"{self.fpath}/{fid}/{key}.jpg"
        img = Image.open(img_path).convert('RGB')
        caption = self.df[self.caption_col][idx]
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, caption
    
    def __len__(self):
        return len(self.df)        

                 
class LaionV2:
    def __init__(self, metadata_path, folder_path, fid='folder', key='key', caption_col=['caption', 'prompt'], p=[0.2, 0.8], transform=None):
        self.df = pd.read_parquet(metadata_path)
        self.fpath = folder_path
        self.fid = fid
        self.key = key
        self.caption_col = caption_col
        self.p = p
        self.transform = transform
        
    def __getitem__(self, idx):
        fid = self.df[self.fid][idx]
        key = self.df[self.key][idx]
        img_path = f"{self.fpath}/{fid}/{key}.jpg"
        img = Image.open(img_path).convert('RGB')
        
        prompts = []
        for col in self.caption_col:
            prompts.append(self.df[col][idx])
        caption = np.random.choice(prompts, p=self.p)
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, caption
    
    def __len__(self):
        return len(self.df)
  
        
class ImageNet:
    def __init__(self, root, split='train', use_vae=False, transform=None, img_size=256, scale=0.8):
        self.transform = coco_panoptic_transforms(split, img_size=img_size, scale=scale)
        if use_vae:
            self.transform = vae_transforms(split, img_size=img_size, scale=scale)
        # __import__("ipdb").set_trace()
        self.dataset = torchvision.datasets.ImageNet(root=root, split=split)
        # self.transform = transform
        self.prefix = ["an image of ", "a picture of "]
        
    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        caption = np.random.choice(self.prefix) + np.random.choice(self.dataset.classes[target])
        
        if self.transform is not None:
            image, _ = self.transform(image, None)
            
        return image, caption
    
    def __len__(self):
        return len(self.dataset)

class ImageNet_debug:
    def __init__(self, root, split='train', use_vae=False, transform=None, debug_len=64, img_size=256, scale=0.8):
        self.transform = coco_panoptic_transforms(split, img_size=img_size, scale=scale)
        if use_vae:
            self.transform = vae_transforms(split, img_size=img_size, scale=scale)
        # __import__("ipdb").set_trace()
        self.dataset = torchvision.datasets.ImageNet(root=root, split=split)
        # self.transform = transform
        self.prefix = ["an image of ", "a picture of "]
        self.debug_len = debug_len
        
    def __getitem__(self, idx):
        image, target = self.dataset[idx % self.debug_len]
        caption = np.random.choice(self.prefix) + np.random.choice(self.dataset.classes[target])
        
        if self.transform is not None:
            image, _ = self.transform(image, None)
            
        return image, caption
    
    def __len__(self):
        return len(self.dataset)

class Clevr4:
    def __init__(self, root, use_vae=False, img_size=256):
        if use_vae:
            self.transform = vae_transforms('train', img_size=img_size, scale=0.8)
        else:
            self.transform = None
        self.img_list = glob(os.path.join(root, '*.png'))
    
    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        caption = "a CLEVR image"
        if self.transform is not None:
            img, _ = self.transform(img, None)
        return img, caption

    def __len__(self):
        return len(self.img_list)

    
# NOTE: Imagewoof
class ImageWoof:
    def __init__(self, root, split='train', use_vae=False, 
                     cache_mode=False, cache_latent_file=None,
                     transform=None, img_size=256, scale=0.8):
        self.dataset = torchvision.datasets.ImageFolder(osp.join(root, split))
        self.transform = coco_panoptic_transforms(split, img_size=img_size, scale=scale)
        if use_vae:
            self.transform = vae_transforms(split, img_size=img_size, scale=scale)
            if cache_mode:
                self.transform = vae_cache_transforms(split, img_size=img_size, scale=scale)
        self.prefix = ["an image of ", "a picture of "]

        if cache_mode and os.path.exists(cache_latent_file):
            self.cache_mode = True
            self.latent_cache = torch.load(cache_latent_file)
            print("Cache loaded.")
        else:
            self.cache_mode = False

    def __getitem__(self, idx):
        image, target = self.dataset[idx]
        # caption = np.random.choice(self.prefix) + np.random.choice(self.dataset.classes[target])
        
        if self.transform is not None:
            image, _ = self.transform(image, None)

        if self.cache_mode:
            latent = self.latent_cache[idx]
            return image, latent, target
            
        return image, target

    def __len__(self):
        return len(self.dataset)
        

class FFHQDataset:
    def __init__(self, root, img_size=256, use_vae=False, transform=None):
        self.folder_path = root
        self.img_files = glob(os.path.join(root, '*.png'))
        self.transform = transform
        if use_vae:
            self.transform = vae_transforms('train', img_size=img_size, scale=0.8)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = Image.open(img_path).convert('RGB')
        target = -1

        if self.transform is not None:
            img, _ = self.transform(img, None)
        
        return img, target

    def __len__(self):
        return len(self.img_files)
    
class CelebADataset:
    def __init__(self, root, img_size=64, use_vae=False, transform=None):
        self.folder_path = root
        self.img_files = glob(os.path.join(root, '*.jpg'))
        self.transform = transform
        if use_vae:
            self.transform = vae_transforms('train', img_size=img_size, scale=0.8)

    def __getitem__(self, idx):
        img_path = self.img_files[idx]
        img = Image.open(img_path).convert('RGB')
        target = -1

        if self.transform is not None:
            img, _ = self.transform(img, None)
        
        return img, target

    def __len__(self):
        return len(self.img_files)
    
           
        
class Flickr30k:
    def __init__(self, img_dir, ann_file, transform=None):
        self.dataset = torchvision.datasets.Flickr30k(root=img_dir, ann_file=ann_file)
        self.transform = transform
        
    def __getitem__(self, idx):
        img, captions = self.dataset[idx]
        caption = np.random.choice(captions)
        
        if self.transform is not None:
            img = self.transform(img)
            
        return img, caption
    
    def __len__(self):
        return len(self.dataset)
        

class DiffusionDB:
    def __init__(self, version='large_random_100k', transform=None):
        self.dataset = load_dataset("poloclub/diffusiondb", version)['train']
        self.transform = transform
        
    def __getitem__(self, idx):
        data = self.dataset[idx]
        
        image = data['image']
        prompt = data['prompt']
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image, prompt
        
    def __len__(self):
        return len(self.dataset)

class CoCoPanoptic:

    def __init__(self, root, image_set = 'train', img_size=256, scale=0.8):

        self.transforms = coco_panoptic_transforms(image_set, img_size=img_size, scale=scale)
        self.dataset = build_coco_panotic_dataset(image_set,  root, transforms = self.transforms)
    
    def add_transform(self, transforms=None):

        self.transforms = transforms

    def __len__(self):

        return len(self.dataset)
    
    def __getitem__(self, idx):

        return self.dataset[idx % len(self)]

        
class CoCo:
    def __init__(self, root, dataType='train2017', annType='captions', transform=None):
        self.root = root
        self.img_dir = '{}/{}'.format(root, dataType)
        annFile = '{}/annotations/{}_{}.json'.format(root, annType, dataType)
        self.coco = COCO(annFile)
        self.imgids = self.coco.getImgIds()
        self.transform = transform
    
    def add_transform(self, transform=None):

        self.transform = transform
        
    def __getitem__(self, idx):

        imgid = self.imgids[idx]
        img_name = self.coco.loadImgs(imgid)[0]['file_name']
        annid = self.coco.getAnnIds(imgIds=imgid)
        img = Image.open(osp.join(self.img_dir, img_name)).convert('RGB')
        ann = np.random.choice(self.coco.loadAnns(annid))['caption']
        if self.transform is not None:
            img = self.transform(img)
        
        return img, ann     
        
    def __len__(self):

        return len(self.imgids)


class CelebA:
    def __init__(self, root, type='identity', transform=None):
        """CelebA Dataset http://personal.ie.cuhk.edu.hk/~lz013/projects/CelebA.html
        Args:
            root (str): CelebA Dataset folder path
            type (str, optional): 'identity' or 'attr'. Defaults to 'identity'.
            transform (torchvision.transforms, optional): torchvision.transforms. Defaults to None.
        """        
        ann_dir = os.path.join(root, 'Anno')
        base_dir = os.path.join(root, 'Img')
        zfile_path = os.path.join(base_dir, 'img_align_celeba.zip')
        self.img_dir = os.path.join(base_dir, 'img_align_celeba')
        if os.path.exists(self.img_dir):
            pass
        elif os.path.exists(zfile_path):
            unzip_file(zfile_path, base_dir)
        else:
            raise RuntimeError("Dataset not found.")
        self.imgs = os.listdir(self.img_dir)
        if type == 'identity':
            self.img2id = {}
            with open(os.path.join(ann_dir, 'identity_CelebA.txt'), 'r') as f:
                for line in f.readlines():
                    name, id = line.strip().split(' ')
                    self.img2id[name] = int(id)
        self.transform = transform
        
    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img = Image.open(os.path.join(self.img_dir, img_name)).convert('RGB')
        ann = self.img2id[img_name]
        if self.transform is not None:
            img = self.transform(img)
        
        ann = torch.tensor(ann)
            
        return img, ann
    
    def __len__(self):
        return len(self.imgs)


class LSUN:
    def __init__(self, root, split='train', use_vae=False, transform=None, img_size=256, scale=0.8):
        import string
        import pickle
        import lmdb
        self.transform = coco_panoptic_transforms(split, img_size=img_size, scale=scale)
        if use_vae:
            self.transform = vae_transforms(split, img_size=img_size, scale=scale)
        
        self.env = lmdb.open(root, max_readers=1, readonly=True, lock=False, readahead=False, meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = txn.stat()["entries"]
        cache_file = "_cache_" + "".join(c for c in root if c in string.ascii_letters)
        if os.path.isfile(cache_file):
            self.keys = pickle.load(open(cache_file, "rb"))
        else:
            with self.env.begin(write=False) as txn:
                self.keys = [key for key in txn.cursor().iternext(keys=True, values=False)]
            pickle.dump(self.keys, open(cache_file, "wb"))

    def __getitem__(self, index):
        img, target = None, 0
        env = self.env
        with env.begin(write=False) as txn:
            imgbuf = txn.get(self.keys[index])

        buf = io.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

        if self.transform is not None:
            img, _ = self.transform(img, None)

        return img, target

    def __len__(self):
        return self.length