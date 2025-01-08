import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
from diffusers import AutoencoderKL
from PIL import Image
import numpy as np
import argparse

class ImageDataset(Dataset):
    def __init__(self, input_dir, transform=None):
        self.input_dir = input_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.input_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, img_name

def process_batch(batch, vae, scaling_factor, res, output_dir):
    images, filenames = batch
    
    with torch.no_grad():
        enc = vae.encode(images).latent_dist.sample().mul_(scaling_factor)
        dec = vae.decode(enc / scaling_factor).sample

    for i, (input_img, output_img, filename) in enumerate(zip(images, dec, filenames)):
        input_img = (input_img.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        Image.fromarray(input_img).save(os.path.join(output_dir, f'{res}x{res}', 'inputs', filename))
        
        output_img = output_img.permute(1, 2, 0).cpu().clamp(0, 1).numpy()
        output_img = (output_img * 255).astype(np.uint8)
        Image.fromarray(output_img).save(os.path.join(output_dir, f'{res}x{res}', 'outputs', filename))

def main(args):
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # vae = AutoencoderKL.from_pretrained('stabilityai/sd-vae-ft-ema').to(device)
    vae = AutoencoderKL.from_pretrained('stabilityai/sdxl-vae').to(device)
    scaling_factor = vae.config.scaling_factor

    input_dir = '/mnt/ceph_rbd/zbc/ffhq-dataset/images1024x1024'
    output_dir = './face_results'  # Replace with your desired output path

    os.makedirs(os.path.join(output_dir, f'{args.res}x{args.res}', 'inputs'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, f'{args.res}x{args.res}', 'outputs'), exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((args.res, args.res)),
        transforms.ToTensor(),
    ])

    dataset = ImageDataset(input_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=16)

    for batch in tqdm(dataloader, desc=f"Processing {args.res}x{args.res}"):
        images, filenames = batch
        images = images.to(device)
        process_batch((images, filenames), vae, scaling_factor, args.res, output_dir)

    print("Processing complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process images using VAE")
    parser.add_argument("--res", type=int, choices=[128, 256], required=True, help="resolution to process images")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID to use")
    args = parser.parse_args()
    main(args)