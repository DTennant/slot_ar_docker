import torch
import torchvision
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from diffusers import DDPMScheduler, UNet2DModel, AutoencoderKL
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using device: {device}')

from pathlib import Path

path_data = Path('test_data')
path_data.mkdir(exist_ok=True)
path = path_data/'bedroom'

from PIL import Image
import numpy as np
bs = 64
def to_img(f): 
    return np.array(Image.open(f))/255

from glob import glob
class ImagesDS:
    def __init__(self, spec):
        self.path = Path(path)
        self.files = glob(str(spec), recursive=True)
    def __len__(self): 
        return len(self.files)
    def __getitem__(self, i): 
        return torch.from_numpy(to_img(self.files[i]).transpose(2, 0, 1)[:, :256,:256]).float(), 0
    
dataset = ImagesDS(path / f'**/*.jpg')

vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-ema").cuda().requires_grad_(False)

from paintmind.stage1.diffusion_transfomers import DiT_B_4
from paintmind.stage1.diffusion import create_diffusion

model = DiT_B_4()
diffusion = create_diffusion(timestep_respacing="")

from tqdm import tqdm
# Redefining the dataloader to set the batch size higher than the demo of 8
train_dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=32)

# How many runs through the data should we do?
n_epochs = 25

# Our network 
net = model.to(device)

# Our loss function
loss_fn = nn.MSELoss()

# The optimizer
opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=0) 

# Keeping a record of the losses for later viewing
losses = []
print_every = 100



# The training loop
step = 0
total_steps = n_epochs * len(train_dataloader)
pbar = tqdm(total=total_steps)
for epoch in range(n_epochs):
    for x, y in train_dataloader:
        step += 1

        x = x.to(device) # Data on the GPU (mapped to (-1, 1))
        with torch.no_grad():
            # Map input images to latent space + normalize latents:
            x = vae.encode(x).latent_dist.sample().mul_(0.18215)
        y = y.to(device)

        t = torch.randint(0, diffusion.num_timesteps, (x.shape[0],), device=device)
        model_kwargs = dict(y=y)
        loss_dict = diffusion.training_losses(model, x, t, model_kwargs)
        loss = loss_dict["loss"].mean()

        # Backprop and update the params:
        opt.zero_grad()
        loss.backward()
        opt.step()

        # Store the loss for later
        losses.append(loss.item())
        if step % print_every == 0:
            print(f'Step {step}. Loss: {loss.item()}')
        pbar.set_postfix({'Loss': loss.item()})
        pbar.update(1)

    # Sample per epoch
    z = torch.randn(4, 4, 32, 32, device=device)
    y = torch.tensor([0, 0, 0, 0], device=device)

    # Setup classifier-free guidance:
    # z = torch.cat([z, z], 0)
    # y_null = torch.tensor([1000] * 4, device=device)
    # y = torch.cat([y, y_null], 0)
    model_kwargs = dict(y=y)#, cfg_scale=0.0)

    # Sample images:
    samples = diffusion.p_sample_loop(
        model.forward, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True, device=device
    )
    # samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
    samples = vae.decode(samples / 0.18215).sample
    samples = torchvision.utils.make_grid(samples.cpu()).numpy()
    samples = np.clip(samples, 0, 1)
    plt.figure(figsize=(10, 10))
    plt.imshow(samples.transpose(1, 2, 0))
    plt.savefig(f"output/dit_lsun_base/step{str(step).zfill(6)}.png", dpi=150)

    torch.save(net.state_dict(), f"output/dit_lsun_base/step{str(step).zfill(6)}.pt")

    avg_loss = sum(losses[-100:])/100
    print(f'Finished epoch {epoch}. Average of the last 100 loss values: {avg_loss:05f}')


pbar.close()