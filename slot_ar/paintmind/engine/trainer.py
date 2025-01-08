import os, torch
import random, pdb
import numpy as np
import os.path as osp
import torch_fidelity
import torch.nn as nn
from lpips import LPIPS
from tqdm.auto import tqdm
from einops import rearrange
import torch.nn.functional as F
from paintmind.optim import Lion
from accelerate import Accelerator
from ..data.misc import collate_fn, collate_fn_cache
from torch.optim import Adam, AdamW
from torch.autograd import Variable
from .util import instantiate_from_config
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader, random_split, DistributedSampler
from torchtnt.utils.data import CudaDataPrefetcher
from paintmind.utils.lr_scheduler import build_scheduler
from accelerate.utils import DistributedDataParallelKwargs
from paintmind.stage1.discriminator import NLayerDiscriminator


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        if self.batch_sampler is None:
            self.sampler = _RepeatSampler(self.sampler)
        else:
            self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return (
            len(self.sampler)
            if self.batch_sampler is None
            else len(self.batch_sampler.sampler)
        )

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def hinge_d_loss(fake, real):
    loss_fake = torch.mean(F.relu(1.0 + fake))
    loss_real = torch.mean(F.relu(1.0 - real))
    d_loss = 0.5 * (loss_real + loss_fake)
    return d_loss


def g_nonsaturating_loss(fake):
    loss = F.softplus(-fake).mean()

    return loss


class Log:
    def __init__(self):
        self.data = {}

    def add(self, name_value):
        for name, value in name_value.items():
            if name not in self.data:
                self.data[name] = value
            else:
                self.data[name] += value

    def update(self, name_value):
        for name, value in name_value.items():
            self.data[name] = value

    def reset(self):
        self.data = {}

    def __getitem__(self, name):
        return self.data[name]


class VQGANTrainer(nn.Module):
    def __init__(
        self,
        model,
        dataset,
        num_epoch,
        valid_size=32,
        lr=1e-4,
        lr_min=5e-5,
        warmup_steps=50000,
        warmup_lr_init=1e-6,
        decay_steps=None,
        batch_size=32,
        num_workers=0,
        pin_memory=False,
        max_grad_norm=1.0,
        grad_accum_steps=1,
        mixed_precision="bf16",
        save_every=10000,
        sample_every=1000,
        result_folder=None,
        log_dir="./log",
        steps=0,
        d_weight=0.1,
        per_weight=0.1,
    ):
        super().__init__()
        # kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        self.accelerator = Accelerator(
            kwargs_handlers=[kwargs],
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=grad_accum_steps,
            log_with="tensorboard",
            project_dir=log_dir,
        )

        self.vqvae = instantiate_from_config(model)

        dataset = instantiate_from_config(dataset)

        self.discr = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3)

        train_size = len(dataset) - valid_size
        self.train_ds, self.valid_ds = random_split(
            dataset,
            [train_size, valid_size],
            generator=torch.Generator().manual_seed(42),
        )
        print(f"train dataset size: {train_size}, valid dataset size: {valid_size}")

        self.train_dl = DataLoader(
            self.train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
        self.valid_dl = DataLoader(
            self.valid_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

        self.g_optim = Adam(self.vqvae.parameters(), lr=lr, betas=(0.9, 0.99))
        self.d_optim = Adam(self.discr.parameters(), lr=lr, betas=(0.9, 0.99))
        self.g_sched = build_scheduler(
            self.g_optim,
            num_epoch,
            len(self.train_dl),
            lr_min,
            warmup_steps,
            warmup_lr_init,
            decay_steps,
        )
        self.d_sched = build_scheduler(
            self.d_optim,
            num_epoch,
            len(self.train_dl),
            lr_min,
            warmup_steps,
            warmup_lr_init,
            decay_steps,
        )

        self.per_loss = LPIPS(net="vgg").to(self.device).eval()
        for param in self.per_loss.parameters():
            param.requires_grad = False
        self.d_loss = hinge_d_loss
        self.g_loss = g_nonsaturating_loss
        self.d_weight = d_weight
        self.per_weight = per_weight

        self.num_epoch = num_epoch
        self.save_every = save_every
        self.samp_every = sample_every
        self.max_grad_norm = max_grad_norm

        self.model_saved_dir = os.path.join(result_folder, "models")
        os.makedirs(self.model_saved_dir, exist_ok=True)

        self.image_saved_dir = os.path.join(result_folder, "images")
        os.makedirs(self.image_saved_dir, exist_ok=True)

        (
            self.vqvae,
            self.discr,
            self.g_optim,
            self.d_optim,
            self.g_sched,
            self.d_sched,
            self.train_dl,
            self.valid_dl,
        ) = self.accelerator.prepare(
            self.vqvae,
            self.discr,
            self.g_optim,
            self.d_optim,
            self.g_sched,
            self.d_sched,
            self.train_dl,
            self.valid_dl,
        )
        self.accelerator.register_for_checkpointing(self.g_sched, self.d_sched)
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"number of learnable parameters: {n_parameters//1e6}M")

        # __import__("ipdb").set_trace()
        self.steps = steps
        if (model.params.ckpt_path is not None) & osp.exists(model.params.ckpt_path):
            # model.params.ckpt_path is something like 'path/to/models/step10/'
            self.loaded_steps = int(
                model.params.ckpt_path.split("step")[-1].split("/")[0]
            )
            path = model.params.ckpt_path
            self.accelerator.load_state(path)
        else:
            # NOTE: ckpt_path is None
            self.loaded_steps = -1

    @property
    def device(self):
        return self.accelerator.device

    def calculate_gradient_penalty(self, real_images, fake_images, lambda_term=10):

        eta = (
            torch.FloatTensor(real_images.shape[0], 1, 1, 1)
            .uniform_(0, 1)
            .to(self.device)
        )
        eta = eta.expand(
            real_images.shape[0],
            real_images.size(1),
            real_images.size(2),
            real_images.size(3),
        )

        interpolated = eta * real_images + ((1 - eta) * fake_images)
        interpolated = Variable(interpolated, requires_grad=True)
        return interpolated

    def calculate_loss_gradient_penalty(self, rec, img, lambda_term=10):

        bs = rec.size(0)
        assert bs == img.size(0)
        interpolated = self.calculate_gradient_penalty(
            img,
            rec,
        )
        images = torch.cat((rec, img, interpolated), dim=0)

        preds = self.discr(images)

        preds = rearrange(preds, "(b f) ... -> b f ...", b=3, f=bs)
        fakes, reals, prob_interpolated = preds[0], preds[1], preds[2]

        d_loss = hinge_d_loss(fakes, reals)

        gradients = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size()).to(rec),
            create_graph=True,
            retain_graph=True,
        )[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_term
        return d_loss, grad_penalty

    def train(self, config_path=None):

        self.accelerator.init_trackers("vqgan")
        self.log = Log()
        for epoch in range(self.num_epoch):
            with tqdm(
                self.train_dl,
                dynamic_ncols=True,
                disable=not self.accelerator.is_main_process,
            ) as train_dl:
                # __import__("ipdb").set_trace()
                for batch in train_dl:
                    if isinstance(batch, tuple) or isinstance(batch, list):
                        img, targets = batch[0].tensors, batch[1]
                    else:
                        img = batch

                    self.steps += 1
                    if self.steps <= self.loaded_steps:
                        continue

                    # discriminator part
                    requires_grad(self.vqvae, False)
                    requires_grad(self.discr, True)
                    with self.accelerator.accumulate(self.discr):
                        with self.accelerator.autocast():
                            rec, codebook_loss = self.vqvae(img, targets)
                            d_loss, grad_penalty = self.calculate_loss_gradient_penalty(
                                rec, img
                            )
                            d_loss = d_loss + grad_penalty

                        self.accelerator.backward(d_loss)
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(
                                self.discr.parameters(), self.max_grad_norm
                            )
                        self.d_optim.step()
                        self.d_sched.step_update(self.steps)
                        self.d_optim.zero_grad()

                        self.log.update(
                            {
                                "d loss": d_loss.item(),
                                "d lr": self.d_optim.param_groups[0]["lr"],
                            }
                        )

                    # generator part
                    requires_grad(self.vqvae, True)
                    # requires_grad(self.vqvae.module.pixel_quantize, False)
                    # requires_grad(self.vqvae.module.pixel_decoder, False)
                    requires_grad(self.discr, False)
                    with self.accelerator.accumulate(self.vqvae):
                        with self.accelerator.autocast():
                            rec, codebook_loss = self.vqvae(img, targets)
                            # reconstruction loss
                            rec_loss = F.l1_loss(
                                rec, img
                            )  # + F.mse_loss(rec, img) #NOTE: because I am using slotcon for training now
                            # perception loss
                            per_loss = self.per_loss(rec, img).mean()
                            # gan loss
                            g_loss = self.g_loss(self.discr(rec))
                            # combine
                            loss = (
                                sum([v for _, v in codebook_loss.items()])
                                + rec_loss
                                + self.per_weight * per_loss
                                + self.d_weight * g_loss
                            )

                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(
                                self.vqvae.parameters(), self.max_grad_norm
                            )
                        self.g_optim.step()
                        self.g_sched.step_update(self.steps)
                        self.g_optim.zero_grad()

                    self.log.update(
                        {
                            "rec loss": rec_loss.item(),
                            "per loss": per_loss.item(),
                            "g loss": g_loss.item(),
                            "g lr": self.g_optim.param_groups[0]["lr"],
                        }
                    )

                    if not (self.steps % self.save_every):
                        self.save()

                    if not (self.steps % self.samp_every):
                        self.evaluate()

                    train_dl.set_postfix(
                        ordered_dict={
                            "epoch": epoch,
                            "reconstruct loss": self.log["rec loss"],
                            "perceptual loss": self.log["per loss"],
                            "g_loss": self.log["g loss"],
                            "d_loss": self.log["d loss"],
                            "g_lr": self.log["g lr"],
                        }
                    )
                    self.accelerator.log(
                        {
                            "reconstruct loss": self.log["rec loss"],
                            "perceptual loss": self.log["per loss"],
                            "g_loss": self.log["g loss"],
                            "d_loss": self.log["d loss"],
                            "g_lr": self.log["g lr"],
                            "d_lr": self.log["d lr"],
                        },
                        step=self.steps,
                    )

        self.accelerator.end_training()
        print("Train finished!")

    def unwrap_state_dict(self, item):
        state_dict = self.accelerator.unwrap_model(item).state_dict()
        return state_dict

    def save(self):
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(
            os.path.join(self.model_saved_dir, f"step{self.steps}")
        )

        # state_dict = self.accelerator.unwrap_model(self.vqvae).state_dict()
        # os.makedirs(os.path.join(self.model_saved_dir, f'step{self.steps}'), exist_ok=True)
        # self.accelerator.save(state_dict, os.path.join(self.model_saved_dir, f'step{self.steps}/vit_vq_step_{self.steps}.pt'))
        # discr = self.unwrap_state_dict(self.discr)
        # g_optim = self.unwrap_state_dict(self.g_optim)
        # d_optim = self.unwrap_state_dict(self.d_optim)
        # g_sched = self.unwrap_state_dict(self.g_sched)
        # d_sched = self.unwrap_state_dict(self.d_sched)
        # self.accelerator.save(discr, os.path.join(self.model_saved_dir, f'step{self.steps}/discr_step_{self.steps}.pt'))
        # self.accelerator.save(g_optim, os.path.join(self.model_saved_dir, f'step{self.steps}/g_optim_step_{self.steps}.pt'))
        # self.accelerator.save(d_optim, os.path.join(self.model_saved_dir, f'step{self.steps}/d_optim_step_{self.steps}.pt'))
        # self.accelerator.save(g_sched, os.path.join(self.model_saved_dir, f'step{self.steps}/g_sched_step_{self.steps}.pt'))
        # self.accelerator.save(d_sched, os.path.join(self.model_saved_dir, f'step{self.steps}/d_sched_step_{self.steps}.pt'))

    @torch.no_grad()
    def evaluate(self):
        self.vqvae.eval()
        with tqdm(
            self.valid_dl,
            dynamic_ncols=True,
            disable=not self.accelerator.is_local_main_process,
        ) as valid_dl:
            for i, batch in enumerate(valid_dl):
                if isinstance(batch, tuple) or isinstance(batch, list):
                    img, targets = batch[0].tensors, batch[1]
                else:
                    img = batch

                rec, _ = self.vqvae(img, targets)
                imgs_and_recs = torch.stack((img, rec), dim=0)
                imgs_and_recs = rearrange(imgs_and_recs, "r b ... -> (b r) ...")
                imgs_and_recs = imgs_and_recs.detach().cpu().float()

                grid = make_grid(
                    imgs_and_recs, nrow=6, normalize=True, value_range=(-1, 1)
                )
                save_image(
                    grid,
                    os.path.join(self.image_saved_dir, f"step_{self.steps}_{i}.png"),
                )
        self.vqvae.train()


class VAETrainer(nn.Module):
    def __init__(
        self,
        model,
        dataset,
        num_epoch,
        valid_size=32,
        lr=1e-4,
        lr_min=5e-5,
        warmup_steps=50000,
        warmup_lr_init=1e-6,
        decay_steps=None,
        batch_size=32,
        num_workers=0,
        pin_memory=False,
        max_grad_norm=1.0,
        grad_accum_steps=1,
        mixed_precision="bf16",
        save_every=10000,
        sample_every=1000,
        result_folder=None,
        log_dir="./log",
        steps=0,
        d_weight=0.1,
        per_weight=0.1,
    ):
        super().__init__()
        # kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        self.accelerator = Accelerator(
            kwargs_handlers=[kwargs],
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=grad_accum_steps,
            log_with="tensorboard",
            project_dir=log_dir,
        )

        self.vqvae = instantiate_from_config(model)

        dataset = instantiate_from_config(dataset)

        self.discr = NLayerDiscriminator(input_nc=3, ndf=64, n_layers=3)

        train_size = len(dataset) - valid_size
        self.train_ds, self.valid_ds = random_split(
            dataset,
            [train_size, valid_size],
            generator=torch.Generator().manual_seed(42),
        )
        print(f"train dataset size: {train_size}, valid dataset size: {valid_size}")

        self.train_dl = DataLoader(
            self.train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )
        self.valid_dl = DataLoader(
            self.valid_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=collate_fn,
        )

        self.g_optim = Adam(self.vqvae.parameters(), lr=lr, betas=(0.9, 0.99))
        self.d_optim = Adam(self.discr.parameters(), lr=lr, betas=(0.9, 0.99))
        self.g_sched = build_scheduler(
            self.g_optim,
            num_epoch,
            len(self.train_dl),
            lr_min,
            warmup_steps,
            warmup_lr_init,
            decay_steps,
        )
        self.d_sched = build_scheduler(
            self.d_optim,
            num_epoch,
            len(self.train_dl),
            lr_min,
            warmup_steps,
            warmup_lr_init,
            decay_steps,
        )

        self.per_loss = LPIPS(net="vgg").to(self.device).eval()
        for param in self.per_loss.parameters():
            param.requires_grad = False
        self.d_loss = hinge_d_loss
        self.g_loss = g_nonsaturating_loss
        self.d_weight = d_weight
        self.per_weight = per_weight

        self.num_epoch = num_epoch
        self.save_every = save_every
        self.samp_every = sample_every
        self.max_grad_norm = max_grad_norm

        self.model_saved_dir = os.path.join(result_folder, "models")
        os.makedirs(self.model_saved_dir, exist_ok=True)

        self.image_saved_dir = os.path.join(result_folder, "images")
        os.makedirs(self.image_saved_dir, exist_ok=True)

        (
            self.vqvae,
            self.discr,
            self.g_optim,
            self.d_optim,
            self.g_sched,
            self.d_sched,
            self.train_dl,
            self.valid_dl,
        ) = self.accelerator.prepare(
            self.vqvae,
            self.discr,
            self.g_optim,
            self.d_optim,
            self.g_sched,
            self.d_sched,
            self.train_dl,
            self.valid_dl,
        )
        self.accelerator.register_for_checkpointing(self.g_sched, self.d_sched)
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"number of learnable parameters: {n_parameters//1e6}M")

        # __import__("ipdb").set_trace()
        self.steps = steps
        if (model.params.ckpt_path is not None) & osp.exists(model.params.ckpt_path):
            # model.params.ckpt_path is something like 'path/to/models/step10/'
            self.loaded_steps = int(
                model.params.ckpt_path.split("step")[-1].split("/")[0]
            )
            path = model.params.ckpt_path
            self.accelerator.load_state(path)
        else:
            # NOTE: ckpt_path is None
            self.loaded_steps = -1

    @property
    def device(self):
        return self.accelerator.device

    def calculate_gradient_penalty(self, real_images, fake_images, lambda_term=10):

        eta = (
            torch.FloatTensor(real_images.shape[0], 1, 1, 1)
            .uniform_(0, 1)
            .to(self.device)
        )
        eta = eta.expand(
            real_images.shape[0],
            real_images.size(1),
            real_images.size(2),
            real_images.size(3),
        )

        interpolated = eta * real_images + ((1 - eta) * fake_images)
        interpolated = Variable(interpolated, requires_grad=True)
        return interpolated

    def calculate_loss_gradient_penalty(self, rec, img, lambda_term=10):

        bs = rec.size(0)
        assert bs == img.size(0)
        interpolated = self.calculate_gradient_penalty(
            img,
            rec,
        )
        images = torch.cat((rec, img, interpolated), dim=0)

        preds = self.discr(images)

        preds = rearrange(preds, "(b f) ... -> b f ...", b=3, f=bs)
        fakes, reals, prob_interpolated = preds[0], preds[1], preds[2]

        d_loss = hinge_d_loss(fakes, reals)

        gradients = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones(prob_interpolated.size()).to(rec),
            create_graph=True,
            retain_graph=True,
        )[0]

        grad_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_term
        return d_loss, grad_penalty

    def train(self, config_path=None):

        self.accelerator.init_trackers("vqgan")
        self.log = Log()
        self.total_steps = self.num_epoch * len(self.train_dl)
        for epoch in range(self.num_epoch):
            with tqdm(
                self.train_dl,
                dynamic_ncols=True,
                disable=not self.accelerator.is_main_process,
            ) as train_dl:
                # __import__("ipdb").set_trace()
                for batch in train_dl:
                    if isinstance(batch, tuple) or isinstance(batch, list):
                        img, targets = batch[0].tensors, batch[1]
                    else:
                        img = batch

                    self.steps += 1
                    if self.steps <= self.loaded_steps:
                        continue

                    # discriminator part
                    requires_grad(self.vqvae, False)
                    self.vqvae.eval()  # For nested dropout
                    requires_grad(self.discr, True)
                    with self.accelerator.accumulate(self.discr):
                        with self.accelerator.autocast():
                            rec, codebook_loss, _ = self.vqvae(
                                img,
                                targets,
                                ramp_step=self.steps,
                                ramp_length=self.total_steps,
                            )
                            d_loss, grad_penalty = self.calculate_loss_gradient_penalty(
                                rec, img
                            )
                            d_loss = d_loss + grad_penalty

                        self.accelerator.backward(d_loss)
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(
                                self.discr.parameters(), self.max_grad_norm
                            )
                        self.d_optim.step()
                        self.d_sched.step_update(self.steps)
                        self.d_optim.zero_grad()

                        self.log.update(
                            {
                                "d loss": d_loss.item(),
                                "d lr": self.d_optim.param_groups[0]["lr"],
                            }
                        )

                    # generator part
                    self.vqvae.train()
                    requires_grad(self.vqvae, True)
                    # requires_grad(self.vqvae.module.pixel_quantize, False)
                    # requires_grad(self.vqvae.module.pixel_decoder, False)
                    requires_grad(self.discr, False)
                    with self.accelerator.accumulate(self.vqvae):
                        with self.accelerator.autocast():
                            rec, codebook_loss, logvar = self.vqvae(
                                img,
                                targets,
                                ramp_step=self.steps,
                                ramp_length=self.total_steps,
                            )
                            # reconstruction loss
                            # rec_loss = F.l1_loss(rec, img) #+ F.mse_loss(rec, img) #NOTE: because I am using slotcon for training now
                            rec_loss = F.l1_loss(rec, img) / torch.exp(logvar) + logvar
                            # perception loss
                            per_loss = self.per_loss(rec, img).mean()
                            # gan loss
                            g_loss = self.g_loss(self.discr(rec))
                            # combine
                            loss = (
                                sum([v for _, v in codebook_loss.items()])
                                + rec_loss
                                + self.per_weight * per_loss
                                + self.d_weight * g_loss
                            )

                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(
                                self.vqvae.parameters(), self.max_grad_norm
                            )
                        self.g_optim.step()
                        self.g_sched.step_update(self.steps)
                        self.g_optim.zero_grad()

                    self.log.update(
                        {
                            "rec loss": rec_loss.item(),
                            "per loss": per_loss.item(),
                            "g loss": g_loss.item(),
                            "g lr": self.g_optim.param_groups[0]["lr"],
                        }
                    )

                    if not (self.steps % self.save_every):
                        self.save()

                    if not (self.steps % self.samp_every):
                        self.evaluate()

                    train_dl.set_postfix(
                        ordered_dict={
                            "epoch": epoch,
                            "reconstruct loss": self.log["rec loss"],
                            "perceptual loss": self.log["per loss"],
                            "g_loss": self.log["g loss"],
                            "d_loss": self.log["d loss"],
                            "g_lr": self.log["g lr"],
                        }
                    )
                    self.accelerator.log(
                        {
                            "reconstruct loss": self.log["rec loss"],
                            "perceptual loss": self.log["per loss"],
                            "g_loss": self.log["g loss"],
                            "d_loss": self.log["d loss"],
                            "g_lr": self.log["g lr"],
                            "d_lr": self.log["d lr"],
                        },
                        step=self.steps,
                    )

        self.accelerator.end_training()
        print("Train finished!")

    def unwrap_state_dict(self, item):
        state_dict = self.accelerator.unwrap_model(item).state_dict()
        return state_dict

    def save(self):
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(
            os.path.join(self.model_saved_dir, f"step{self.steps}")
        )

        # state_dict = self.accelerator.unwrap_model(self.vqvae).state_dict()
        # os.makedirs(os.path.join(self.model_saved_dir, f'step{self.steps}'), exist_ok=True)
        # self.accelerator.save(state_dict, os.path.join(self.model_saved_dir, f'step{self.steps}/vit_vq_step_{self.steps}.pt'))
        # discr = self.unwrap_state_dict(self.discr)
        # g_optim = self.unwrap_state_dict(self.g_optim)
        # d_optim = self.unwrap_state_dict(self.d_optim)
        # g_sched = self.unwrap_state_dict(self.g_sched)
        # d_sched = self.unwrap_state_dict(self.d_sched)
        # self.accelerator.save(discr, os.path.join(self.model_saved_dir, f'step{self.steps}/discr_step_{self.steps}.pt'))
        # self.accelerator.save(g_optim, os.path.join(self.model_saved_dir, f'step{self.steps}/g_optim_step_{self.steps}.pt'))
        # self.accelerator.save(d_optim, os.path.join(self.model_saved_dir, f'step{self.steps}/d_optim_step_{self.steps}.pt'))
        # self.accelerator.save(g_sched, os.path.join(self.model_saved_dir, f'step{self.steps}/g_sched_step_{self.steps}.pt'))
        # self.accelerator.save(d_sched, os.path.join(self.model_saved_dir, f'step{self.steps}/d_sched_step_{self.steps}.pt'))

    @torch.no_grad()
    def evaluate(self):
        self.vqvae.eval()
        with tqdm(
            self.valid_dl,
            dynamic_ncols=True,
            disable=not self.accelerator.is_local_main_process,
        ) as valid_dl:
            for i, batch in enumerate(valid_dl):
                if isinstance(batch, tuple) or isinstance(batch, list):
                    img, targets = batch[0].tensors, batch[1]
                else:
                    img = batch

                rec, _, _ = self.vqvae(
                    img, targets, ramp_step=self.steps, ramp_length=self.total_steps
                )
                imgs_and_recs = torch.stack((img, rec), dim=0)
                imgs_and_recs = rearrange(imgs_and_recs, "r b ... -> (b r) ...")
                imgs_and_recs = imgs_and_recs.detach().cpu().float()

                grid = make_grid(
                    imgs_and_recs, nrow=6, normalize=True, value_range=(-1, 1)
                )
                save_image(
                    grid,
                    os.path.join(self.image_saved_dir, f"step_{self.steps}_{i}.png"),
                )
        self.vqvae.train()


class DiffusionTrainer(nn.Module):
    def __init__(
        self,
        model,
        dataset,
        num_epoch,
        valid_size=32,
        lr=1e-4,
        lr_min=5e-5,
        warmup_steps=50000,
        warmup_lr_init=1e-6,
        decay_steps=None,
        batch_size=32,
        num_workers=0,
        pin_memory=False,
        max_grad_norm=1.0,
        grad_accum_steps=1,
        mixed_precision="bf16",
        save_every=10000,
        sample_every=1000,
        result_folder=None,
        log_dir="./log",
        steps=0,
        d_weight=0.1,
        per_weight=0.1,
        eval_fid=False,
        fid_stats=None,
        use_multi_epochs_dataloader=False,
        compile=False,
        overfit=False,
        making_cache=False,
        cache_mode=False,
        latent_cache_file=None,
    ):
        super().__init__()
        # kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        self.accelerator = Accelerator(
            kwargs_handlers=[kwargs],
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=grad_accum_steps,
            log_with="tensorboard",
            project_dir=log_dir,
        )

        self.vqvae = instantiate_from_config(model)

        dataset = instantiate_from_config(dataset)

        train_size = len(dataset) - valid_size
        self.train_ds, self.valid_ds = random_split(
            dataset,
            [train_size, valid_size],
            generator=torch.Generator().manual_seed(42),
        )
        if overfit:
            from torch.utils.data import Subset
            self.valid_ds = Subset(self.train_ds, range(valid_size))
        print(f"train dataset size: {train_size}, valid dataset size: {valid_size}")
        if cache_mode:
            my_collate_fn = collate_fn_cache
        else:
            my_collate_fn = collate_fn

        if not use_multi_epochs_dataloader:
            self.train_dl = DataLoader(
                self.train_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=my_collate_fn,
            )
        else:
            self.accelerator.even_batches = False
            sampler = DistributedSampler(
                self.train_ds,
                num_replicas=self.accelerator.num_processes,
                rank=self.accelerator.process_index,
                shuffle=True,
            )
            self.train_dl = MultiEpochsDataLoader(
                self.train_ds,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=pin_memory,
                collate_fn=my_collate_fn,
            )
        self.valid_dl = DataLoader(
            self.valid_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=my_collate_fn,
        )

        # self.g_optim = Adam(self.vqvae.parameters(), lr=lr, betas=(0.9, 0.99))
        # only update require_grad = True
        effective_bs = batch_size * grad_accum_steps * self.accelerator.num_processes
        self.g_optim = Adam(
            filter(lambda p: p.requires_grad, self.vqvae.parameters()),
            lr=lr * effective_bs / 512,
            betas=(0.9, 0.99),
        )
        self.g_sched = build_scheduler(
            self.g_optim,
            num_epoch,
            len(self.train_dl),
            lr_min,
            warmup_steps,
            warmup_lr_init,
            decay_steps,
        )
        if not use_multi_epochs_dataloader:
            (self.vqvae, self.g_optim, self.g_sched, self.train_dl, self.valid_dl) = (
                self.accelerator.prepare(
                    self.vqvae, self.g_optim, self.g_sched, self.train_dl, self.valid_dl
                )
            )
        else:
            (
                self.vqvae,
                self.g_optim,
                self.g_sched,
            ) = self.accelerator.prepare(
                self.vqvae,
                self.g_optim,
                self.g_sched,
            )

        self.num_epoch = num_epoch
        self.save_every = save_every
        self.samp_every = sample_every
        self.max_grad_norm = max_grad_norm

        self.result_folder = result_folder
        self.model_saved_dir = os.path.join(result_folder, "models")
        os.makedirs(self.model_saved_dir, exist_ok=True)

        self.image_saved_dir = os.path.join(result_folder, "images")
        os.makedirs(self.image_saved_dir, exist_ok=True)

        self.accelerator.register_for_checkpointing(self.g_sched)
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"number of learnable parameters: {n_parameters//1e6}M")

        self.cache_mode = cache_mode

        # __import__("ipdb").set_trace()
        self.steps = steps
        if (model.params.ckpt_path is not None) & osp.exists(model.params.ckpt_path):
            # model.params.ckpt_path is something like 'path/to/models/step10/'
            self.loaded_steps = int(
                model.params.ckpt_path.split("step")[-1].split("/")[0]
            )
            path = model.params.ckpt_path
            self.accelerator.load_state(path)
        else:
            # NOTE: ckpt_path is None
            self.loaded_steps = -1
        self.eval_fid = eval_fid
        if eval_fid:
            assert fid_stats is not None
        self.fid_stats = fid_stats

        self.use_vq = model.params.use_vq
        self.vq_beta = model.params.code_beta

        # compile after state_dict is loaded
        if compile:
            for idx, _model in enumerate(self.accelerator._models):
                if hasattr(_model, "dit"):
                    _model.dit = torch.compile(
                        _model.dit, mode="reduce-overhead", fullgraph=False
                    )
                if hasattr(_model, "encoder"):
                    _model.encoder = torch.compile(
                        _model.encoder, mode="reduce-overhead", fullgraph=False
                    )
                # if hasattr(_model, 'vae'):
                #     _model.vae = torch.compile(_model.vae, mode="reduce-overhead", fullgraph=False)
                self.accelerator._models[idx] = _model

    @property
    def device(self):
        return self.accelerator.device

    def train(self, config_path=None):
        if config_path is not None:
            # copy the config and save
            import shutil

            shutil.copy(config_path, self.result_folder)

        self.accelerator.init_trackers("vqgan")
        self.log = Log()
        for epoch in range(self.num_epoch):
            if ((epoch + 1) * len(self.train_dl)) <= self.loaded_steps:
                print(f"Epoch {epoch} is skipped because it is loaded from ckpt")
                self.steps += len(self.train_dl)
                continue
            with tqdm(
                self.train_dl,
                dynamic_ncols=True,
                disable=not self.accelerator.is_main_process,
            ) as train_dl:
                for batch in train_dl:
                    if isinstance(batch, tuple) or isinstance(batch, list):
                        if self.cache_mode:
                            # import ipdb; ipdb.set_trace()
                            img, latent, targets = batch[0].tensors, batch[1].tensors, batch[2]
                        else:
                            latent = None
                            img, targets = batch[0].tensors, batch[1]
                    else:
                        img = batch
                        latent = None

                    self.steps += 1
                    if self.steps <= self.loaded_steps:
                        print(
                            f"Step {self.steps} is skipped because it is loaded from ckpt"
                        )
                        continue

                    # generator part
                    # now diffuseslot will handle what get updated
                    # requires_grad(self.vqvae, True)
                    with self.accelerator.accumulate(self.vqvae):
                        with self.accelerator.autocast():
                            if self.steps == 1:
                                print(f"Training batch size: {img.size(0)}")
                                print(f"Hello from index {self.accelerator.local_process_index}")
                            codebook_loss = self.vqvae(img, targets, latents=latent, epoch=epoch)
                            # combine
                            loss = sum([v for _, v in codebook_loss.items()])
                            diff_loss = codebook_loss["diff_loss"]
                            if self.use_vq:
                                vq_loss = codebook_loss["vq_loss"]

                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(
                                self.vqvae.parameters(), self.max_grad_norm
                            )
                        self.g_optim.step()
                        self.g_sched.step_update(self.steps)
                        self.g_optim.zero_grad()

                    if self.use_vq:
                        self.log.update(
                            {
                                "diff loss": diff_loss.item(),
                                "vq loss": vq_loss.item() / self.vq_beta,
                                "g lr": self.g_optim.param_groups[0]["lr"],
                            }
                        )
                    else:
                        self.log.update(
                            {
                                "diff loss": diff_loss.item(),
                                "g lr": self.g_optim.param_groups[0]["lr"],
                            } if 'kl_loss' not in codebook_loss else {
                                "diff loss": diff_loss.item(),
                                "g lr": self.g_optim.param_groups[0]["lr"],
                                "kl loss": codebook_loss["kl_loss"].item()
                            }
                        )

                    if not (self.steps % self.save_every):
                        self.save()

                    if not (self.steps % self.samp_every):
                        self.evaluate()

                    postfix_dict = {
                        "epoch": epoch,
                        "diffuse loss": self.log["diff loss"],
                        "g_lr": self.log["g lr"],
                    } if "kl_loss" not in codebook_loss else {
                        "epoch": epoch,
                        "diffuse loss": self.log["diff loss"],
                        "g_lr": self.log["g lr"], 
                        "kl_loss": codebook_loss["kl_loss"].item()
                    }
                    if self.use_vq:
                        postfix_dict.update({"vq loss": self.log["vq loss"]})
                    train_dl.set_postfix(ordered_dict=postfix_dict)
                    self.accelerator.log(postfix_dict, step=self.steps)

        self.accelerator.end_training()
        self.save()
        print("Train finished!")

    def save(self):
        self.accelerator.wait_for_everyone()
        self.accelerator.save_state(
            os.path.join(self.model_saved_dir, f"step{self.steps}")
        )

    @torch.no_grad()
    def evaluate(self, return_metrics=False):
        self.vqvae.eval()
        with tqdm(
            self.valid_dl,
            dynamic_ncols=True,
            disable=not self.accelerator.is_local_main_process,
        ) as valid_dl:
            for batch_i, batch in enumerate(valid_dl):
                if isinstance(batch, tuple) or isinstance(batch, list):
                    img, targets = batch[0].tensors, batch[1]
                else:
                    img = batch

                rec = self.vqvae(img, targets, sample=True)

                imgs_and_recs = torch.stack((img.to(rec.device), rec), dim=0)
                imgs_and_recs = rearrange(imgs_and_recs, "r b ... -> (b r) ...")
                imgs_and_recs = imgs_and_recs.detach().cpu().float()

                grid = make_grid(
                    imgs_and_recs, nrow=6, normalize=True, value_range=(0, 1)
                )
                save_image(
                    grid,
                    os.path.join(
                        self.image_saved_dir, f"step_{self.steps}_{batch_i}.png"
                    ),
                )

                # __import__("ipdb").set_trace()
                if self.eval_fid:
                    if not os.path.exists(
                        os.path.join(self.image_saved_dir, f"real_step{self.steps}")
                    ):
                        os.mkdir(
                            os.path.join(self.image_saved_dir, f"real_step{self.steps}")
                        )
                        os.mkdir(
                            os.path.join(self.image_saved_dir, f"rec_step{self.steps}")
                        )

                    def norm_ip(img, low, high):
                        img.clamp_(min=low, max=high)
                        img.sub_(low).div_(max(high - low, 1e-5))

                    def norm_range(t, value_range):
                        if value_range is not None:
                            norm_ip(t, value_range[0], value_range[1])
                        else:
                            norm_ip(t, float(t.min()), float(t.max()))

                    for i in range(img.size(0)):
                        norm_range(img[i], (0, 1))
                        norm_range(rec[i], (0, 1))
                    for j in range(img.size(0)):
                        save_image(
                            img[j],
                            os.path.join(
                                self.image_saved_dir,
                                f"real_step{self.steps}",
                                f"step_{self.steps}_{batch_i}_{j}_real.png",
                            ),
                        )
                        save_image(
                            rec[j],
                            os.path.join(
                                self.image_saved_dir,
                                f"rec_step{self.steps}",
                                f"step_{self.steps}_{batch_i}_{j}_rec.png",
                            ),
                        )
        if self.eval_fid:
            metrics_dict = torch_fidelity.calculate_metrics(
                input1=os.path.join(self.image_saved_dir, f"real_step{self.steps}"),
                input2=os.path.join(self.image_saved_dir, f"rec_step{self.steps}"),
                fid_statistics_file=self.fid_stats,
                cuda=True,
                isc=True,
                fid=True,
                kid=False,
                prc=False,
                verbose=False,
            )
            if return_metrics:
                return metrics_dict
            fid = metrics_dict["frechet_inception_distance"]
            inception_score = metrics_dict["inception_score_mean"]

            self.log.update({"fid": fid, "inception_score": inception_score})
            self.accelerator.log(
                {
                    "fid": self.log["fid"],
                    "inception_score": self.log["inception_score"],
                },
                step=self.steps,
            )
        self.vqvae.train()


def masked_p_generator():
    p = np.cos(0.5 * np.pi * np.random.rand(1))
    return p.item()


class PaintMindTrainer(nn.Module):
    def __init__(
        self,
        model,
        dataset,
        num_epoch,
        valid_size=10,
        optim="lion",  # or 'adamw'
        lr=6e-5,
        lr_min=1e-5,
        warmup_steps=5000,
        warmup_lr_init=1e-6,
        decay_steps=80000,
        weight_decay=0.05,
        batch_size=32,
        num_workers=0,
        pin_memory=False,
        grad_accum_steps=1,
        mixed_precision="fp16",
        max_grad_norm=1.0,
        save_every=10000,
        sample_every=1000,
        result_folder=None,
        log_dir="./log",
    ):
        super().__init__()
        # kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        self.accelerator = Accelerator(
            kwargs_handlers=[kwargs],
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=grad_accum_steps,
            log_with="tensorboard",
            logging_dir=log_dir,
        )

        train_size = len(dataset) - valid_size
        self.train_ds, self.valid_ds = random_split(
            dataset,
            [train_size, valid_size],
            generator=torch.Generator().manual_seed(42),
        )

        self.train_dl = DataLoader(
            self.train_ds,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        self.valid_dl = DataLoader(
            self.valid_ds,
            batch_size=6,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

        self.model = model

        if optim == "lion":
            self.optim = Lion(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=lr,
                weight_decay=weight_decay,
            )
        elif optim == "adamw":
            self.optim = AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=lr,
                betas=(0.9, 0.96),
                weight_decay=weight_decay,
            )
        else:
            raise NotImplementedError

        self.scheduler = build_scheduler(
            self.optim,
            num_epoch,
            len(self.train_dl),
            lr_min,
            warmup_steps,
            warmup_lr_init,
            decay_steps,
        )

        (
            self.model,
            self.optim,
            self.scheduler,
            self.train_dl,
            self.valid_dl,
        ) = self.accelerator.prepare(
            self.model, self.optim, self.scheduler, self.train_dl, self.valid_dl
        )

        self.num_epoch = num_epoch
        self.save_every = save_every
        self.sample_every = sample_every
        self.max_grad_norm = max_grad_norm

        self.model_saved_dir = os.path.join(result_folder, "models")
        os.makedirs(self.model_saved_dir, exist_ok=True)

        self.image_saved_dir = os.path.join(result_folder, "images")
        os.makedirs(self.image_saved_dir, exist_ok=True)

        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"number of learnable parameters: {n_parameters//1e6}M")
        print(f"train dataset size: {train_size}, valid dataset size: {valid_size}")

    def save(self):
        self.accelerator.wait_for_everyone()
        state_dict = self.accelerator.unwrap_model(self.model).state_dict()
        self.accelerator.save(
            state_dict,
            os.path.join(self.model_saved_dir, f"paintmind_step_{self.steps}.pt"),
        )

    def train(self):
        self.steps = 0
        self.cfg_p = 0.1
        self.accelerator.init_trackers("paintmind")
        self.log = Log()
        for epoch in range(self.num_epoch):
            with tqdm(
                self.train_dl,
                dynamic_ncols=True,
                disable=not self.accelerator.is_main_process,
            ) as train_dl:
                for batch in train_dl:
                    with self.accelerator.accumulate(self.model):
                        imgs, text = batch
                        if random.random() < self.cfg_p:
                            text = None

                        with self.accelerator.autocast():
                            loss = self.model(
                                imgs, text, mask_ratio=masked_p_generator()
                            )

                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(
                                self.model.parameters(), self.max_grad_norm
                            )
                        self.optim.step()
                        self.scheduler.step_update(self.steps)
                        self.optim.zero_grad()

                    self.steps += 1
                    self.log.update(
                        {"loss": loss.item(), "lr": self.optim.param_groups[0]["lr"]}
                    )

                    if not (self.steps % self.sample_every):
                        self.evaluate()

                    if not (self.steps % self.save_every):
                        self.save()

                    train_dl.set_postfix(
                        ordered_dict={
                            "Epoch": epoch,
                            "Loss": self.log["loss"],
                            "LR": self.log["lr"],
                        }
                    )
                    self.accelerator.log(
                        {"loss": self.log["loss"], "lr": self.log["lr"]},
                        step=self.steps,
                    )

        self.accelerator.end_training()
        print("Train finished!")

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        with tqdm(
            self.valid_dl,
            dynamic_ncols=True,
            disable=not self.accelerator.is_main_process,
        ) as valid_dl:
            for i, batch in enumerate(valid_dl):
                imgs, text = batch

                with self.accelerator.autocast():
                    gens = self.model.generate(
                        text=text,
                        timesteps=18,
                        temperature=1.0,
                        topk=5,
                        save_interval=2,
                    )

                imgs_and_gens = [imgs.cpu()] + gens
                imgs_and_gens = torch.cat(imgs_and_gens, dim=0)
                imgs_and_gens = imgs_and_gens.detach().cpu().float()

                grid = make_grid(
                    imgs_and_gens, nrow=6, normalize=True, value_range=(-1, 1)
                )
                save_image(
                    grid,
                    os.path.join(self.image_saved_dir, f"step_{self.steps}_{i}.png"),
                )
        self.model.train()
