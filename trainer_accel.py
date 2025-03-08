import torch
import torch.nn as nn
import torch.nn.functional as F

from model import DiT
from cfm import CFM

import gc
import os
import copy
import wandb
from tqdm import tqdm
import numpy as np

import torch.distributed as dist
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import DataLoader

from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from PIL import Image

from contextlib import nullcontext

from transformers import (
    get_cosine_schedule_with_warmup,
    get_constant_schedule_with_warmup,
)

from ema_pytorch import EMA
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs



def get_scheduler(optimizer, warmup_steps, total_steps, cosine=False):
    if cosine:
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
            num_cycles=0.5,  # standard half cycle cosine
        )
    else:
        scheduler = get_constant_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps
        )

    return scheduler


class CifarConfig:
    train_images = 50000
    val_images = 10000
    root = "/media/newhd/ayan/diffusion_exps/data"
    # DIT config
    num_classes = 10
    in_channels = 3
    input_size = 32
    patch_size = 2
    dim = 384
    depth = 12
    num_heads = 6
    hidden_scale=4
    class_dropout_prob = 0.1
    cfg_scale = [1.0, 2.0, 5.0]
    

    # Trainer config
    total_steps = 250000
    batch_size = 64
    gradient_accumulation_steps = 1
    max_grad_norm=1.0
    # ema kwargs
    ema_update_after_step = 100
    ema_update_every = 10
    ema_decay = 0.99

    warmup_steps = 1000
    positional_encoding = "rope"
    use_cosine_scheduler = False

    # optimizer config
    lr = 1e-4

    # save_every = 5
    # log_every = 4
    save_every = 10000
    log_every = 1000
    use_wandb = True
    wandb_project = "dit-cifar1"
    savedir = "cifar_models/checkpoints"
    sampledir = "cifar_models/samples"

def get_data(config, generator):
    dataset = datasets.CIFAR10(
        root=config.root,
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ]
        ),
    )
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        generator=generator
    )
        
    def cycle(iterable):
        while True:
            for x in iterable:
                yield x

    train_loader = cycle(train_loader)
    return train_loader


class Trainer:

    def __init__(self, config):
        self.config = config
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        self.logger = "wandb"
        self.accelerator = Accelerator(
            log_with=self.logger,
            kwargs_handlers=[ddp_kwargs],
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,

        )

        wandb_kwargs = {"wandb": {"resume": 'auto'}}
        self.accelerator.init_trackers(
            project_name=self.config.wandb_project,
            init_kwargs=wandb_kwargs,
            config=config
        )

        # initialize models
        dit = DiT(
            input_size=self.config.input_size,
            patch_size=self.config.patch_size,
            in_channels=self.config.in_channels,
            dim=self.config.dim,
            depth=self.config.depth,
            num_heads=self.config.num_heads,
            num_classes=self.config.num_classes,
            class_dropout_prob=self.config.class_dropout_prob,
            pos_encoding=self.config.positional_encoding,
            hidden_scale=self.config.hidden_scale,
        )
        
        self.device = self.accelerator.device
        self.model = CFM(model=dit, device=self.device)
        self.max_grad_norm = self.config.max_grad_norm

        if self.is_main:
            self.ema_model = EMA(
                self.model,
                include_online_model=False,
                beta=self.config.ema_decay,
                update_after_step=self.config.ema_update_after_step,
                update_every=self.config.ema_update_every,
            )
            self.ema_model.to(self.device)

        self.optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            eps=1e-8,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )
        self.scheduler = get_scheduler(
            self.optim,
            warmup_steps=self.config.warmup_steps,
            total_steps=self.config.total_steps,
            cosine=self.config.use_cosine_scheduler,
        )

        self.save_dir = self.config.savedir
        os.makedirs(self.save_dir, exist_ok=True)
        self.sample_dir = self.config.sampledir 
        os.makedirs(self.sample_dir, exist_ok=True)

        self.model, self.optim = self.accelerator.prepare(
            self.model, self.optim
        )

        total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        print(f"Total parameters: {total_params:.2f}M")


    @property
    def is_main(self):
        return self.accelerator.is_main_process

    def save_checkpoint(self, step):
        self.accelerator.wait_for_everyone()
        if self.is_main:
            save_path = os.path.join(self.save_dir, f"model_{step}.pt")
            self.accelerator.save(
                {
                    "model": self.accelerator.unwrap_model(self.model).state_dict(), 
                    "optim": self.accelerator.unwrap_model(self.optim).state_dict(),
                    "ema": self.ema_model.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "step": step,
                },
                save_path,
            )
            print(f"Saved checkpoint at step {step}")
    

    def load_checkpoint(self):
        self.accelerator.wait_for_everyone()
        checkpoints = os.listdir(self.save_dir)
        if len(checkpoints) == 0:
            print("No checkpoints found")
            return 0
        checkpoint = sorted(checkpoints)[-1]
        print("Loading checkpoint: ", checkpoint)
        checkpoint = torch.load(os.path.join(self.save_dir, checkpoint), weights_only=True, map_location="cpu")

        self.accelerator.unwrap_model(self.model).load_state_dict(checkpoint["model"])
        self.accelerator.unwrap_model(self.optim).load_state_dict(checkpoint["optim"])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        if self.is_main:
            self.ema_model.load_state_dict(checkpoint['ema'])

        step = checkpoint['step']

        del checkpoint
        gc.collect()
        return step

    def sample_and_log_imgs(self, step):
        log_imgs = []
        num_classes = self.config.num_classes
        c = torch.arange(num_classes)  # Create tensor ranging from 0-9 (all classes)
        
        for cfg_scale in self.config.cfg_scale:
            print(f"Sampling images at step {step} with cfg_scale {cfg_scale}...")
            z = self.accelerator.unwrap_model(self.model).sample(num_classes, c=c, steps=50, cfg_scale=cfg_scale)  # Batch sampling
            log_imgs.append((cfg_scale, z))
        
        wandb_imgs = []
        for cfg_scale, images in log_imgs:
            images = images.permute(0, 2, 3, 1).cpu().numpy()
            images = (images * 255).astype("uint8")
            
            # Create 2x5 grid
            grid_img = np.vstack([
                np.hstack(images[:5]),
                np.hstack(images[5:])
            ])
            
            img_pil = Image.fromarray(grid_img)
            img_path = os.path.join(self.sample_dir, f"cfg_{cfg_scale}_{step}.png")
            img_pil.save(img_path)
            
            wandb_imgs.append(wandb.Image(img_path, caption=f"cfg_{cfg_scale}, step_{step}"))

        wandb_tracker = self.accelerator.get_tracker("wandb")
        wandb_tracker.log({"Images": wandb_imgs})


    def train(self,):
        self.model.train()
        generator = torch.Generator()
        generator.manual_seed(1331)
        train_loader = get_data(self.config, generator)
        total_steps = self.config.total_steps
        train_loader, self.scheduler = self.accelerator.prepare(train_loader, self.scheduler)
        start_step = self.load_checkpoint()
        global_step = start_step

        self.model.train()
        pbar = tqdm(
            train_loader,
            initial=start_step,
            total=total_steps,
            desc=f"Training",
            unit="step",
            disable=not self.accelerator.is_local_main_process
        )

        for batch in pbar:
            with self.accelerator.accumulate(self.model):
                x1 = batch[0]
                c = batch[1]

                self.optim.zero_grad()
                loss, _ = self.model(x1, c)
                self.accelerator.backward(loss)

                if self.max_grad_norm > 0 and self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.optim.step()
                self.scheduler.step()

            if self.is_main and self.accelerator.sync_gradients:
                self.ema_model.update()

            global_step +=1

            if self.accelerator.is_local_main_process:
                self.accelerator.log({"loss": loss.item(), "lr": self.scheduler.get_last_lr()[0]} )

            pbar.set_postfix(step=str(global_step), loss=f"{loss.item()}")

            if global_step % self.config.save_every * self.config.gradient_accumulation_steps == 0:
                self.save_checkpoint(global_step)

            if global_step % self.config.log_every == 0 and self.accelerator.is_local_main_process:
                self.sample_and_log_imgs(global_step)

        
        self.save_checkpoint(global_step)
        self.accelerator.end_training()





if __name__ == "__main__":
    config = CifarConfig()
    trainer = Trainer(config)
    trainer.train()
    # runs = wandb.Api().runs("ayankashayp/dit-cifar")
    # for run in runs:
    #     run.delete()
