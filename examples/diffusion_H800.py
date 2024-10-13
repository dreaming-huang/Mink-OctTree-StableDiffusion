import gc
import os
import sys
# sys.path.append('/mnt/data4/hjm_code_4/MinkowskiEngine-master')
# from memory_profiler import profile
import subprocess
import argparse
import logging
import glob
from typing import Optional
# from matplotlib import pyplot as plt
from matplotlib import pyplot as plt
import numpy as np
from time import time
import urllib

# Must be imported before large libs
try:
    import open3d as o3d
except ImportError:
    raise ImportError("Please install open3d with `pip install open3d`.")
try:
    from pytorch_lightning.core import LightningModule
    from pytorch_lightning import Trainer,seed_everything
    import pytorch_lightning as pl
except ImportError:
    raise ImportError(
        "Please install requirements with `pip install open3d pytorch_lightning`."
    )

import torch
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from torch.optim import SGD
import MinkowskiEngine as ME
from torch.utils.data import Dataset, DataLoader,ConcatDataset
from examples.reconstruction import InfSampler, resample_mesh
from MinkowskiEngine.modules.vae_block import ResNetBlock,ResNet2,ResNet3
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import diffusers
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from examples.denoise_unet import UNet

from examples.ae_res_v100 import rotate_point_cloud,VaeModule,VAE,ModelNet40Dataset, PointCloud, ShapeNetDataset, save_png
from diffusers.models.embeddings import TimestepEmbedding, Timesteps

def collate_pointcloud_fn(list_data):
    list_data=list(filter(lambda x: x is not None,list_data))
    sorted_list = sorted(list_data, key=lambda x: len(x[0]))
    
    coords, feats, labels = list(zip(*sorted_list))
    while sum(len(item) for item in coords) > config.max_batch_len:
        print(
            f"this batch is too big, some data will be dropped, sum: {sum(len(item) for item in coords)}"
        )
        sorted_list = sorted_list[1:]
        coords, feats, labels = list(zip(*sorted_list))
    # Concatenate all lists
    return {
        "coords": ME.utils.batched_coordinates(coords),
        "xyzs": [torch.from_numpy(feat).float() for feat in feats],
        "labels": torch.LongTensor(labels),
    }

class diffusionModule(LightningModule):
    def __init__(self,vae,unet,config,time_embedding_type="positional",flip_sin_to_cos=True,freq_shift: int = 0,num_train_timesteps: Optional[int] = None):
        super().__init__()
        for name, value in vars().items():
            if name != "self":
                setattr(self, name, value)
        self.config = config
        self.noise_scheduler = DDPMScheduler(num_train_timesteps=config.ddpm_num_steps, beta_schedule=config.ddpm_beta_schedule,)
        if time_embedding_type == "positional":
            self.time_proj = Timesteps(320, flip_sin_to_cos, freq_shift)
            timestep_input_dim = 320
        elif time_embedding_type == "learned":
            pass

        time_embed_dim=320*4
        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

            # self.time_proj = nn.Embedding(num_train_timesteps, 320)
            # timestep_input_dim = 320
    def predict_noise(self, sin,temb):
        return self.unet(sin,temb)
    def training_step(self,batch, batch_idx):
        gc.collect()

        sin = ME.SparseTensor(
            features=torch.ones(len(batch["coords"]), 1).to(batch["coords"].device),
            coordinates=batch["coords"].int(),
        )
        # print(sin)
        with torch.no_grad():
            clean,_=self.vae.encoder(sin)
        coords, features = clean.C,clean.F
        noise=torch.rand_like(features,device=features.device)
        batch_indices = coords[:, 0]
        unique_batch_indices = torch.unique(batch_indices)
        bsz=unique_batch_indices.shape[0]
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (bsz,), device=features.device
        ).long()

        t_emb=self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb) #b,1280
        for batch_idx in unique_batch_indices:
            batch_mask = (batch_indices == batch_idx)
            features[batch_mask] = self.noise_scheduler.add_noise(features[batch_mask], noise[batch_mask], timesteps[batch_idx])

        
        noise_clean=ME.SparseTensor(
            features=features,
            # coordinates=coords,
            coordinate_map_key=clean.coordinate_map_key,
            tensor_stride=clean.tensor_stride,
            coordinate_manager=clean.coordinate_manager,
            )
        loss=0
        model_output = self.predict_noise(noise_clean,emb)
        if self.config.prediction_type == "epsilon":
            # list_of_featurs = stensor.decomposed_features
            # list_of_permutations = stensor.decomposition_permutations
            # list_of_decomposed_labels = [labels[inds] for inds in list_of_permutations]
            # loss += torch.functional.mse_loss(curr_feats, curr_labels)
            loss=(model_output.F-noise).pow(2).mean() #不一定对，ME坐标可能会变，后续看看
        elif self.config.prediction_type == "sample":
            pass
            # TODO 现在是diffusers的实现，后续修改
            # alpha_t = _extract_into_tensor(
            #     noise_scheduler.alphas_cumprod, timesteps, (clean_images.shape[0], 1, 1, 1)
            # )
            # snr_weights = alpha_t / (1 - alpha_t)
            # # use SNR weighting from distillation paper
            # loss = snr_weights * F.mse_loss(model_output.float(), clean_images.float(), reduction="none")
            # loss = loss.mean()
        # print(loss)
        self.log('train_loss',loss.detach().item(),batch_size=self.config.batch_size, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self,batch, batch_idx):
        if batch_idx!=0 or str(self.device) != 'cuda:0':
            return

        gc.collect()
        sin = ME.SparseTensor(
            features=torch.ones(len(batch["coords"]), 1).to(batch["coords"].device),
            coordinates=batch["coords"].int(),
        )

        clean,_=self.vae.encoder(sin)
        coords, features = clean.C,clean.F
        noise=torch.rand_like(features)
        batch_indices = coords[:, 0]
        unique_batch_indices = torch.unique(batch_indices)
        bsz=unique_batch_indices.shape[0]

        features=self.noise_scheduler.add_noise(features, noise, torch.tensor(self.config.ddpm_num_steps-1))
        noise_clean=ME.SparseTensor(
            features=features,
            coordinate_map_key=clean.coordinate_map_key,
            tensor_stride=clean.tensor_stride,
            coordinate_manager=clean.coordinate_manager,
            )
        self.noise_scheduler.set_timesteps(50)
        for t in self.noise_scheduler.timesteps:
            tt=torch.zeros(bsz,device=clean.device).long()+t
            t_emb=self.time_proj(tt)
            t_emb = t_emb.to(dtype=self.dtype)
            emb = self.time_embedding(t_emb) #b,1280
            model_output = self.predict_noise(noise_clean,emb)
            noise_clean=ME.SparseTensor(
                features=self.noise_scheduler.step(model_output.F,tt[0],noise_clean.F,return_dict=False)[0],
                coordinate_map_key=clean.coordinate_map_key,
                tensor_stride=clean.tensor_stride,
                coordinate_manager=clean.coordinate_manager,
                )
        target_key = sin.coordinate_map_key
        _,_,sout=self.vae.decoder(noise_clean,target_key)

        if batch_idx==0 and str(self.device) == 'cuda:0':
            fig = plt.figure(figsize=(20, 20))
            batch_coords, batch_feats = sout.decomposed_coordinates_and_features
            real_batch_coords,_=sin.decomposed_coordinates_and_features
            for b, (coords, real_coords) in enumerate(zip(batch_coords, real_batch_coords)):
                if b==4:
                    break
                # print(len(coords))
                pcd = PointCloud(coords.cpu())
                pcd.estimate_normals()
                pcd.translate([0.6 * self.config.resolution, 0, 0])
                # pcd.rotate(M)
                opcd = PointCloud(real_coords.cpu())
                opcd.translate([-0.6 * self.config.resolution, 0, 0])
                opcd.estimate_normals()
                # opcd.rotate(M)
                # save_png(pcd,opcd,os.path.join(config.floder,f'{i:09}_{b}.png'))
                ax=fig.add_subplot(2,2,b+1,projection='3d')
                save_png(pcd,opcd,ax)
                plt.axis('equal')
                plt.tight_layout()
                plt.savefig(os.path.join(self.config.floder,f'epoch_{self.current_epoch:04}.png'))

    

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.unet.parameters(),
            lr=self.config.lr,
        )
        return optimizer
    
    def train_dataloader(self):
        args = {
        "batch_size": self.config.batch_size,
        "num_workers": self.config.num_workers,
        "collate_fn": collate_pointcloud_fn,
        "pin_memory": False,
        "drop_last": False,
        "shuffle": True,
        }
        return DataLoader(
            train_dataset,
            **args
        )
    def val_dataloader(self):
        args = {
        "batch_size": self.config.batch_size,
        "num_workers": self.config.num_workers,
        "collate_fn": collate_pointcloud_fn,
        "pin_memory": False,
        "drop_last": False,
        }
        return DataLoader(
            val_dataset,
            **args
        )






if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--max_epochs", type=int, default=1000, help="Max epochs")
    pa.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    pa.add_argument("--batch_size", type=int, default=1, help="batch size per GPU")
    pa.add_argument("--ngpus", type=int, default=1, help="num_gpus")
    pa.add_argument("--num_workers", type=int, default=32, help="num_workers")
    pa.add_argument("--resolution", type=int, default=256, help="resolution")
    pa.add_argument("--window_size", type=int, default=50, help="window_size")
    pa.add_argument("--floder", type=str, default="diffusion")
    pa.add_argument("--max_batch_len", type=int, default=2000000,help="relate to resolution")
    pa.add_argument("--recover",type=str,default=None,help="recover from checkpoint")
    pa.add_argument("--seed",type=int,default=42,help="seed")
    pa.add_argument("--save_every",type=int,default=20,help="save after n epoch")
    pa.add_argument("--cache",type=bool,default=False,help="whether cache pointcloud.npy")
    pa.add_argument("--ddpm_num_steps", type=int, default=1000, help="num steps")
    pa.add_argument("--ddpm_beta_schedule", type=str, default="linear", help="beta schedule. linear, scaled_linear, or squaredcos_cap_v2")
    pa.add_argument("--prediction_type", type=str, default="epsilon", help="prediction type.can be `epsilon` (predicts the noise of the diffusion process),`sample` (directly predicts the noisy sample`)")




    config = pa.parse_args()
    os.makedirs(config.floder,exist_ok=True)
    seed_everything(config.seed,workers=True)
    torch.cuda.manual_seed_all(config.seed)
    train_dataset=ConcatDataset([
            ModelNet40Dataset(
                phase="train",
                transform=rotate_point_cloud,
                config=config),
            ShapeNetDataset(
                transform=rotate_point_cloud,
                config=config),
            ])
    val_dataset=ModelNet40Dataset(
            phase="test",
            config=config)
    
    num_devices = min(config.ngpus, torch.cuda.device_count())
    print(f"Testing {num_devices} GPUs.")
    unet=UNet([4,320, 640, 1280],config,embedding_dim=1280)
    if config.ngpus > 1:
        unet = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(unet)
    vae=VAE(config)
    checkpoint = torch.load("/root/hjm/octVae/checkpoints/vae-epoch=0119-val_loss_epoch=0.05.ckpt",map_location='cpu')

    vae.load_state_dict({k[6:]: v for k, v in checkpoint["state_dict"].items()})
    # print({k: v for k, v in checkpoint["state_dict"].items()if k.startswith("model.encoder.")})

    diffusion = diffusionModule(vae,unet,config)

    trainer = Trainer(
        max_epochs=config.max_epochs,
        # precision="16-mixed",
        # precision='bf16-mixed',
        default_root_dir=config.floder,
        # callbacks=[checkpoint_callback],
        strategy='ddp_find_unused_parameters_true'
        )
    
    trainer.fit(diffusion)
    
