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
import torch.backends
import torch.nn.functional as F
from MinkowskiEngine.MinkowskiBroadcast import MinkowskiBroadcastAddition, MinkowskiBroadcastMultiplication
from torch.optim.lr_scheduler import SequentialLR
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
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel,DDIMScheduler
from examples.ae_res import ModelNet40Dataset, PointCloud, ShapeNetDataset, save_png, sorted_by_morton_code
from examples.ae_res_v100 import rotate_point_cloud,VaeModule,VAE
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
import MinkowskiEngine.MinkowskiFunctional as MF




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


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

class diffusionModule(LightningModule):
    def __init__(self,vae,unet,config,time_embedding_type="positional",flip_sin_to_cos=True,freq_shift: int = 0,num_train_timesteps: Optional[int] = None):
        super().__init__()
        for name, value in vars().items():
            if name != "self":
                setattr(self, name, value)
        self.config = config
        # self.noise_scheduler = DDPMScheduler(num_train_timesteps=config.ddpm_num_steps, beta_schedule=config.ddpm_beta_schedule,prediction_type=config.prediction_type,clip_sample=False)
        # self.noise_scheduler = DDPMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")
        self.noise_scheduler=DDPMScheduler(     num_train_timesteps=config.ddpm_num_steps,
                                                beta_start= 0.00085,
                                                beta_end= 0.012,
                                                prediction_type=config.prediction_type,
                                                beta_schedule=config.ddpm_beta_schedule,
                                                clip_sample=False,
                                                variance_type= "fixed_small",
                                                )
        self.denoise_scheduler=DDIMScheduler(   num_train_timesteps=config.ddpm_num_steps,
                                                beta_start= 0.00085,
                                                beta_end= 0.012,
                                                prediction_type=config.prediction_type,
                                                beta_schedule=config.ddpm_beta_schedule,
                                                clip_sample=False,
                                                set_alpha_to_one= False,
                                                steps_offset= 1
                                                )
        
        if time_embedding_type == "positional":
            self.time_proj = Timesteps(320, flip_sin_to_cos, freq_shift)
            timestep_input_dim = 320
        elif time_embedding_type == "learned":
            pass

        time_embed_dim=320*4
        self.time_embedding = TimestepEmbedding(timestep_input_dim, time_embed_dim)

        self.noise_point_mu=nn.Parameter(torch.zeros(3))
        self.noise_point_sigma=nn.Parameter(torch.eye(3))
        # self.expansion= ME.MinkowskiGenerativeConvolutionTranspose(in_channels=1,                                                                                                     
        #                                                 out_channels=1,                                                                                                    
        #                                                 kernel_size=2,                                                                                                     
        #                                                 stride=1,                                                                                                          
        #                                                 bias=False,                                                                                                        
        #                                                 dimension=A.dimension)




            # self.time_proj = nn.Embedding(num_train_timesteps, 320)
            # timestep_input_dim = 320
    def predict_noise(self, noise_latent,timesteps,encoder_hidden_states):
        return self.unet(noise_latent,timesteps,encoder_hidden_states)
    def is_batch_dim_sorted(self,tensor):
        # 获取张量的batch维
        batch_dim = tensor[:, 0]
        
        # 判断是否升序排列
        is_sorted = torch.all(batch_dim[1:] >= batch_dim[:-1])
        
        return is_sorted
    def training_step(self,batch, batch_idx):
        # lr = self.optimizer.param_groups[0]['lr']
        # self.log('learning_rate', lr, on_step=True,prog_bar=True)
        gc.collect()
        loss=0

        sin = ME.SparseTensor(
            features=torch.ones(len(batch["coords"]), 1).to(batch["coords"].device),
            coordinates=batch["coords"].int(),
        )

        with torch.no_grad():
            clean,_=self.vae.encoder(sin)
        normal_dist = torch.distributions.MultivariateNormal(self.noise_point_mu, covariance_matrix=self.noise_point_sigma)
        real_coord=clean.C[:,1:]/self.config.resolution
        nll_loss = -normal_dist.log_prob(real_coord).mean()
        loss+=nll_loss*0.01
        if config.noise_point_mode!="none":
        
            with torch.no_grad():
                latent_resolution=self.config.resolution//clean.tensor_stride[0]
                batch_size=self.config.batch_size
                # noise_point_num=torch.randint(low=1,high=self.config.noise_point_max,size=(1,),device=clean.F.device)
                noise_point_num=self.config.noise_point_max

                if config.noise_point_mode=="uniform":
                    noise_point=torch.rand((batch_size*noise_point_num,3),device=clean.F.device)*(latent_resolution-0.01)
                    
                elif config.noise_point_mode=="normal":
                    noise_point=normal_dist.sample((batch_size*noise_point_num,)).clamp(0,1)*(latent_resolution-0.01)

                noise_point=torch.split(noise_point, noise_point_num)
                # 生成量化坐标
                quantized_coords = []
                for batch in noise_point:
                    quantized_batch = ME.utils.sparse_quantize(batch)
                    quantized_coords.append(quantized_batch)
                quantized_coords=list(quantized_coords)
                quantized_coords=ME.utils.batched_coordinates(quantized_coords).to(clean.F.device)
                quantized_coords[:,1:]=quantized_coords[:,1:]*clean.tensor_stride[0]
                noise_point_tensor = ME.SparseTensor(
                    features=torch.zeros(quantized_coords.shape[0],clean.F.shape[1],device=clean.F.device),
                    coordinates=quantized_coords,
                    tensor_stride=clean.tensor_stride,
                    coordinate_manager=clean.coordinate_manager,
                )
                #生成和输入点临近的噪声点
                if config.noise_near:
                    expansion= ME.MinkowskiGenerativeConvolutionTranspose(in_channels=clean.F.shape[1],out_channels=1,kernel_size=3,stride=1,bias=False,dimension=clean.dimension).to(clean.F.device)
                    expansion_point=expansion(clean).C
                    expansion_point = expansion_point[(expansion_point>=0).all(dim=1)]
                    expansion_point_tensor = ME.SparseTensor(
                        features=torch.zeros(expansion_point.shape[0],clean.F.shape[1],device=clean.F.device),
                        coordinates=expansion_point,
                        tensor_stride=clean.tensor_stride,
                        coordinate_manager=clean.coordinate_manager,
                    )

                    noise_point_tensor=noise_point_tensor+expansion_point_tensor

            clean=clean+noise_point_tensor
        
        # clean=sorted_by_morton_code(clean)

        clean_dense, _, _ = clean.dense(shape=torch.Size([self.config.batch_size, self.config.unet_channel, self.config.resolution//clean.tensor_stride, self.config.resolution//clean.tensor_stride, self.config.resolution//clean.tensor_stride]))

        # coords, features = clean.C,clean.F

        noise=torch.randn_like(clean_dense,device=clean_dense.device)
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, (self.config.batch_size,), device=noise.device
        ).long()

        noised_latent=self.noise_scheduler.add_noise(clean_dense, noise, timesteps)

        t_emb=self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=self.dtype)
        emb = self.time_embedding(t_emb) #b,1280

        
        model_output = self.predict_noise(noised_latent,timesteps,)

        if self.config.prediction_type == "epsilon":
            denoise_loss=F.mse_loss(model_output,noise).mean() #每个样本贡献差不多的loss(有的点多有的点少)

            #下面的代码考虑了点多的样本贡献更多的loss(通过instance内求sum保留点数的影响)
            # batch_masks = [batch_indices == idx for idx in unique_batch_indices]
            # batch_masks = torch.stack(batch_masks)  # 将掩码堆叠起来    
            # masked_outputs = model_output.F[None, :] * batch_masks[:, :, None]
            # masked_noise = noise[None, :] * batch_masks[:, :, None]
            # loss = F.mse_loss(masked_outputs, masked_noise).mean(dim=-1).sum(dim=-1).mean()

        elif self.config.prediction_type == "sample":
            # TODO 现在是diffusers的实现，后续修改
            alpha_t = _extract_into_tensor(
                self.noise_scheduler.alphas_cumprod, timesteps, (bsz,)
            )
            snr_weights = alpha_t / (1 - alpha_t)

            batch_masks = [batch_indices == idx for idx in unique_batch_indices]
            batch_masks = torch.stack(batch_masks)  # 将掩码堆叠起来    
            # import ipdb;ipdb.set_trace()
            masked_outputs = model_output.F[None, :] * batch_masks[:, :, None]  # 对输出进行掩码 b,len,4 * b,len,1
            masked_clean = clean.F[None, :] * batch_masks[:, :, None]  # 对清洁数据进行掩码
            mse_losses = F.mse_loss(masked_outputs, masked_clean, reduction="none").mean(dim=-1)  # 计算均方误差损失

            # 计算加权损失
            weighted_losses = snr_weights[unique_batch_indices] * mse_losses.mean(dim=-1)

            denoise_loss = weighted_losses.mean()
        loss+=denoise_loss
        # print(loss)
        self.log('denoise_loss',denoise_loss.detach().item(),batch_size=self.config.batch_size, on_step=True, on_epoch=True, prog_bar=True)
        self.log('nll_loss',nll_loss.detach().item(),batch_size=self.config.batch_size, on_step=True, on_epoch=True, prog_bar=True)

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
        #加人噪点
        if config.noise_point_mode!="none":
            normal_dist = torch.distributions.MultivariateNormal(self.noise_point_mu, covariance_matrix=self.noise_point_sigma)
            with torch.no_grad():
                latent_resolution=self.config.resolution//clean.tensor_stride[0]
                batch_size=self.config.batch_size

                noise_point_num=self.config.noise_point_max

                if config.noise_point_mode=="uniform":
                    noise_point=torch.rand((batch_size*noise_point_num,3),device=clean.F.device)*(latent_resolution-0.01)
                    
                elif config.noise_point_mode=="normal":
                    noise_point=normal_dist.sample((batch_size*noise_point_num,)).clamp(0,1)*(latent_resolution-0.01)

                noise_point=torch.split(noise_point, noise_point_num)
                # 生成量化坐标
                quantized_coords = []
                for batch in noise_point:
                    quantized_batch = ME.utils.sparse_quantize(batch)
                    quantized_coords.append(quantized_batch)
                quantized_coords=list(quantized_coords)
                quantized_coords=ME.utils.batched_coordinates(quantized_coords).to(clean.F.device)
                quantized_coords[:,1:]=quantized_coords[:,1:]*clean.tensor_stride[0]
                noise_point_tensor = ME.SparseTensor(
                    features=torch.zeros(quantized_coords.shape[0],clean.F.shape[1],device=clean.F.device),
                    coordinates=quantized_coords,
                    tensor_stride=clean.tensor_stride,
                    coordinate_manager=clean.coordinate_manager,
                )
                # print(clean.C)
                #生成和输入点临近的噪声点
                if config.noise_near:
                    expansion= ME.MinkowskiGenerativeConvolutionTranspose(in_channels=clean.F.shape[1],out_channels=1,kernel_size=3,stride=1,bias=False,dimension=clean.dimension).to(clean.F.device) #可以生成和输入点临近的噪声点，通过kernel_size控制半径
                    expansion_point=expansion(clean).C
                    expansion_point = expansion_point[(expansion_point>=0).all(dim=1)]
                    expansion_point_tensor = ME.SparseTensor(
                        features=torch.zeros(expansion_point.shape[0],clean.F.shape[1],device=clean.F.device),
                        coordinates=expansion_point,
                        tensor_stride=clean.tensor_stride,
                        coordinate_manager=clean.coordinate_manager,
                    )

                    noise_point_tensor=noise_point_tensor+expansion_point_tensor
            clean=clean+noise_point_tensor
        
        clean=sorted_by_morton_code(clean)

        assert self.is_batch_dim_sorted(clean.C)
        
        coords, features = clean.C,clean.F
        noise=torch.randn_like(features)
        batch_indices = coords[:, 0]
        unique_batch_indices = torch.unique(batch_indices)
        bsz=unique_batch_indices.shape[0]
        self.denoise_scheduler.set_timesteps(50)
        # features=self.noise_scheduler.add_noise(clean.F, noise, self.noise_scheduler.timesteps[0]) #越前面加噪越多
        features=torch.randn_like(features)

        noise_sparse=ME.SparseTensor(
            features=noise,
            coordinate_map_key=clean.coordinate_map_key,
            tensor_stride=clean.tensor_stride,
            coordinate_manager=clean.coordinate_manager,
            )
        noise_clean=ME.SparseTensor(
            features=features,
            coordinate_map_key=clean.coordinate_map_key,
            tensor_stride=clean.tensor_stride,
            coordinate_manager=clean.coordinate_manager,
            )

        for t in self.denoise_scheduler.timesteps:
            tt=torch.zeros(bsz,device=clean.device).long()+t
            t_emb=self.time_proj(tt)
            t_emb = t_emb.to(dtype=self.dtype)
            emb = self.time_embedding(t_emb) #b,1280
            model_output = self.predict_noise(noise_clean,emb)
            # print(noise_clean.coordinate_map_key)
            # print(model_output.coordinate_map_key)
            # print(model_output.F.shape)
            # print(noise_sparse.F.shape)
            noise_clean=ME.SparseTensor(
                features=self.denoise_scheduler.step(model_output.F,tt[0],noise_clean.F).prev_sample,
                coordinate_map_key=clean.coordinate_map_key,
                tensor_stride=clean.tensor_stride,
                coordinate_manager=clean.coordinate_manager,
                )
        target_key = sin.coordinate_map_key
        _,_,sout=self.vae.decoder(noise_clean,target_key)
        _,_,clean=self.vae.decoder(clean,target_key)

        if batch_idx==0 and str(self.device) == 'cuda:0':
            fig = plt.figure(figsize=(20, 20))
            batch_coords, _ = sout.decomposed_coordinates_and_features
            real_batch_coords,_=clean.decomposed_coordinates_and_features
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
        optimizer = optim.AdamW(
            [
                {'params':self.unet.parameters(),'lr':config.lr},
                {'params':self.time_embedding.parameters(),'lr':config.lr},
                {'params':[self.noise_point_mu,self.noise_point_sigma],'lr':config.lr}
             ],
             betas=(0.9, 0.99),
             eps=1e-6,
        )

        linearLR = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-2, end_factor=1.0, total_iters=5000)
        cosineAnnealingLR = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000, eta_min=0)  
        scheduler=SequentialLR(optimizer,schedulers=[linearLR,cosineAnnealingLR],milestones=[5000])
        # return [optimizer],[scheduler]
        # return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
    
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
        "batch_size": min(self.config.batch_size,4),
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
    pa.add_argument("--max_epochs", type=int, default=10000, help="Max epochs")
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
    pa.add_argument("--ddpm_beta_schedule", type=str, default="scaled_linear", help="beta schedule. linear, scaled_linear, or squaredcos_cap_v2")
    pa.add_argument("--prediction_type", type=str, default="epsilon", help="prediction type.can be `epsilon` (predicts the noise of the diffusion process),`sample` (directly predicts the noisy sample`)")
    pa.add_argument("--vae_ckpt", type=str, default= "/mnt/data4/hjm_code_4/MinkowskiEngine-master/vae-epoch=0119-val_loss_epoch=0.05.ckpt", help="vae path")
    pa.add_argument("--unet_channel", type=int,nargs=4, default=(4,320,640,960), help="unet channel")
    pa.add_argument("--noise_point_max", type=int, default=1000, help="noise point max")
    pa.add_argument("--noise_point_mode", type=str, default="normal", help="noise point mode uniform or normal or none")
    pa.add_argument("--noise_near", default=False, action='store_true', help="noise near")







    config = pa.parse_args()
    os.makedirs(config.floder,exist_ok=True)
    seed_everything(config.seed,workers=True)
    torch.cuda.manual_seed_all(config.seed)
    train_dataset=ConcatDataset([
            ModelNet40Dataset(
                phase="train",
                # transform=rotate_point_cloud,
                config=config),

            # ShapeNetDataset(
            #     transform=rotate_point_cloud,
            #     config=config),
            ])
    val_dataset=ModelNet40Dataset(
            # phase="test",
            phase="train",
            config=config)
    
    num_devices = min(config.ngpus, torch.cuda.device_count())
    print(f"Testing {num_devices} GPUs.")
    unet=UNet3DConditionModel()
    if config.ngpus > 1:
        unet = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(unet)
    vae=VAE(config)
    checkpoint = torch.load(config.vae_ckpt, map_location="cpu")

    vae.load_state_dict({k[6:]: v for k, v in checkpoint["state_dict"].items()})
    # print({k: v for k, v in checkpoint["state_dict"].items()if k.startswith("model.encoder.")})

    diffusion = diffusionModule(vae,unet,config)

    trainer = Trainer(
        max_epochs=config.max_epochs,
        # precision="16-mixed",
        # precision='bf16-mixed',
        default_root_dir=config.floder,
        # callbacks=[checkpoint_callback],
        strategy='ddp_find_unused_parameters_true',
        profiler="simple",
        accumulate_grad_batches=1,#梯度累计
        gradient_clip_val=0.5,
        check_val_every_n_epoch=5
        )
    
    if config.recover:
        trainer.fit(diffusion, ckpt_path=config.recover)
    else:
        files = glob.glob(os.path.join(config.floder, 'checkpoints/*'))
        # 按文件名排序
        ckpt_path = sorted(files)[-1] if files else None
        print("loding checkpoint: ",ckpt_path)
        trainer.fit(diffusion,ckpt_path=ckpt_path)
    
