# Copyright (c) 2020 NVIDIA CORPORATION.
# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu).
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Please cite "4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural
# Networks", CVPR'19 (https://arxiv.org/abs/1904.08755) if you use any part
# of the code.
import os
import sys
# sys.path.append('/mnt/data4/hjm_code_4/MinkowskiEngine-master')
# from memory_profiler import profile
import subprocess
import argparse
import logging
import glob
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
    from pytorch_lightning import Trainer
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
from torch.utils.data import Dataset, DataLoader
from examples.reconstruction import InfSampler, resample_mesh
from MinkowskiEngine.modules.vae_block import ResNetBlock,ResNet2,ResNet3
from torch.nn import TransformerEncoder, TransformerEncoderLayer

M = np.array(
    [
        [0.80656762, -0.5868724, -0.07091862],
        [0.3770505, 0.418344, 0.82632997],
        [-0.45528188, -0.6932309, 0.55870326],
    ]
)

if not os.path.exists("ModelNet40"):
    logging.info("Downloading the fixed ModelNet40 dataset...")
    subprocess.run(["sh", "./examples/download_modelnet40.sh"])


###############################################################################
# Utility functions
###############################################################################
def PointCloud(points, colors=None):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd

def sorted_by_morton_code(sinput):
    """计算莫顿码，默认所有数都小于32位(这里最大应该是1024分辨率，2048需要给dim乘更大的数，保证batch不被打乱)"""
    coords = sinput.C
    dim0 = coords[:, 0] if coords.shape[1] == 4 else 0 #如果是批数据，要分开
    x = coords[:, -3].long()
    y = coords[:, -2].long()
    z = coords[:, -1].long()

    morton_code=0
    for i in range(31, -1, -1):
        morton_code |= ((x >> i) & 1) << (3 * i + 2)
        morton_code |= ((y >> i) & 1) << (3 * i + 1)
        morton_code |= ((z >> i) & 1) << (3 * i)
    sort_key = dim0 * 1e10 + morton_code
    indices = torch.argsort(sort_key)
    out_sorted_F=sinput.F.index_select(0,indices)
    out_sorted_C=sinput.C.index_select(0,indices)

    out_sorted = ME.SparseTensor(
        features=out_sorted_F,
        coordinates=out_sorted_C,
        tensor_stride=sinput.tensor_stride,
        coordinate_manager=sinput.coordinate_manager,
    )
    return out_sorted

def collate_pointcloud_fn(list_data):
    list_data=list(filter(lambda x: x is not None,list_data))
    coords, feats, labels = list(zip(*list_data))
    # Concatenate all lists
    return {
        "coords": ME.utils.batched_coordinates(coords),
        "xyzs": [torch.from_numpy(feat).float() for feat in feats],
        "labels": torch.LongTensor(labels),
    }

class ModelNet40Dataset(torch.utils.data.Dataset):
    def __init__(self, phase, transform=None, config=None):
        print('init ModelNet40Dataset')
        self.phase = phase
        self.files = []
        # self.cache = {}
        self.data_objects = []
        self.transform = transform
        self.resolution = config.resolution
        self.last_cache_percent = 0

        # self.root = "/dev/shm/ModelNet40"
        self.root = "./ModelNet40"
        # fnames = glob.glob(os.path.join(self.root, f"chair/{phase}/*.off"))
        fnames = glob.glob(os.path.join(self.root, f"*/{phase}/*.off"))
        fnames = sorted([os.path.relpath(fname, self.root) for fname in fnames])
        self.files = fnames
        assert len(self.files) > 0, "No file loaded"
        logging.info(
            f"Loading the subset {phase} from {self.root} with {len(self.files)} files"
        )
        self.density = 30000

        # Ignore warnings in obj loader
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mesh_file = os.path.join(self.root, self.files[idx])
        preprocessed_file = mesh_file.replace('.off', '_xyz.npy')

        if os.path.exists(preprocessed_file):
            xyz = np.load(preprocessed_file)
            # feats = data[:, 3:]
        else:
            assert os.path.exists(mesh_file)
            pcd = o3d.io.read_triangle_mesh(mesh_file)
            vertices = np.asarray(pcd.vertices)
            vmax = vertices.max(0, keepdims=True)
            vmin = vertices.min(0, keepdims=True)
            pcd.vertices = o3d.utility.Vector3dVector((vertices - vmin) / (vmax - vmin).max())
            xyz = resample_mesh(pcd, density=self.density)

            feats = np.ones((len(xyz), 1))
            np.save(preprocessed_file, xyz)

            if len(xyz) < 1000:
                logging.info(
                    f"Skipping {mesh_file}: does not have sufficient CAD sampling density after resampling: {len(xyz)}."
                )
                return None

            if self.transform:
                xyz, feats = self.transform(xyz, feats)

        # Get coords
        xyz = xyz * (self.resolution-1)
        coords = np.floor(xyz)
        inds = ME.utils.sparse_quantize(coords, return_index=True, return_maps_only=True)

        return (coords[inds], xyz[inds], idx)


def rotate_point_cloud(xyz,feat):
    """ Randomly rotate the point clouds to augment the dataset."""
    # rotated_xyz = torch.zeros(xyz.size(), dtype=torch.float32)

    rotation_angle = torch.rand(3) * 2 * np.pi
    cosval = torch.cos(rotation_angle)
    sinval = torch.sin(rotation_angle)
    rotation_matrix_x = torch.tensor([[1, 0, 0],
                                        [0, cosval[0], -sinval[0]],
                                        [0, sinval[0], cosval[0]]])
    rotation_matrix_y = torch.tensor([[cosval[1], 0, sinval[1]],
                                        [0, 1, 0],
                                        [-sinval[1], 0, cosval[1]]])
    rotation_matrix_z = torch.tensor([[cosval[2], -sinval[2], 0],
                                        [sinval[2], cosval[2], 0],
                                        [0, 0, 1]])
    rotation_matrix = torch.mm(torch.mm(rotation_matrix_x, rotation_matrix_y), rotation_matrix_z)
    shape_pc = torch.from_numpy(xyz).float()
    rotated_xyz = torch.mm(shape_pc.reshape((-1, 3)), rotation_matrix).numpy()

    vmax = rotated_xyz.max(0, keepdims=True)
    vmin = rotated_xyz.min(0, keepdims=True)
    # pcd.rotated_xyz = o3d.utility.Vector3dVector(
    #     (vertices - vmin) / (vmax - vmin).max()
    # )
    rotated_xyz=(rotated_xyz - vmin) / (vmax - vmin).max()

    return rotated_xyz,feat


def make_data_loader(
    phase, augment_data, batch_size, shuffle, num_workers, repeat, config
):
    if augment_data:
        dset = ModelNet40Dataset(phase,transform=rotate_point_cloud,config=config)
    else:
        dset = ModelNet40Dataset(phase,config=config)


    args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "collate_fn": collate_pointcloud_fn,
        "pin_memory": False,
        "drop_last": False,
    }

    if repeat:
        args["sampler"] = InfSampler(dset, shuffle)
    else:
        args["shuffle"] = shuffle

    loader = torch.utils.data.DataLoader(dset, **args)

    return loader



ch = logging.StreamHandler(sys.stdout)
logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    format=os.uname()[1].split(".")[0] + " %(asctime)s %(message)s",
    datefmt="%m/%d %H:%M:%S",
    handlers=[ch],
)

###############################################################################
# End of utility functions
###############################################################################


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_x, max_y, max_z,max_stride):
        super(LearnedPositionalEncoding, self).__init__()
        self.x_embedding = nn.Embedding(max_x, d_model)
        self.y_embedding = nn.Embedding(max_y, d_model)
        self.z_embedding = nn.Embedding(max_z, d_model)
        self.stride_embedding = nn.Embedding(max_stride, d_model)

        
    def forward(self, coords,stride=1):
        stride=torch.ones_like(coords[:, 0])*stride
        x, y, z = coords[:, 1], coords[:, 2], coords[:, 3]
        x_pos = self.x_embedding(x)
        y_pos = self.y_embedding(y)
        z_pos = self.z_embedding(z)
        stride_emb = self.stride_embedding(stride)
        return x_pos + y_pos + z_pos+stride_emb
    
class MortonWindowTransformer(nn.Module):
    '''经过莫顿编码排序，固定步长切分窗口，然后进行Transformer编码，最后拼接回大batch'''
    def __init__(self, d_model, nhead, num_layers,resolution=128,window_size=50,interval=1):
        super(MortonWindowTransformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead) #包含多头注意力块和前馈神经网络，多头注意力机制块是点积、缩放、softmax后的加权求和
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pe=LearnedPositionalEncoding(d_model,max_x=resolution,max_y=resolution,max_z=resolution,max_stride=resolution)
        self.window_size=window_size
        self.interval=interval
    def forward(self,sinput):
        #N,C 排序
        sorted_sinput=sorted_by_morton_code(sinput)
        #add positionEncoding 位置编码
        sorted_sinput_C,sorted_sinput_F=sorted_sinput.C,sorted_sinput.F
        pe=self.pe(sorted_sinput_C,sorted_sinput.tensor_stride[0])
        sorted_sinput_F=sorted_sinput_F+pe
        # Split the features based on batch indices,then split to windows with padding mask 把大batch分成小窗，不同实例的被分开，不够分的补齐
        coords, features = sorted_sinput_C, sorted_sinput_F
        batch_indices = coords[:, 0].cpu().numpy()
        unique_batch_indices = np.unique(batch_indices)
        windows = []
        window_masks = []

        #siplt batch into instances then split into windows
        for batch_idx in unique_batch_indices:
            batch_mask = (batch_indices == batch_idx)
            batch_features = features[batch_mask]
            # Split the instance into windows
            # print("batch_features.shape: ",batch_features.shape)
            start=[]
            for i in range(self.interval):
                for j in range(batch_features.shape[0]//self.window_size+1):
                    if j*self.window_size*self.interval+i<batch_features.shape[0]:
                        start.append(j*self.window_size*self.interval+i) 
            start.sort()
             
            for i in start:
                #切分窗口,窗口内元素间隔为interval
                if i+self.window_size*self.interval>batch_features.shape[0]:
                    window = batch_features[i::self.interval]
                else:
                    window = batch_features[i:i+self.window_size*self.interval:self.interval]
                #窗口mask，如果窗口长度不够，补齐部分为False，不计算
                window_mask = torch.ones(self.window_size, dtype=torch.bool)
                window_mask[window.shape[0]:] = False
                # print("window_mask: ",window_mask)
                #如果窗口长度不够预设长度，补齐
                if window.shape[0] < self.window_size:
                    pad_size = self.window_size - window.shape[0]
                    window = torch.cat([window, torch.zeros(pad_size, window.shape[1]).to(window.device)], dim=0)
                windows.append(window)
                window_masks.append(window_mask)

        windows = torch.stack(windows, dim=0)#n/window_size,window_size,C
        window_masks = torch.stack(window_masks, dim=0).to(windows.device)
        transformer_output = self.transformer_encoder(windows.permute(1,0,2), src_key_padding_mask=~window_masks)#mask取反，TransformerEncoder中mask为true才忽略

        #把窗口结果去除padding并拼接回大batch
        batch_output=transformer_output.permute(1,0,2)
        final_outputs = []
        for i in range(batch_output.shape[0]):
            output = batch_output[i]
            final_outputs.append(output[window_masks[i]])

        output=torch.cat(final_outputs, dim=0)

        out_sorted = ME.SparseTensor(
            features=output,
            coordinates=sorted_sinput_C,
            tensor_stride=sinput.tensor_stride,
            coordinate_manager=sinput.coordinate_manager,
        )
        return out_sorted

class Encoder(nn.Module):

    # CHANNELS = [16, 32, 64, 128, 256, 512, 1024]
    # CHANNELS = [128, 256, 512, 512 ,256 ,128]
    CHANNELS = [32, 128, 512, 512,4]

    def __init__(self,config):
        nn.Module.__init__(self)
        self.resoluthon=config.resolution
        self.window_size=config.window_size
        # Input sparse tensor must have tensor stride 128.
        ch = self.CHANNELS
        # self.window_size=50
        # Block 1
        self.block1 =ResNet2(1, ch[0],after='downsample')
        # self.block1 = nn.Sequential(
        #     ME.MinkowskiConvolution(1, ch[0], kernel_size=3, stride=2, dimension=3),
        #     ME.MinkowskiBatchNorm(ch[0]),
        #     ME.MinkowskiELU(),
        #     ME.MinkowskiConvolution(ch[0], ch[0], kernel_size=3, dimension=3),
        #     ME.MinkowskiBatchNorm(ch[0]),
        #     ME.MinkowskiELU(),
        # )
        self.block2 =ResNet2(ch[0], ch[1],after='downsample')
        self.block3 =ResNet2(ch[1], ch[2],after='downsample')
        self.block4 =ResNet2(ch[2], ch[3],after=None)
        self.block5 =ResNet2(ch[3], ch[4],after='downsample')
        # self.block6 =ResNet2(ch[4], ch[5],after='downsample')

        self.sorted_window_transformer=MortonWindowTransformer(d_model=ch[3],nhead=8,num_layers=2,resolution=self.resoluthon,window_size=self.window_size,interval=1)

        self.mean_conv=ME.MinkowskiConvolution(ch[-1], ch[-1], kernel_size=3, dimension=3)
        # self.log_var_conv=ME.MinkowskiConvolution(ch[5], ch[5], kernel_size=3, dimension=3)
        
        self.weight_initialization()

    def weight_initialization(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(m.kernel, mode="fan_out", nonlinearity="relu")

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def forward(self, sinput):
        # import ipdb;ipdb.set_trace()
        out = self.block1(sinput)
        out = self.block2(out)
        out = self.block3(out)
        out = self.block4(out)
        out=self.sorted_window_transformer(out)
        out = self.block5(out)

        mean = self.mean_conv(out)
        # log_var = self.log_var_conv(out)
        # return mean, log_var
        return mean,None


class Decoder(nn.Module):

    # CHANNELS = [1024, 512, 256, 128, 64, 32, 16]
    # CHANNELS = [128, 256, 512, 512 ,256 ,128]
    CHANNELS = [4, 512, 512 ,128 ,32]

    # resolution = 128

    def __init__(self,config):
        nn.Module.__init__(self)
        self.resoluthon=config.resolution
        self.window_size=config.window_size
        # Input sparse tensor must have tensor stride 128.
        ch = self.CHANNELS

        # Block 1
        self.block1 = ResNet2(ch[0], ch[1],after='upsample')
        # self.block1 = nn.Sequential(
        #     ME.MinkowskiGenerativeConvolutionTranspose(
        #         ch[0], ch[1], kernel_size=2, stride=2, dimension=3
        #     ),
        #     ME.MinkowskiBatchNorm(ch[1]),
        #     ME.MinkowskiELU(),
        #     ME.MinkowskiConvolution(ch[1], ch[1], kernel_size=3, dimension=3),
        #     ME.MinkowskiBatchNorm(ch[1]),
        #     ME.MinkowskiELU(),
        # )

        self.block1_cls = ME.MinkowskiConvolution(
            ch[1], 1, kernel_size=1, bias=True, dimension=3
        )

        # Block 2

        self.block2 = ResNet2(ch[1], ch[2],after='upsample')
        self.block2_cls = ME.MinkowskiConvolution(
            ch[2], 1, kernel_size=1, bias=True, dimension=3
        )

        # Block 3

        self.block3 = ResNet2(ch[2], ch[3],after='upsample')
        self.block3_cls = ME.MinkowskiConvolution(
            ch[3], 1, kernel_size=1, bias=True, dimension=3
        )

        # Block 4

        self.block4 = ResNet2(ch[3], ch[4],after='upsample')
        self.block4_cls = ME.MinkowskiConvolution(
            ch[4], 1, kernel_size=1, bias=True, dimension=3
        )

        # Block 5
        self.sorted_window_transformer=MortonWindowTransformer(d_model=ch[1],nhead=8,num_layers=2,resolution=self.resoluthon,window_size=self.window_size,interval=1)

        # self.block5 = ResNet2(ch[4], ch[5],after='upsample')
        # self.block5_cls = ME.MinkowskiConvolution(
        #     ch[5], 1, kernel_size=1, bias=True, dimension=3
        # )

        # pruning
        self.pruning = ME.MinkowskiPruning()

    def get_batch_indices(self, out):
        return out.coords_man.get_row_indices_per_batch(out.coords_key)

    @torch.no_grad()
    def get_target(self, out, target_key, kernel_size=1):
        target = torch.zeros(len(out), dtype=torch.bool, device=out.device)
        cm = out.coordinate_manager
        strided_target_key = cm.stride(target_key, out.tensor_stride[0])
        kernel_map = cm.kernel_map(
            out.coordinate_map_key,
            strided_target_key,
            kernel_size=kernel_size,
            region_type=1,
        )
        for k, curr_in in kernel_map.items():
            target[curr_in[0].long()] = 1
        return target

    def valid_batch_map(self, batch_map):
        for b in batch_map:
            if len(b) == 0:
                return False
        return True

    def forward(self, z_glob, target_key):
        out_cls, targets = [], []
        # print(z_glob.F.shape,z_glob.C.shape)
        # z = ME.SparseTensor(
        #     features=z_glob.F,
        #     coordinates=z_glob.C,
        #     tensor_stride=z_glob.tensor_stride,
        #     coordinate_manager=z_glob.coordinate_manager,
        # )
        # z=z_glob

        # Block1
        out1 = self.block1(z_glob)
        out1_cls = self.block1_cls(out1)
        target = self.get_target(out1, target_key)
        targets.append(target)
        out_cls.append(out1_cls)
        keep1 = (out1_cls.F > 0).squeeze()

        # If training, force target shape generation, use net.eval() to disable
        if self.training:
            keep1 += target

        # Remove voxels 32
        if keep1.sum() > 1:
            out1 = self.pruning(out1, keep1)

        out1=self.sorted_window_transformer(out1)

        # Block 2
        out2 = self.block2(out1)
        out2_cls = self.block2_cls(out2)
        target = self.get_target(out2, target_key)
        targets.append(target)
        out_cls.append(out2_cls)
        keep2 = (out2_cls.F > 0).squeeze()

        if self.training:
            keep2 += target
        
        # Remove voxels 16
        if keep2.sum() > 1:
            out2 = self.pruning(out2, keep2)

        # Block 3
        out3 = self.block3(out2)
        out3_cls = self.block3_cls(out3)
        target = self.get_target(out3, target_key)
        targets.append(target)
        out_cls.append(out3_cls)
        keep3 = (out3_cls.F > 0).squeeze()

        if self.training:
            keep3 += target
        
        # Remove voxels 8
        if keep3.sum() > 1:
            out3 = self.pruning(out3, keep3)

        # Block 4
        out4 = self.block4(out3)
        out4_cls = self.block4_cls(out4)
        target = self.get_target(out4, target_key)
        targets.append(target)
        out_cls.append(out4_cls)
        keep4 = (out4_cls.F > 0).squeeze()

        # if self.training:
        #     keep4 += target
        
        # Remove voxels 4
        if keep4.sum() > 1:
            out4 = self.pruning(out4, keep4)

        # # Block 5
        # out5 = self.block5(out4)
        # out5_cls = self.block5_cls(out5)
        # target = self.get_target(out5, target_key)
        # targets.append(target)
        # out_cls.append(out5_cls)
        # keep5 = (out5_cls.F > 0).squeeze()

        # # if self.training:
        # #     keep5 += target
        
        # # Remove voxels 2
        # if keep5.sum() > 1:
        #     out5 = self.pruning(out5, keep5)

        # return out_cls, targets, out5
        return out_cls, targets, out4
        


class VAE(nn.Module):
    def __init__(self,config):
        nn.Module.__init__(self)
        self.config=config
        self.encoder = Encoder(self.config)
        self.decoder = Decoder(self.config)


    def forward(self, sinput, gt_target):
        means, log_vars = self.encoder(sinput)
        zs = means
        # if self.training:
        #     zs = zs + torch.exp(0.5 * log_vars.F) * torch.randn_like(log_vars.F)
        out_cls, targets, sout = self.decoder(zs, gt_target)
        return out_cls, targets, sout, means, log_vars, zs
    
    # def training_step(self,batch,batch_idx):
    #     stensor = ME.SparseTensor(
    #         coordinates=batch["coordinates"], features=batch["features"]
    #     )        
    #     sin = ME.SparseTensor(
    #         features=torch.ones(len(batch["coords"]), 1),
    #         coordinates=batch["coords"].int(),
    #     )




class VaeModule(LightningModule):
    def __init__(self,model,config):
        super().__init__()
        for name, value in vars().items():
            if name != "self":
                setattr(self, name, value)
        self.config = config
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, sin,target_key):
        return self.model(sin,target_key)

    def training_step(self, batch, batch_idx):
        # stensor = ME.SparseTensor(
        #     coordinates=batch["coordinates"], features=batch["features"]
        # )
        # import ipdb;ipdb.set_trace()
        sin = ME.SparseTensor(
            features=torch.ones(len(batch["coords"]), 1).to(batch["coords"].device),
            coordinates=batch["coords"].int(),
        )
        # print(sin.C.shape)
        target_key = sin.coordinate_map_key
        out_cls, targets, sout, means, log_vars, zs = self(sin,target_key)
        num_layers, BCE = len(out_cls), 0
        losses = []
        loss_scales=[10 * (0.8**i) for i in range(len(out_cls))]
        # loss_scales = np.geomspace(10, 1, num=num_layers)
        for i,out_cl, target in zip(range(num_layers),out_cls, targets):
            # pos_weight = torch.tensor(loss_scale)
            # criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            curr_loss = self.criterion(out_cl.F.squeeze(), target.type(out_cl.F.dtype))
            losses.append(curr_loss.item())
            BCE += curr_loss / num_layers

        # for i in range(len(losses)):
        #     self.log(f'train_loss_{i:02}',losses[i],batch_size=self.config.batch_size, on_epoch=True, sync_dist=True)

        # KLD = -0.5 * torch.mean(
        #     torch.mean(1 + log_vars.F - means.F.pow(2) - log_vars.F.exp(), 1)
        # )
        # loss = KLD + BCE
        loss=BCE
        self.log('train_loss',loss.detach().item(),batch_size=self.config.batch_size, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        if self.current_epoch ==0:
            return
        sin = ME.SparseTensor(
            features=torch.ones(len(batch["coords"]), 1).to(batch["coords"].device),
            coordinates=batch["coords"].int(),
        )
        target_key = sin.coordinate_map_key
        # print(sin.F.dtype)
        out_cls, targets, sout, means, log_vars, zs = self.model(sin,target_key)
        if batch_idx==0 and str(self.device) == 'cuda:0':
            fig = plt.figure(figsize=(20, 20))
            batch_coords, batch_feats = sout.decomposed_coordinates_and_features
            for b, (coords, feats) in enumerate(zip(batch_coords, batch_feats)):
                if b==4:
                    break
                # print(len(coords))
                pcd = PointCloud(coords.cpu())
                pcd.estimate_normals()
                pcd.translate([0.6 * self.config.resolution, 0, 0])
                # pcd.rotate(M)
                opcd = PointCloud(batch["xyzs"][b].cpu())
                opcd.translate([-0.6 * self.config.resolution, 0, 0])
                opcd.estimate_normals()
                # opcd.rotate(M)
                # save_png(pcd,opcd,os.path.join(config.floder,f'{i:09}_{b}.png'))
                ax=fig.add_subplot(2,2,b+1,projection='3d')
                save_png(pcd,opcd,ax)
                plt.axis('equal')
                plt.tight_layout()
                plt.savefig(os.path.join(self.config.floder,f'epoch_{self.current_epoch:04}.png'))
        
        num_layers, BCE = len(out_cls), 0
        losses = []
        for out_cl, target in zip(out_cls, targets):
            curr_loss = self.criterion(out_cl.F.squeeze(), target.type(out_cl.F.dtype))
            losses.append(curr_loss.item())
            BCE += curr_loss / num_layers

        # KLD = -0.5 * torch.mean(
        #     torch.mean(1 + log_vars.F - means.F.pow(2) - log_vars.F.exp(), 1)
        # )
        # loss = KLD + BCE
        loss=BCE
        self.log('val_loss',loss.detach().item(),batch_size=self.config.batch_size, on_step=True, on_epoch=True, prog_bar=True, logger=True,sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(
            self.model.parameters(),
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

def save_png(pcd,opcd,ax):

    pcd_point=np.asarray(pcd.points)
    opcd_point=np.asarray(opcd.points)

    # fig = plt.figure(figsize=(5, 5))  # 创建一个新的图形窗口
    
    # ax1=plt.axes(projection='3d')  # 在图形窗口中添加一个3D绘图区域
    ax.scatter(pcd_point[:, 0], pcd_point[:, 1], pcd_point[:, 2], c='r', s=0.05,
                alpha=0.5)  # 点的大小为0.05，透明度为0.5
    ax.scatter(opcd_point[:, 0], opcd_point[:, 1], opcd_point[:, 2], c='g', s=0.05,
            alpha=0.5)


if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--max_epochs", type=int, default=1000, help="Max epochs")
    pa.add_argument("--lr", type=float, default=1e-2, help="Learning rate")
    pa.add_argument("--batch_size", type=int, default=1, help="batch size per GPU")
    pa.add_argument("--ngpus", type=int, default=1, help="num_gpus")
    pa.add_argument("--num_workers", type=int, default=16, help="num_workers")
    pa.add_argument("--resolution", type=int, default=256, help="resolution")
    pa.add_argument("--window_size", type=int, default=50, help="window_size")
    pa.add_argument("--floder", type=str, default="ae_v100")

    os.makedirs(pa.parse_args().floder,exist_ok=True)

    config = pa.parse_args()
    train_dataset=ModelNet40Dataset(
            phase="train",
            transform=rotate_point_cloud,
            config=config)
    val_dataset=ModelNet40Dataset(
            phase="test",
            config=config)
    num_devices = min(config.ngpus, torch.cuda.device_count())
    print(f"Testing {num_devices} GPUs.")
    model=VAE(config)
    if config.ngpus > 1:
        model = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(model)
    pl_module=VaeModule(model,config)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss_epoch', #我们想要监视的指标 
        dirpath=config.floder+'/checkpoints',  #模型缓存目录
        every_n_epochs = 20, 
        save_top_k=-1,
        filename='vae-{epoch:04d}-{val_loss_epoch:.2f}'
        )

    trainer = Trainer(
        max_epochs=config.max_epochs,
        precision=16,
        default_root_dir=config.floder,
        callbacks=[checkpoint_callback],
        # strategy='ddp_find_unused_parameters_true'
        )
    trainer.fit(pl_module)
    # trainer.fit(pl_module,train_dataloaders=train_loader,val_dataloaders=val_loader)
