import gc
import os
import sys
# sys.path.append('/mnt/data4/hjm_code_4/MinkowskiEngine-master')
# sys.path.append('./examples')

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
# from examples.reconstruction import InfSampler, resample_mesh
from examples.reconstruction import InfSampler, resample_mesh

from MinkowskiEngine.modules.vae_block import ResNetBlock,ResNet2,ResNet3
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import configparser

from transformers import CLIPProcessor, CLIPModel,CLIPImageProcessor
from PIL import Image
import random
M = np.array(
    [
        [0.80656762, -0.5868724, -0.07091862],
        [0.3770505, 0.418344, 0.82632997],
        [-0.45528188, -0.6932309, 0.55870326],
    ]
)


# if not os.path.exists("ModelNet40"):
#     logging.info("Downloading the fixed ModelNet40 dataset...")
#     subprocess.run(["sh", "./examples/download_modelnet40.sh"])


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

def collate_pointcloud_fn(list_data,max_batch_len=1000000):
    list_data=list(filter(lambda x: x is not None,list_data))
    sorted_list = sorted(list_data, key=lambda x: len(x[0]))
    

    # coords, feats, labels = list(zip(*sorted_list))
    # coords, feats, obj_clss,labels = list(zip(*sorted_list))
    coords, feats, obj_clss,image_conditions,labels = list(zip(*sorted_list))


    while sum(len(item) for item in coords) > max_batch_len:
        print(
            f"this batch is too big, some data will be dropped, sum: {sum(len(item) for item in coords)}"
        )
        sorted_list = sorted_list[1:]
        # coords, feats, labels = list(zip(*sorted_list))
        coords, feats, obj_clss,labels= list(zip(*sorted_list))

    # Concatenate all lists
    return {
        "coords": ME.utils.batched_coordinates(coords),
        "xyzs": [torch.from_numpy(feat).float() for feat in feats],
        "captions":[f"a picture of a {o_cls}" for o_cls in obj_clss],
        "labels": torch.LongTensor(labels),
    }

class ModelNet40Dataset(torch.utils.data.Dataset):
    def __init__(self, phase, transform=None, config=None,root="/mnt/data4/hjm_code_4/MinkowskiEngine-master/data/ModelNet40",with_class=False):
        print('init ModelNet40Dataset')
        self.phase = phase
        self.files = []
        # self.cache = {}
        self.data_objects = []
        self.transform = transform
        self.resolution = config.resolution
        self.last_cache_percent = 0
        self.min=self.resolution**1.25+1000
        self.max=self.resolution**2.4+50000 #3090大约支持60万
        self.cache=config.cache
        self.with_class=with_class
        self.small_dataset=config.small_dataset

        # self.root = "/dev/shm/ModelNet40"
        self.root = root
        # fnames = glob.glob(os.path.join(self.root, f"bathtub/{phase}/*.off"))
        # fnames = glob.glob(os.path.join(self.root, f"chair/{phase}/*.off"))
        # fnames = glob.glob(os.path.join(self.root, f"chair/{phase}/*.off"))+glob.glob(os.path.join(self.root, f"bathtub/{phase}/*.off"))+glob.glob(os.path.join(self.root, f"airplane/{phase}/*.off"))
        if phase=="train":
            fnames = glob.glob(os.path.join(self.root, f"chair/{phase}/*.off"))
        elif phase=="test":
            fnames = glob.glob(os.path.join(self.root, f"chair/{phase}/*.off"))[:100]

        # if phase=="train":
        #     fnames = glob.glob(os.path.join(self.root, f"chair/{phase}/*.off"))
        # else:
        #     fnames = glob.glob(os.path.join(self.root, f"bathtub/{phase}/*.off"))

        # fnames = glob.glob(os.path.join(self.root, f"*/{phase}/*.off"))
        fnames = sorted([os.path.relpath(fname, self.root) for fname in fnames])
        self.files = fnames
        assert len(self.files) > 0, "No file loaded"
        logging.info(
            f"Loading the subset {phase} from {self.root} with {len(self.files)} files"
        )
        if self.small_dataset:
            logging.info(
                f"using small dataset"
            )
        self.density = 30000

        # Ignore warnings in obj loader
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        # idx=idx%4
        if self.small_dataset:
            idx=idx%4
        mesh_file = os.path.join(self.root, self.files[idx])
        obj_cls=self.files[idx].split('/')[-3]


        preprocessed_file = mesh_file.replace('.off', '_xyz.npy')

        if self.cache and os.path.exists(preprocessed_file):
            xyz = np.load(preprocessed_file)
            feats = np.ones((len(xyz), 1))
        else:
            assert os.path.exists(mesh_file)
            pcd = o3d.io.read_triangle_mesh(mesh_file)
            vertices = np.asarray(pcd.vertices)
            vmax = vertices.max(0, keepdims=True)
            vmin = vertices.min(0, keepdims=True)
            pcd.vertices = o3d.utility.Vector3dVector((vertices - vmin) / (vmax - vmin).max())
            xyz = resample_mesh(pcd, density=self.density)
            feats = np.ones((len(xyz), 1))
            if self.cache:
                np.save(preprocessed_file, xyz)

        if self.transform:
            xyz, feats = self.transform(xyz, feats)

        # Get coords
        xyz = xyz * (self.resolution-0.01)
        coords = np.floor(xyz)
        inds = ME.utils.sparse_quantize(coords, return_index=True, return_maps_only=True)

        if len(inds) < self.min or len(inds) > self.max:
            # logging.info(
            #     f"Skipping {mesh_file}: too big or too small: {len(inds)}."
            # )
            return self.__getitem__(np.random.randint(0, len(self.files)))
        return (coords[inds], xyz[inds], idx) if not self.with_class else (coords[inds], xyz[inds], obj_cls,idx)


class ShapeNetDataset(torch.utils.data.Dataset):
    def __init__(self, transform=None, config=None,root="/mnt/data4/hjm_code_4/MinkowskiEngine-master/data/ShapeNetCore.v2_only_obj",with_class=False):
        self.files = []
        self.data_objects = []
        self.transform = transform
        self.resolution = config.resolution
        self.last_cache_percent = 0
        self.min=self.resolution**1.25
        self.max=self.resolution**2.4+50000 #3090大约支持60万
        self.root = root
        self.with_class=with_class
        self.mapping_dict = {
            '04379243': 'table',
            '02958343': 'car',
            '03001627': 'chair',
            '02691156': 'airplane',
            '04256520': 'sofa',
            '04090263': 'rifle',
            '03636649': 'lamp',
            '04530566': 'watercraft',
            '02828884': 'bench',
            '03691459': 'loudspeaker',
            '02933112': 'cabinet',
            '03211117': 'display',
            '04401088': 'telephone',
            '02924116': 'bus',
            '02808440': 'bathtub',
            '03467517': 'guitar',
            '03325088': 'faucet',
            '03046257': 'clock',
            '03991062': 'flowerpot',
            "03593526": "jar",
            "02876657": "bottle",
            "02871439": "bookshelf",
            "03642806": "laptop",
            "03624134": "knife",
            "04468005": "train",
            "02747177": "trash bin",
            "03790512": "motorbike",
            "03948459": "pistol",
            "03337140": "file cabinet",
            "02818832": "bed",
            "03928116": "piano",
            "04330267": "stove",
            "03797390": "mug",
            "02880940": "bowl",
            "04554684": "washer",
            "04004475": "printer",
            "03513137": "helmet",
            "03761084": "microwaves",
            "04225987": "skateboard",
            "04460130": "tower",
            "02942699": "camera",
            "02801938": "basket",
            "02946921": "can",
            "03938244": "pillow",
            "03710193": "mailbox",
            "03207941": "dishwasher",
            "04099429": "rocket",
            "02773838": "bag",
            "02843684": "birdhouse",
            "03261776": "earphone",
            "03759954": "microphone",
            "04074963": "remote",
            "03085013": "keyboard",
            "02834778": "bicycle",
            "02954340": "cap"
        }

        fnames = glob.glob(os.path.join(self.root, f"*/*/models/*.obj"))
        fnames = sorted([os.path.relpath(fname, self.root) for fname in fnames])
        self.files = fnames
        self.cache=config.cache
        assert len(self.files) > 0, "No file loaded"
        logging.info(
            f"Loading the data from {self.root} with {len(self.files)} files"
        )
        self.density = 30000

        # Ignore warnings in obj loader
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        mesh_file = os.path.join(self.root, self.files[idx])
        obj_cls_num=self.files[idx].split('/')[-4]
        obj_cls=self.mapping_dict[obj_cls_num]
        # print(mesh_file)
        preprocessed_file = mesh_file.replace('.obj', '_xyz.npy')

        if os.path.exists(preprocessed_file):
            xyz = np.load(preprocessed_file)
            feats = np.ones((len(xyz), 1))
        else:
            assert os.path.exists(mesh_file)
            pcd = o3d.io.read_triangle_mesh(mesh_file, enable_post_processing=False)
            vertices = np.asarray(pcd.vertices)
            vmax = vertices.max(0, keepdims=True)
            vmin = vertices.min(0, keepdims=True)
            pcd.vertices = o3d.utility.Vector3dVector((vertices - vmin) / (vmax - vmin).max())
            xyz = resample_mesh(pcd, density=self.density)
            feats = np.ones((len(xyz), 1))
            if self.cache:
                np.save(preprocessed_file, xyz)

        if self.transform:
            xyz, feats = self.transform(xyz, feats)

        # Get coords
        # print(xyz.shape)
        xyz = xyz * (self.resolution-0.01)
        coords = np.floor(xyz)
        inds = ME.utils.sparse_quantize(coords, return_index=True, return_maps_only=True)

        if len(inds) < self.min or len(inds) > self.max:
            # logging.info(
            #     f"Skipping {mesh_file}: too big or too small: {len(inds)}."
            # )
            return self.__getitem__(np.random.randint(0, len(self.files)))
        # return (coords[inds], xyz[inds], idx)
        return (coords[inds], xyz[inds], idx) if not self.with_class else (coords[inds], xyz[inds], obj_cls,idx)


class Objaverse(torch.utils.data.Dataset):
    def __init__(self, transform=None, config=None,root="/mnt/data_objaverse/objaverse/hf-objaverse-v1/glbs",image_root="/mnt/data_objaverse/Objaverse-MIX/rendered_images",with_class=False,phase="train"):
        print('init Objaverse')
        self.files = []
        # self.cache = {}
        self.data_objects = []
        self.transform = transform
        self.resolution = config.resolution
        self.last_cache_percent = 0
        self.min=self.resolution**1.25+1000
        self.max=self.resolution**2.4+50000 #3090大约支持60万
        self.cache=config.cache
        self.with_class=with_class

        self.root = root
        self.image_root=image_root
        self.image_preprocess = CLIPImageProcessor()
        if phase=="train":
            fnames = glob.glob(os.path.join(self.root, f"000-000/*.glb"))
        else:
            fnames = glob.glob(os.path.join(self.root, f"000-000/*.glb"))[:100]

        fnames = sorted([os.path.relpath(fname, self.root) for fname in fnames])
        self.files = fnames
        assert len(self.files) > 0, "No file loaded"
        logging.info(
            f"Loading the data from {self.root} with {len(self.files)} files"
        )
        self.density = 30000
        self.max_v=0
        self.max_f=0

        # Ignore warnings in obj loader
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        mesh_file = os.path.join(self.root, self.files[idx])
        # obj_cls=self.files[idx].split('/')[-3]
        obj_cls="test"
        # print(mesh_file)
        obj_image_dir =os.path.join(self.image_root,self.files[idx].split('/')[-2],self.files[idx].split('/')[-2],self.files[idx].split('/')[-1].replace(".glb",""))
        # obj_image=
        obj_images = glob.glob(os.path.join(obj_image_dir, f"*.png"))
        obj_images.sort()

        # images_path = [Image.open(image_path) for image_path in obj_images]
        # print(obj_image_dir)
        # print(len(obj_images))

        # images=[Image.open(random.choice(obj_images))]
        images=[Image.open(obj_images[0])]

        image_condition = self.image_preprocess(images=images, return_tensors="pt", do_rescale=True).pixel_values


        preprocessed_file = mesh_file.replace('.glb', '_xyz.npy')

        if os.path.exists(preprocessed_file):
            xyz = np.load(preprocessed_file)
            feats = np.ones((len(xyz), 1))
        else:
            assert os.path.exists(mesh_file)
            pcd = o3d.io.read_triangle_mesh(mesh_file)
            # o3d.visualization.draw_geometries([pcd])
            vertices = np.asarray(pcd.vertices)
            # if vertices.shape[0]>self.max_v:
            #     self.max_v=vertices.shape[0]
            #     print("now max_v: ",self.max_v)
            # faces = np.array(pcd.triangles).astype(int)
            # if faces.shape[0]>self.max_f:
            #     self.max_f=faces.shape[0]
            #     print("now max_f: ",self.max_f)
            # print(vertices.shape)
            if vertices.shape[0]==0:
                # print(f"fail load {mesh_file}, 0 vertices")
                return self.__getitem__(np.random.randint(0, len(self.files)))
            if np.array(pcd.triangles).astype(int).shape[0]>=500000:
                # print(f"fail load {mesh_file}, 0 vertices")
                return self.__getitem__(np.random.randint(0, len(self.files)))

            vmax = vertices.max(0, keepdims=True)
            vmin = vertices.min(0, keepdims=True)
            pcd.vertices = o3d.utility.Vector3dVector((vertices - vmin) / (vmax - vmin).max())

            xyz = resample_mesh(pcd, density=self.density)
            if xyz is None:
                return self.__getitem__(np.random.randint(0, len(self.files)))
            feats = np.ones((len(xyz), 1))
            if self.cache:
                np.save(preprocessed_file, xyz)

        if self.transform:
            xyz, feats = self.transform(xyz, feats)

        # Get coords
        xyz = xyz * (self.resolution-0.01)
        coords = np.floor(xyz)
        inds = ME.utils.sparse_quantize(coords, return_index=True, return_maps_only=True)

        if len(inds) < self.min or len(inds) > self.max:
            # logging.info(
            #     f"Skipping {mesh_file}: too big or too small: {len(inds)}."
            # )
            return self.__getitem__(np.random.randint(0, len(self.files)))
        return (coords[inds], xyz[inds], idx) if not self.with_class else (coords[inds], xyz[inds],obj_cls,image_condition,idx)




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


if __name__ =='__main__':
    config=configparser.ConfigParser()
    config.resolution=128
    config.cache=False
    config.max_batch_len=100000
    config.batch_size=16
    config.num_workers=0

    train_dataset=ConcatDataset([
        # ModelNet40Dataset(
        #     phase="train",
        #     transform=rotate_point_cloud,
        #     config=config,
        #     with_class=True),

        # ShapeNetDataset(
        #     transform=rotate_point_cloud,
        #     config=config,
        #     with_class=True),

        Objaverse(
            transform=rotate_point_cloud,
            config=config,
            with_class=True)
        ])

    args = {
    "batch_size": config.batch_size,
    "num_workers": config.num_workers,
    "collate_fn": collate_pointcloud_fn,
    "pin_memory": False,
    "drop_last": False,
    "shuffle": True,
    }
    train_loader = DataLoader(
        train_dataset,
        **args
    )

    train_iter = iter(train_loader)

    for data_dict in enumerate(train_iter):
        # print(data_dict)
        pass
    #     a=1
    # data_dict=next(train_iter)
    # print(data_dict)




