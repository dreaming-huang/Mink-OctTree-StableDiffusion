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
import torch
import torch.nn as nn

import MinkowskiEngine as ME
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F




class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 inplanes,
                 planes,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 bn_momentum=0.1,
                 dimension=-1,
                 time_embedding_norm='default',
                 embedding_dim=None,
                 with_attn=None,
                 attn_head_dim=8,
                 attn_layer=1,
                 ):
        super(BasicBlock, self).__init__()
        assert dimension > 0
        self.time_embedding_norm=time_embedding_norm
        self.embedding_dim=embedding_dim
        self.with_attn=with_attn

        if embedding_dim is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = nn.Linear(embedding_dim, planes)
            elif self.time_embedding_norm == "scale_shift":
                self.time_emb_proj = nn.Linear(embedding_dim, 2 * planes)

        self.conv1 = ME.MinkowskiConvolution(
            inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
        if embedding_dim is None:
            self.norm1 = ME.MinkowskiBatchNorm(planes)
        else:
            self.norm1 = ME.MinkowskiStableInstanceNorm(planes)

        if with_attn:
            self.attentions=sparseTransformer(input_dim=planes,head_dim=attn_head_dim,num_layers=attn_layer)
        self.conv2 = ME.MinkowskiConvolution(
            planes, planes, kernel_size=3, stride=1, dilation=dilation, dimension=dimension)
        if embedding_dim is None:
            self.norm2 = ME.MinkowskiBatchNorm(planes)
        else:
            # self.norm2 = ME.AdaStableInstanceNorm(planes,embedding_dim)
            self.norm2=ME.MinkowskiStableInstanceNorm(planes)

        self.relu = ME.MinkowskiELU()
        
        self.downsample = downsample

    def forward(self, x,emb=None,coordinates=None):
        residual = x

        

        out = self.conv1(x)

        if self.embedding_dim is None:
            out = self.norm1(out)
        else:
            # emb=self.relu(emb)
            # emb=F.silu(emb)
            emb=self.time_emb_proj(emb)

            if self.time_embedding_norm=="default":
                emb=emb[out.C[:,0]]
                # print(out.F.shape,emb.shape)
                out = self.norm1(out)
                out = ME.SparseTensor(
                    out.F+emb,
                    coordinate_map_key=out.coordinate_map_key,
                    coordinate_manager=out.coordinate_manager,
                )

            elif self.time_embedding_norm=="scale_shift":
                out = self.norm1(out)
                emb=emb[out.C[:,0]]
                emb=emb.view(-1,2,out.F.shape[1])
                scale,shift=emb[:,0],emb[:,1]
                out = ME.SparseTensor(
                    out.F*(1+scale)+shift,
                    coordinate_map_key=out.coordinate_map_key,
                    coordinate_manager=out.coordinate_manager,
                )
        
        out = self.relu(out)

        out = self.conv2(out)

        out = self.norm2(out)

        if self.downsample is not None:
          residual = self.downsample(x)

        out += residual
        #完成res后加attention
        if self.with_attn:
            out = self.relu(out)
            out=self.attentions(out)
        out = self.relu(out)

        return out
    


class ResNetBlock(nn.Module):
    BLOCK = BasicBlock
    LAYERS = 2
    # INIT_DIM = 64
    # PLANES = 128

    def __init__(self, in_channels, out_channels,after=None, D=3,embedding_dim=None,with_attn=False,):
        '''after应该是downsample或者upsample或者None或者upsample_determine'''
        nn.Module.__init__(self)
        self.D = D
        self.after = after
        assert self.BLOCK is not None

        self.network_initialization(in_channels, out_channels, D,embedding_dim=embedding_dim,with_attn=with_attn)

    def network_initialization(self, in_channels, out_channels, D,embedding_dim=None,with_attn=False):

        # self.inplanes = self.INIT_DIM
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.layer1 = self._make_layer(
            self.BLOCK, in_channels, self.LAYERS, stride=1,embedding_dim=embedding_dim,with_attn=with_attn
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, bn_momentum=0.1,embedding_dim=None,with_attn=False):
        layers = []

        for i in range(1, blocks):
            layers.append(
                block(
                    self.out_channels,self.out_channels,stride=stride, dilation=dilation, dimension=self.D,embedding_dim=embedding_dim,with_attn=with_attn
                )
            )
        if self.after=='downsample':
            layers.append(
                Downsample(self.in_channels,self.out_channels,self.D,norm="batch" if embedding_dim is None else "instance")
            )
        elif self.after=='upsample':
            layers.append(
                Upsample(self.in_channels,self.out_channels,self.D,norm="batch" if embedding_dim is None else "instance")
            )
        elif self.after=='upsample_determine':
            layers.append(
                Upsample_determine(self.in_channels,self.out_channels,self.D,norm="batch" if embedding_dim is None else "instance")
            )
        else:
            layers.append(
                adapt(self.in_channels,self.out_channels,self.D,norm="batch" if embedding_dim is None else "instance")
            )
        #为了对齐最后的coordinate_map_key
        # print((self.after=='upsample' or self.after=='upsample_determine') and embedding_dim is not None)
        if embedding_dim is not None:
            layers.append(
                adapt(self.out_channels,self.out_channels,self.D,norm="batch" if embedding_dim is None else "instance")
            )
        
        return nn.Sequential(*layers)
        # return nn.ModuleList(layers)

    def forward(self, x: ME.SparseTensor,emb=None,coordinates=None):
        # return self.layer1(x,emb)
        if coordinates is None:
            for layer in self.layer1:
                x=layer(x,emb)
        else:
            for i in range(len(self.layer1)):
                # if i==0:
                #     x=self.layer1[i](x,emb,coordinates=coordinates)
                if i==len(self.layer1)-1:
                    # print("aa")
                    # print(self.layer1[i])
                    x=self.layer1[i](x,emb,coordinates=coordinates)
                    # print(coordinates)
                    # print(x.coordinate_key)

                else:
                    x=self.layer1[i](x,emb)
        return x



class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, D=3,norm="batch"):
        """norm: batch or instance"""
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, stride=2, dimension=D),
            ME.MinkowskiBatchNorm(out_channels) if norm=="batch" else ME.MinkowskiStableInstanceNorm(out_channels),
            ME.MinkowskiELU(),
        )

    def forward(self, x,emb=None,coordinates=None):
        # return self.net(x)
        return self.net[2](self.net[1](self.net[0](x,coordinates=coordinates)))


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, D=3,norm="batch"):
        """norm: batch or instance"""
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            ME.MinkowskiGenerativeConvolutionTranspose(in_channels, out_channels, kernel_size=2, stride=2, dimension=D),
            # ME.MinkowskiConvolutionTranspose(in_channels, out_channels, kernel_size=2, stride=2, dimension=D),

            ME.MinkowskiBatchNorm(out_channels) if norm=="batch" else ME.MinkowskiStableInstanceNorm(out_channels),

            ME.MinkowskiELU(),

        )


    def forward(self, x,emb=None,coordinates=None):
        # return self.net(x)
        return self.net[2](self.net[1](self.net[0](x,coordinates=coordinates)))


class Upsample_determine(nn.Module):
    def __init__(self, in_channels, out_channels, D=3,norm="batch"):
        """norm: batch or instance"""
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(in_channels, out_channels, kernel_size=2, stride=2, dimension=D),
            ME.MinkowskiBatchNorm(out_channels) if norm=="batch" else ME.MinkowskiStableInstanceNorm(out_channels),

            ME.MinkowskiELU(),
        )

    def forward(self, x,emb=None,coordinates=None):
        # return self.net(x)
        return self.net[2](self.net[1](self.net[0](x,coordinates=coordinates)))

    
class adapt(nn.Module):
    def __init__(self, in_channels, out_channels, D=3,norm="batch"):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, dimension=D),
            ME.MinkowskiBatchNorm(out_channels) if norm=="batch" else ME.MinkowskiStableInstanceNorm(out_channels),

            ME.MinkowskiELU(),
        )

    def forward(self, x,emb=None,coordinates=None):
        # return self.net(x)
        # print(coordinates)
        return self.net[2](self.net[1](self.net[0](x,coordinates=coordinates)))



class ResNet2(ResNetBlock):
    BLOCK = BasicBlock
    LAYERS = 2

class ResNet3(ResNetBlock):
    BLOCK = BasicBlock
    LAYERS = 3

class LinearPositionalEncoding(nn.Module):
    def __init__(self, d_model):
        super(LinearPositionalEncoding, self).__init__()
        # self.x_embedding = nn.Embedding(max_x, d_model)
        # self.y_embedding = nn.Embedding(max_y, d_model)
        # self.z_embedding = nn.Embedding(max_z, d_model)
        # self.stride_embedding = nn.Embedding(max_stride, d_model)
        self.fc=nn.Linear(4,d_model)

        
    def forward(self, coords,stride=1):
        stride=torch.ones_like(coords[:, 0])*stride
        x, y, z = coords[:, 1], coords[:, 2], coords[:, 3]
        # x_pos = self.x_embedding(x)
        # y_pos = self.y_embedding(y)
        # z_pos = self.z_embedding(z)
        # stride_emb = self.stride_embedding(stride)
        pos=torch.stack([x,y,z,stride],dim=1).to(torch.float)
        # print(pos.dtype)
        return self.fc(pos)
        # return x_pos + y_pos + z_pos+stride_emb

class sparseTransformer(nn.Module):
    def __init__(self, input_dim, head_dim, num_layers,attention_head_dim=8):
        super(sparseTransformer, self).__init__()
        # d_model=nhead*
        nhead=input_dim//head_dim
        inner_dim=nhead*head_dim
        self.input=ME.MinkowskiConvolution(input_dim, inner_dim, kernel_size=1, dimension=3)
        self.output=ME.MinkowskiConvolution(inner_dim, input_dim, kernel_size=1, dimension=3)
        
        encoder_layer = TransformerEncoderLayer(d_model=inner_dim, nhead=nhead) #包含多头注意力块和前馈神经网络，多头注意力机制块是点积、缩放、softmax后的加权求和
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        # self.pe=LinearPositionalEncoding(inner_dim)

    def is_batch_dim_sorted(self,tensor):
        # 获取张量的batch维
        batch_dim = tensor[:, 0]
        
        # 判断是否升序排列
        is_sorted = torch.all(batch_dim[1:] >= batch_dim[:-1])
        
        return is_sorted

        

    def forward(self,sinput):
        #N,C 排序

        sinput=self.input(sinput)
        assert self.is_batch_dim_sorted(sinput.C)

        sinput_C,sinput_F=sinput.C,sinput.F
        # sinput_F=sinput_F+self.pe(sinput_C,sinput.tensor_stride[0])
        # Split the features based on batch indices,then split to windows with padding mask 把大batch分成小窗，不同实例的被分开，不够分的补齐
        coords, features = sinput_C, sinput_F
        batch_indices = coords[:, 0]
        # unique_batch_indices = torch.unique(batch_indices)
        unique_batch_indices, inverse_indices = torch.unique(batch_indices, return_inverse=True)
        length_max=torch.max(torch.bincount(batch_indices))
        instances = []
        instance_masks = []

        #siplt batch into instances
        #方法1 mask取数值拼接
        # for batch_idx in unique_batch_indices:
        #     batch_mask = (batch_indices == batch_idx)
        #     instance = features[batch_mask]
        #     # batch_coords = coords[batch_mask]
        #     instance_mask = torch.ones(length_max, dtype=torch.bool)
        #     instance_mask[instance.shape[0]:] = False
        #     if instance.shape[0] < length_max:
        #         pad_size = length_max - instance.shape[0]
        #         instance = torch.cat([instance, torch.zeros(pad_size, instance.shape[1]).to(instance.device)], dim=0)
        #     instances.append(instance)
        #     instance_masks.append(instance_mask)
        # instances = torch.stack(instances, dim=0)#n/instance_size,instance_size,C
        # instance_masks = torch.stack(instance_masks, dim=0).to(instances.device)


        # 方法2 预分配内存
        num_batches = unique_batch_indices.shape[0]
        num_features = features.shape[1]
        instances = torch.zeros((num_batches, length_max, num_features), device=features.device)
        instance_masks = torch.zeros((num_batches, length_max), dtype=torch.bool, device=features.device)

        # 获取每个 batch 内的实例数量
        counts = torch.bincount(inverse_indices)

        # 使用高级索引操作来填充
        index_offsets = torch.cumsum(counts, dim=0) - counts

        for i, (offset, count) in enumerate(zip(index_offsets, counts)):
            instances[i, :count] = features[offset:offset + count]
            instance_masks[i, :count] = True

        transformer_output = self.transformer_encoder(instances.permute(1,0,2), src_key_padding_mask=~instance_masks)#mask取反，TransformerEncoder中mask为true才忽略

        #把窗口结果去除padding并拼接回大batch
        batch_output=transformer_output.permute(1,0,2)

        #方法1
        # final_outputs = []
        # for i in range(batch_output.shape[0]):
        #     output = batch_output[i]
        #     final_outputs.append(output[instance_masks[i]])

        # output=torch.cat(final_outputs, dim=0)

        #方法2
        flat_output = batch_output.reshape(-1, batch_output.shape[-1])
        flat_masks = instance_masks.reshape(-1)
        output = flat_output[flat_masks]

        out = ME.SparseTensor(
            features=output,
            coordinate_map_key=sinput.coordinate_map_key,
            tensor_stride=sinput.tensor_stride,
            coordinate_manager=sinput.coordinate_manager,
        )
        out=self.output(out)
        return out