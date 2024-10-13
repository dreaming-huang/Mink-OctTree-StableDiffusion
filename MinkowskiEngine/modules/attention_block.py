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
from typing import Callable, Optional, Union
import torch
import torch.nn as nn

import MinkowskiEngine as ME
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from torch.nn import MultiheadAttention
from torch import Tensor

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

class sparseAttention(nn.Module):
    def __init__(self,d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.attn=  MultiheadAttention(d_model, nhead, dropout=dropout,
                                            bias=bias, batch_first=batch_first,
                                            **factory_kwargs)
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           need_weights=False, is_causal=is_causal)[0]
    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class sparseTransformer(nn.Module):
    def __init__(self, input_dim, attention_head_dim, num_layers,res_attn=True):
        super(sparseTransformer, self).__init__()
        # d_model=nhead*
        nhead=input_dim//attention_head_dim
        inner_dim=nhead*attention_head_dim
        self.res_attn=res_attn
        self.input=ME.MinkowskiConvolution(input_dim, inner_dim, kernel_size=1, dimension=3)
        self.output=ME.MinkowskiConvolution(inner_dim, input_dim, kernel_size=1, dimension=3)
        
        encoder_layer = TransformerEncoderLayer(d_model=inner_dim,dim_feedforward=inner_dim,nhead=nhead,activation=F.silu) #包含多头注意力块和前馈神经网络，多头注意力机制块是点积、缩放、softmax后的加权求和
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
        # print(instances.shape) #B,L,C
        # 输入序列的形状：(sequence length, batch size, feature size)
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


        if self.res_attn:
            output=output+sinput_F

        out = ME.SparseTensor(
            features=output,
            coordinate_map_key=sinput.coordinate_map_key,
            tensor_stride=sinput.tensor_stride,
            coordinate_manager=sinput.coordinate_manager,
        )
        out=self.output(out)
        return out