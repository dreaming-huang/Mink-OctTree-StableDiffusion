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
from typing import Callable, Optional, Union

import MinkowskiEngine as ME
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import torch.nn.functional as F
from torch import Tensor
from torch.nn import MultiheadAttention



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
                 group=1,
                 with_cross_attn=None,
                 cross_attention_dim=768,
                 ):
        super(BasicBlock, self).__init__()
        assert dimension > 0
        self.time_embedding_norm=time_embedding_norm
        self.embedding_dim=embedding_dim
        self.with_attn=with_attn
        self.with_cross_attn=with_cross_attn
        if embedding_dim is not None:
            if self.time_embedding_norm == "default":
                self.time_emb_proj = nn.Linear(embedding_dim, planes)
            elif self.time_embedding_norm == "scale_shift":
                self.time_emb_proj = nn.Linear(embedding_dim, 2 * planes)

        # self.conv1 = ME.MinkowskiConvolution(
        #     inplanes, planes, kernel_size=3, stride=stride, dilation=dilation, dimension=dimension)
        self.conv1=nn.Conv3d(inplanes,planes,kernel_size=3,stride=stride,padding=1,bias=False,padding_mode="replicate")
        # if embedding_dim is None:
        #     self.norm1 = ME.MinkowskiBatchNorm(planes)
        # else:
        #     self.norm1 = ME.MinkowskiStableInstanceNorm(planes,group=group)
        if embedding_dim is None:
            self.norm1 = nn.BatchNorm3d(planes)
        else:
            # self.norm1 = nn.GroupNorm(group,inplanes) if group<inplanes else nn.GroupNorm(inplanes,inplanes)
            # self.norm1 = nn.GroupNorm(planes//group,planes,affine=False) #第一个参数是分为几组
            self.norm1 =ME.HjmGroupNorm(planes//group,planes)


        if with_attn:
            self.attentions=denseTransformer(input_dim=planes,attention_head_dim=attn_head_dim,num_layers=attn_layer)
            if with_cross_attn:
                self.cross_attention=denseTransformer(input_dim=planes,attention_head_dim=attn_head_dim,num_layers=attn_layer,cross_attention_dim=cross_attention_dim,is_cross_attn=True)

        # self.conv2 = ME.MinkowskiConvolution(
        #     planes, planes, kernel_size=3, stride=1, dilation=dilation, dimension=dimension)
        self.conv2=nn.Conv3d(planes,planes,kernel_size=3,stride=1,padding=1,bias=False,padding_mode="replicate")
        # if embedding_dim is None:
        #     self.norm2 = ME.MinkowskiBatchNorm(planes)
        # else:
        #     # self.norm2 = ME.AdaStableInstanceNorm(planes,embedding_dim)
        #     self.norm2=ME.MinkowskiStableInstanceNorm(planes,group=group)
        if embedding_dim is None:
            self.norm2 = nn.BatchNorm3d(planes)
        else:
            # self.norm2 = nn.GroupNorm(group,planes) if group<planes else nn.GroupNorm(planes,planes)
            # self.norm2 = nn.GroupNorm(planes//group,planes,affine=False)
            self.norm2 =ME.HjmGroupNorm(planes//group,planes)



        # self.act = ME.MinkowskiELU()
        # self.act = ME.MinkowskiSiLU()
        # self.act = nn.SiLU()
        self.act = nn.ELU()


        
        self.downsample = downsample

    def forward(self, x,emb=None,encoder_hidden_state=None,coordinates=None):
        residual = x

        # out = self.norm1(x)
        # self.act(out)
        out = self.conv1(x)

        if self.embedding_dim is None:
            out = self.norm1(out)
            pass
        else:
            emb=self.act(emb)
            # emb=F.silu(emb)
            # emb=F.elu(emb)
            emb=self.time_emb_proj(emb)

            if self.time_embedding_norm=="default":
                # emb=emb[out.C[:,0]]
                # print(out.F.shape,emb.shape)
                # print(out.shape,emb.shape)
                
                out = self.norm1(out)+emb.unsqueeze(2).unsqueeze(3).unsqueeze(4) 
                # out=out+emb.unsqueeze(2).unsqueeze(3).unsqueeze(4)
                
                # out = ME.SparseTensor(
                #     out.F+emb,
                #     coordinate_map_key=out.coordinate_map_key,
                #     coordinate_manager=out.coordinate_manager,
                # )

            # elif self.time_embedding_norm=="scale_shift":
            #     out = self.norm1(out)
            #     emb=emb[out.C[:,0]]
            #     emb=emb.view(-1,2,out.F.shape[1])
            #     scale,shift=emb[:,0],emb[:,1]
            #     out = ME.SparseTensor(
            #         out.F*(1+scale)+shift,
            #         coordinate_map_key=out.coordinate_map_key,
            #         coordinate_manager=out.coordinate_manager,
            #     )
        
        out = self.act(out)

        # out = self.norm2(out)
        # out = self.act(out)
        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
          residual = self.downsample(x)

        out += residual

        # batch_indices = out.C[:, 0]
        # # unique_batch_indices = torch.unique(batch_indices)
        # unique_batch_indices, inverse_indices = torch.unique(batch_indices, return_inverse=True)
        # # print(unique_batch_indices)

        # #完成res后加attention
        if self.with_attn:
            out = self.act(out)
            out=self.attentions(out)
            if self.with_cross_attn:
                out = self.act(out)
                out=self.cross_attention(out,encoder_hidden_state=encoder_hidden_state)

        out = self.act(out)

        return out
    


class ResNetBlock(nn.Module):
    BLOCK = BasicBlock
    LAYERS = 2
    # INIT_DIM = 64
    # PLANES = 128

    def __init__(self, in_channels, out_channels,after=None, D=3,embedding_dim=None,with_attn=False,use_conv=True,time_embedding_norm='default',group=1,with_cross_attn=None,cross_attention_dim=768,):
        '''after应该是downsample或者upsample或者None或者upsample_determine'''
        nn.Module.__init__(self)
        self.D = D
        self.after = after
        assert self.BLOCK is not None

        self.network_initialization(in_channels, out_channels, D,embedding_dim=embedding_dim,with_attn=with_attn,use_conv=use_conv,time_embedding_norm=time_embedding_norm,group=group,with_cross_attn=with_cross_attn,cross_attention_dim=cross_attention_dim,)

    def network_initialization(self, in_channels, out_channels, D,embedding_dim=None,with_attn=False,use_conv=True,time_embedding_norm='default',group=1,with_cross_attn=None,cross_attention_dim=768,):

        # self.inplanes = self.INIT_DIM
        self.in_channels=in_channels
        self.out_channels=out_channels
        self.layer1 = self._make_layer(
            self.BLOCK, in_channels, self.LAYERS, stride=1,embedding_dim=embedding_dim,with_attn=with_attn,use_conv=use_conv,time_embedding_norm=time_embedding_norm,group=group,with_cross_attn=with_cross_attn,cross_attention_dim=cross_attention_dim
        )

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1, bn_momentum=0.1,embedding_dim=None,with_attn=False,use_conv=True,time_embedding_norm='default',group=1,with_cross_attn=None,cross_attention_dim=768,):
        layers = []
        if use_conv:
            if self.after=='downsample':
                layers.append(
                    Downsample(self.in_channels,self.out_channels,self.D,norm="batch" if embedding_dim is None else "instance",group=group)
                )
            elif self.after=='upsample':
                layers.append(
                    Upsample(self.in_channels,self.out_channels,self.D,norm="batch" if embedding_dim is None else "instance",group=group)
                )
            elif self.after=='upsample_determine':
                layers.append(
                    Upsample_determine(self.in_channels,self.out_channels,self.D,norm="batch" if embedding_dim is None else "instance",group=group)
                )
            else:
                layers.append(
                    adapt(self.in_channels,self.out_channels,self.D,norm="batch" if embedding_dim is None else "instance",group=group)
                )
        else:
            layers.append(
                adapt(self.in_channels,self.out_channels,self.D,norm="batch" if embedding_dim is None else "instance",group=group,)
            )



        for i in range(1, blocks):
            layers.append(
                block(
                    self.out_channels,self.out_channels,stride=stride, dilation=dilation, dimension=self.D,embedding_dim=embedding_dim,with_attn=with_attn,time_embedding_norm=time_embedding_norm,group=group,with_cross_attn=with_cross_attn,cross_attention_dim=cross_attention_dim
                )
            )

        # if not use_conv:
        #     if self.after=='downsample':
        #         layers.append(
        #             avg_pool(2,2,self.D)
        #         )
        #     elif self.after=='upsample' or self.after=='upsample_determine':
        #         layers.append(
        #             Upsample_interpolate(self.out_channels,self.out_channels,self.D)
        #         )
                
        
        #为了对齐最后的coordinate_map_key
        # print((self.after=='upsample' or self.after=='upsample_determine') and embedding_dim is not None)
        if embedding_dim is not None:
            layers.append(
                adapt(self.out_channels,self.out_channels,self.D,norm="batch" if embedding_dim is None else "instance",group=group)
            )
        
        return nn.Sequential(*layers)
        # return nn.ModuleList(layers)

    def forward(self, x ,emb=None,coordinates=None,encoder_hidden_state=None):
        # import ipdb;ipdb.set_trace()
        # return self.layer1(x,emb)
        # print("aaaaaaaaaa")
        if coordinates is None:
            for layer in self.layer1:
                # print(layer)
                x=layer(x,emb,encoder_hidden_state=encoder_hidden_state)
        else:
            for i in range(len(self.layer1)):


                
                # print(i)
                # if i==0:
                #     x=self.layer1[i](x,emb,coordinates=coordinates)
                if i==len(self.layer1)-1:
                    # print("aa")
                    # print(self.layer1[i])
                    x=self.layer1[i](x,emb,coordinates=coordinates,encoder_hidden_state=encoder_hidden_state)
                    # print(coordinates)
                    # print(x.coordinate_key)


                else:
                    x=self.layer1[i](x,emb,encoder_hidden_state=encoder_hidden_state)


        return x



class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, D=3,norm="batch",group=1):
        """norm: batch or instance"""
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            # ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, stride=2, dimension=D),
            # ME.MinkowskiBatchNorm(out_channels) if norm=="batch" else ME.MinkowskiStableInstanceNorm(out_channels,group=group),
            # nn.GroupNorm(group,in_channels) if group<in_channels else nn.GroupNorm(in_channels,in_channels),
            nn.Conv3d(in_channels,out_channels,kernel_size=3,stride=2,padding=1,bias=False,padding_mode="replicate"),
            
            # nn.GroupNorm(out_channels//group,out_channels,affine=False),
            ME.HjmGroupNorm(out_channels//group,out_channels),

            # ME.MinkowskiELU(),
            # ME.MinkowskiSiLU(),
            nn.ELU(),
            # nn.SiLU(),

        )

    def forward(self, x,emb=None,coordinates=None,**kargs):
        # return self.net(x)
        # print(x.shape)
        # print(x[1].shape)
        return self.net[2](self.net[1](self.net[0](x)))


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, D=3,norm="batch",group=1):
        """norm: batch or instance"""
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            # ME.MinkowskiGenerativeConvolutionTranspose(in_channels, out_channels, kernel_size=2, stride=2, dimension=D),
            # ME.MinkowskiConvolutionTranspose(in_channels, out_channels, kernel_size=2, stride=2, dimension=D),
            # ME.MinkowskiBatchNorm(out_channels) if norm=="batch" else ME.MinkowskiStableInstanceNorm(out_channels,group=group),
            # nn.GroupNorm(group,in_channels) if group<in_channels else nn.GroupNorm(in_channels,in_channels),
            nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2,bias=False),
            
            # nn.GroupNorm(out_channels//group,out_channels,affine=False),
            ME.HjmGroupNorm(out_channels//group,out_channels),


            # ME.MinkowskiELU(),
            # ME.MinkowskiSiLU(),
            # nn.SiLU(),
            nn.ELU(),




        )


    def forward(self, x,emb=None,coordinates=None,**kargs):
        # return self.net(x)
        return self.net[2](self.net[1](self.net[0](x)))


class Upsample_determine(nn.Module):
    def __init__(self, in_channels, out_channels, D=3,norm="batch",group=1):
        """norm: batch or instance"""
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(in_channels, out_channels, kernel_size=2, stride=2, dimension=D),
            ME.MinkowskiBatchNorm(out_channels) if norm=="batch" else ME.MinkowskiStableInstanceNorm(out_channels,group=group),

            # ME.MinkowskiELU(),
            ME.MinkowskiSiLU(),
        )

    def forward(self, x,emb=None,coordinates=None,**kargs):
        # return self.net(x)
        return self.net[2](self.net[1](self.net[0](x)))

    
class adapt(nn.Module):
    def __init__(self, in_channels, out_channels, D=3,norm="batch",group=1):
        nn.Module.__init__(self)
        self.net = nn.Sequential(
            # ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, dimension=D),
            # ME.MinkowskiBatchNorm(out_channels) if norm=="batch" else ME.MinkowskiStableInstanceNorm(out_channels,group=group),
            # nn.GroupNorm(group,in_channels) if group<in_channels else nn.GroupNorm(in_channels,in_channels),
            nn.Conv3d(in_channels,out_channels,kernel_size=3,padding=1,bias=False,padding_mode="replicate"),
            
            # nn.GroupNorm(out_channels//group,out_channels,affine=False),
            ME.HjmGroupNorm(out_channels//group,out_channels),


            # ME.MinkowskiELU(),
            # ME.MinkowskiSiLU(),
            # nn.SiLU(),
            nn.ELU(),



        )

    def forward(self, x,emb=None,coordinates=None,**kargs):
        # return self.net(x)
        # print(coordinates)
        return self.net[2](self.net[1](self.net[0](x)))

# class avg_pool(nn.Module):
#     def __init__(self, kernel_size=2,stride=2, D=3):
#         nn.Module.__init__(self)
#         self.net = ME.MinkowskiAvgPooling(kernel_size=kernel_size,stride=stride,dimension=D)
#     def forward(self, x,emb=None,coordinates=None,**kargs):
#         return self.net(x)

# class pool_transpose(nn.Module):
#     def __init__(self, kernel_size=2,stride=2, D=3):
#         nn.Module.__init__(self)
#         self.net = ME.MinkowskiPoolingTranspose(kernel_size=kernel_size,stride=stride,dimension=D)
#     def forward(self, x,emb=None,coordinates=None,**kargs):
#         return self.net(x)

# class Upsample_interpolate(nn.Module):
#     def __init__(self, in_channels, out_channels, D=3,norm="batch"):
#         nn.Module.__init__(self)
#         self.net = ME.MinkowskiUpsampleInterpolate(in_channels, out_channels, kernel_size=2, stride=2, dimension=D)
#     def forward(self, x,emb=None,coordinates=None,**kargs):
#         assert (torch.all(x.F[0]==self.net(x).F[0]))
#         return self.net(x)




class ResNet2(ResNetBlock):
    BLOCK = BasicBlock
    LAYERS = 2

class ResNet3(ResNetBlock):
    BLOCK = BasicBlock
    LAYERS = 3

# class LinearPositionalEncoding(nn.Module):
#     def __init__(self, d_model):
#         super(LinearPositionalEncoding, self).__init__()
#         # self.x_embedding = nn.Embedding(max_x, d_model)
#         # self.y_embedding = nn.Embedding(max_y, d_model)
#         # self.z_embedding = nn.Embedding(max_z, d_model)
#         # self.stride_embedding = nn.Embedding(max_stride, d_model)
#         self.fc=nn.Linear(4,d_model)

        
#     def forward(self, coords,stride=1):
#         stride=torch.ones_like(coords[:, 0])*stride
#         x, y, z = coords[:, 1], coords[:, 2], coords[:, 3]
#         # x_pos = self.x_embedding(x)
#         # y_pos = self.y_embedding(y)
#         # z_pos = self.z_embedding(z)
#         # stride_emb = self.stride_embedding(stride)
#         pos=torch.stack([x,y,z,stride],dim=1).to(torch.float)
#         # print(pos.dtype)
#         return self.fc(pos)
#         # return x_pos + y_pos + z_pos+stride_emb


# class sparseAttention(nn.Module):
#     # def __init__(self,d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
#     #              activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
#     #              layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
#     #              bias: bool = True, device=None, dtype=None) -> None:
#     def __init__(self,query_dim: int, nhead: int, out_dim: int = None,dim_head:int=8, is_cross_attn=False,cross_attention_dim:int=None,
#                  dropout: float = 0.1,residual_connection=True,
#                  activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
#                  layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
#                  bias: bool = False, out_bias=True,device=None, dtype=None) -> None:
#         super(sparseAttention, self).__init__()

#         self.inner_dim =out_dim if out_dim is not None else query_dim
#         self.query_dim = query_dim
#         self.use_bias = bias
#         self.is_cross_attn = is_cross_attn
#         if is_cross_attn:
#             self.cross_attention_dim = cross_attention_dim
#         else:
#             self.cross_attention_dim = query_dim

#         self.residual_connection = residual_connection
#         self.out_dim = out_dim if out_dim is not None else query_dim


#         factory_kwargs = {'device': device, 'dtype': dtype}
#         self.attn=  MultiheadAttention(self.inner_dim, nhead, dropout=dropout,
#                                             bias=bias, batch_first=batch_first,
#                                             **factory_kwargs)
#         self.to_q=nn.Linear(self.query_dim,self.inner_dim,bias=bias)
#         self.to_kv=nn.Linear(self.cross_attention_dim,2*self.inner_dim,bias=bias)
#         self.to_out=nn.Linear(self.inner_dim,self.out_dim,bias=out_bias)
#         self.norm1 = nn.LayerNorm(self.inner_dim, eps=layer_norm_eps)
#         # self.act=F.elu
#         self.act=F.silu

#         # self.norm2 = nn.LayerNorm(self.inner_dim, eps=layer_norm_eps, bias=bias, **factory_kwargs)


        
#     def forward(
#             self,
#             src: Tensor,
#             src_mask: Optional[Tensor] = None,
#             src_key_padding_mask: Optional[Tensor] = None,
#             is_causal: bool = False,
#             encoder_hidden_state=None) -> Tensor:
        
#         residual=src
#         query=self.to_q(src)
#         if self.is_cross_attn and encoder_hidden_state is not None:
#             encoder_hidden_state=encoder_hidden_state
#         else:
#             encoder_hidden_state=src
        
#         key,value=self.to_kv(encoder_hidden_state).chunk(2,dim=-1)
#         # print(query.shape,key.shape)

#         src_key_padding_mask = F._canonical_mask(
#             mask=src_key_padding_mask,
#             mask_name="src_key_padding_mask",
#             other_type=F._none_or_dtype(src_mask),
#             other_name="src_mask",
#             target_type=src.dtype
#         )

#         src_mask = F._canonical_mask(
#             mask=src_mask,
#             mask_name="src_mask",
#             other_type=None,
#             other_name="",
#             target_type=src.dtype,
#             check_other=False,
#         )

#         # print(query.shape,key.shape,value.shape)
#         #attn_mask是给query的，指定哪些需要attention，key_padding_mask是给key的，指定哪些key是不需要的
#         if self.is_cross_attn:
#             x = self.attn(query, key, value,
#                 attn_mask=src_mask,
#                 # key_padding_mask=src_key_padding_mask,
#                 need_weights=False, is_causal=is_causal)[0]
#         else:
#             x = self.attn(query, key, value,
#                         attn_mask=src_mask,
#                         key_padding_mask=src_key_padding_mask,
#                         need_weights=False, is_causal=is_causal)[0]
#         # x=self.act(x)
#         x=self.norm1(x)
#         x=self.act(x)
#         x=self.to_out(x)
#         # x=self.act(x)
        
#         # print(x)
#         if self.residual_connection:
#             x=x+residual
#         return x
        
        
#     # def _sa_block(self, x: Tensor,
#     #               attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
#     #     q,k,v=self.to_qkv(x).chunk(3,dim=-1)
#     #     x = self.attn(q, k, v,
#     #                        attn_mask=attn_mask,
#     #                        key_padding_mask=key_padding_mask,
#     #                        need_weights=False, is_causal=is_causal)[0]
#     #     return x
#     # feed forward block
#     # def _ff_block(self, x: Tensor) -> Tensor:
#         # x = self.linear2(self.dropout(self.activation(self.linear1(x))))
#         # return self.dropout2(x)

class denseAttention(nn.Module):
    def __init__(self,query_dim: int, nhead: int, out_dim: int = None,dim_head:int=8, is_cross_attn=False,cross_attention_dim:int=None,
                 dropout: float = 0.1,residual_connection=True,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = False, out_bias=True,device=None, dtype=None) -> None:
        super(denseAttention, self).__init__()
        self.inner_dim =out_dim if out_dim is not None else query_dim
        self.query_dim = query_dim
        self.use_bias = bias
        self.is_cross_attn = is_cross_attn
        if is_cross_attn:
            self.cross_attention_dim = cross_attention_dim
        else:
            self.cross_attention_dim = query_dim

        self.residual_connection = residual_connection
        self.out_dim = out_dim if out_dim is not None else query_dim


        factory_kwargs = {'device': device, 'dtype': dtype}
        self.attn=  MultiheadAttention(self.inner_dim, nhead, dropout=dropout,
                                            bias=bias, batch_first=batch_first,
                                            **factory_kwargs)
        self.to_q=nn.Linear(self.query_dim,self.inner_dim,bias=bias)
        self.to_kv=nn.Linear(self.cross_attention_dim,2*self.inner_dim,bias=bias)
        self.to_out=nn.Linear(self.inner_dim,self.out_dim,bias=out_bias)
        self.norm1 = nn.LayerNorm(self.inner_dim, eps=layer_norm_eps)
        # self.act=F.elu
        self.act=F.silu

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        is_causal: bool = False,
        encoder_hidden_state=None) -> Tensor:
        
        residual=src
        query=self.to_q(src)
        if self.is_cross_attn and encoder_hidden_state is not None:
            encoder_hidden_state=encoder_hidden_state
        else:
            encoder_hidden_state=src
        
        key,value=self.to_kv(encoder_hidden_state).chunk(2,dim=-1)
        # print(query.shape,key.shape)

        src_key_padding_mask = F._canonical_mask(
            mask=src_key_padding_mask,
            mask_name="src_key_padding_mask",
            other_type=F._none_or_dtype(src_mask),
            other_name="src_mask",
            target_type=src.dtype
        )

        src_mask = F._canonical_mask(
            mask=src_mask,
            mask_name="src_mask",
            other_type=None,
            other_name="",
            target_type=src.dtype,
            check_other=False,
        )

        # print(query.shape,key.shape,value.shape)
        #attn_mask是给query的，指定哪些需要attention，key_padding_mask是给key的，指定哪些key是不需要的
        if self.is_cross_attn:
            x = self.attn(query, key, value,
                attn_mask=src_mask,
                # key_padding_mask=src_key_padding_mask,
                need_weights=False, is_causal=is_causal)[0]
        else:
            x = self.attn(query, key, value,
                        attn_mask=src_mask,
                        key_padding_mask=src_key_padding_mask,
                        need_weights=False, is_causal=is_causal)[0]
        # x=self.act(x)
        x=self.norm1(x)
        x=self.act(x)
        x=self.to_out(x)
        # x=self.act(x)
        
        # print(x)
        if self.residual_connection:
            x=x+residual
        return x
        



# class sparseTransformer(nn.Module):
#     def __init__(self, input_dim, attention_head_dim, num_layers,res_attn=False,cross_attention_dim=None,is_cross_attn=False):
#         super(sparseTransformer, self).__init__()
#         # d_model=nhead*

#         nhead=input_dim//attention_head_dim
#         inner_dim=nhead*attention_head_dim

#         # nhead=1
#         inner_dim=input_dim
#         self.res_attn=res_attn
#         # self.input=ME.MinkowskiConvolution(input_dim, inner_dim, kernel_size=3, dimension=3)
#         # self.output=ME.MinkowskiConvolution(inner_dim, input_dim, kernel_size=3, dimension=3)
        
        
#         # encoder_layer = TransformerEncoderLayer(d_model=inner_dim,nhead=nhead,activation=F.silu) #包含多头注意力块和前馈神经网络，多头注意力机制块是点积、缩放、softmax后的加权求和
#         # encoder_layer = TransformerEncoderLayer(d_model=input_dim,dim_feedforward=inner_dim,nhead=nhead,activation=F.silu,dropout=0,bias=False) #包含多头注意力块和前馈神经网络，多头注意力机制块是点积、缩放、softmax后的加权求和
#         # self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.transformer_encoder=sparseAttention(query_dim=inner_dim,nhead=nhead,out_dim=inner_dim,residual_connection=True,is_cross_attn=is_cross_attn,cross_attention_dim=cross_attention_dim)

#         # self.transformer_encoder = TransformerEncoderLayer(d_model=input_dim,dim_feedforward=inner_dim,nhead=nhead,activation=F.silu,dropout=0,bias=False) #包含多头注意力块和前馈神经网络，多头注意力机制块是点积、缩放、softmax后的加权求和

#         # self.pe=LinearPositionalEncoding(inner_dim)

#     def is_batch_dim_sorted(self,tensor):
#         # 获取张量的batch维
#         batch_dim = tensor[:, 0]
        
#         # 判断是否升序排列
#         is_sorted = torch.all(batch_dim[1:] >= batch_dim[:-1])
        
#         return is_sorted

        

#     def forward(self,sinput,encoder_hidden_state=None):
#         #N,C 排序

#         # sinput=self.input(sinput)
#         assert self.is_batch_dim_sorted(sinput.C)


#         sinput_C,sinput_F=sinput.C,sinput.F
#         # sinput_F=sinput_F+self.pe(sinput_C,sinput.tensor_stride[0])
#         # Split the features based on batch indices,then split to windows with padding mask 把大batch分成小窗，不同实例的被分开，不够分的补齐
#         coords, features = sinput_C, sinput_F
#         batch_indices = coords[:, 0]
#         # unique_batch_indices = torch.unique(batch_indices)
#         unique_batch_indices, inverse_indices = torch.unique(batch_indices, return_inverse=True)
#         # print(unique_batch_indices)
#         length_max=torch.max(torch.bincount(batch_indices))
#         instances = []
#         instance_masks = []

#         #siplt batch into instances
#         #方法1 mask取数值拼接
#         # for batch_idx in unique_batch_indices:
#         #     batch_mask = (batch_indices == batch_idx)
#         #     instance = features[batch_mask]
#         #     # batch_coords = coords[batch_mask]
#         #     instance_mask = torch.ones(length_max, dtype=torch.bool)
#         #     instance_mask[instance.shape[0]:] = False
#         #     if instance.shape[0] < length_max:
#         #         pad_size = length_max - instance.shape[0]
#         #         instance = torch.cat([instance, torch.zeros(pad_size, instance.shape[1]).to(instance.device)], dim=0)
#         #     instances.append(instance)
#         #     instance_masks.append(instance_mask)
#         # instances = torch.stack(instances, dim=0)#n/instance_size,instance_size,C
#         # instance_masks = torch.stack(instance_masks, dim=0).to(instances.device)


#         # 方法2 预分配内存
#         num_batches = unique_batch_indices.shape[0]
#         num_features = features.shape[1]
#         instances = torch.zeros((num_batches, length_max, num_features), device=features.device)
#         instance_masks = torch.zeros((num_batches, length_max), dtype=torch.bool, device=features.device)

#         # 获取每个 batch 内的实例数量
#         counts = torch.bincount(inverse_indices)

#         # 使用高级索引操作来填充
#         # print(counts)#[81, 64, 83, 92]
#         index_offsets = torch.cumsum(counts, dim=0) - counts #所有的加起来是当前的结尾，减去当前长度就是开始的偏移量
#         # print(index_offsets)#[  0,  81, 145, 228]
#         for i, (offset, count) in enumerate(zip(index_offsets, counts)):
#             instances[i, :count] = features[offset:offset + count]
#             instance_masks[i, :count] = True
#         # print(instances.shape) #B,L,C
#         # 输入序列的形状：(sequence length, batch size, feature size)
#         # import ipdb;ipdb.set_trace()
#         # print(instances)

#         if encoder_hidden_state is not None:
#             encoder_hidden_state=encoder_hidden_state.permute(1,0,2) 
#         transformer_output = self.transformer_encoder(instances.permute(1,0,2), src_key_padding_mask=~instance_masks,encoder_hidden_state=encoder_hidden_state)#mask取反，TransformerEncoder中mask为true才忽略

#         #把窗口结果去除padding并拼接回大batch
#         batch_output=transformer_output.permute(1,0,2)
#         # batch_output=instances

#         #方法1
#         # final_outputs = []
#         # for i in range(batch_output.shape[0]):
#         #     output = batch_output[i]
#         #     final_outputs.append(output[instance_masks[i]])

#         # output=torch.cat(final_outputs, dim=0)

#         #方法2
#         flat_output = batch_output.reshape(-1, batch_output.shape[-1])
#         flat_masks = instance_masks.reshape(-1)
#         # print(flat_masks.shape)
#         output = flat_output[flat_masks]
#         # print(torch.all(output==sinput_F))
#         # assert torch.all(output==sinput_F)

#         # if self.res_attn:
#         #     output=output+sinput_F

#         out = ME.SparseTensor(
#             features=output,
#             coordinate_map_key=sinput.coordinate_map_key,
#             tensor_stride=sinput.tensor_stride,
#             coordinate_manager=sinput.coordinate_manager,
#         )
#         # out=self.output(out)
#         return out

class denseTransformer(nn.Module):
    def __init__(self, input_dim, attention_head_dim, num_layers,res_attn=False,cross_attention_dim=None,is_cross_attn=False):
        super(denseTransformer, self).__init__()
        # d_model=nhead*

        nhead=input_dim//attention_head_dim
        inner_dim=nhead*attention_head_dim

        # nhead=1
        inner_dim=input_dim
        self.res_attn=res_attn

        self.transformer_encoder=denseAttention(query_dim=inner_dim,nhead=nhead,out_dim=inner_dim,residual_connection=True,is_cross_attn=is_cross_attn,cross_attention_dim=cross_attention_dim)

    def forward(self,sinput,encoder_hidden_state=None):
        #N,C 排序

        # sinput=self.input(sinput)
        # assert self.is_batch_dim_sorted(sinput.C)


        # sinput_C,sinput_F=sinput.C,sinput.F
        # # sinput_F=sinput_F+self.pe(sinput_C,sinput.tensor_stride[0])
        # # Split the features based on batch indices,then split to windows with padding mask 把大batch分成小窗，不同实例的被分开，不够分的补齐
        # coords, features = sinput_C, sinput_F
        # batch_indices = coords[:, 0]
        # # unique_batch_indices = torch.unique(batch_indices)
        # unique_batch_indices, inverse_indices = torch.unique(batch_indices, return_inverse=True)
        # # print(unique_batch_indices)
        # length_max=torch.max(torch.bincount(batch_indices))
        # instances = []
        # instance_masks = []

        # #siplt batch into instances
        # #方法1 mask取数值拼接
        # # for batch_idx in unique_batch_indices:
        # #     batch_mask = (batch_indices == batch_idx)
        # #     instance = features[batch_mask]
        # #     # batch_coords = coords[batch_mask]
        # #     instance_mask = torch.ones(length_max, dtype=torch.bool)
        # #     instance_mask[instance.shape[0]:] = False
        # #     if instance.shape[0] < length_max:
        # #         pad_size = length_max - instance.shape[0]
        # #         instance = torch.cat([instance, torch.zeros(pad_size, instance.shape[1]).to(instance.device)], dim=0)
        # #     instances.append(instance)
        # #     instance_masks.append(instance_mask)
        # # instances = torch.stack(instances, dim=0)#n/instance_size,instance_size,C
        # # instance_masks = torch.stack(instance_masks, dim=0).to(instances.device)


        # # 方法2 预分配内存
        # num_batches = unique_batch_indices.shape[0]
        # num_features = features.shape[1]
        # instances = torch.zeros((num_batches, length_max, num_features), device=features.device)
        # instance_masks = torch.zeros((num_batches, length_max), dtype=torch.bool, device=features.device)

        # # 获取每个 batch 内的实例数量
        # counts = torch.bincount(inverse_indices)

        # # 使用高级索引操作来填充
        # # print(counts)#[81, 64, 83, 92]
        # index_offsets = torch.cumsum(counts, dim=0) - counts #所有的加起来是当前的结尾，减去当前长度就是开始的偏移量
        # # print(index_offsets)#[  0,  81, 145, 228]
        # for i, (offset, count) in enumerate(zip(index_offsets, counts)):
        #     instances[i, :count] = features[offset:offset + count]
        #     instance_masks[i, :count] = True
        # print(instances.shape) #B,L,C
        # 输入序列的形状：(sequence length, batch size, feature size)
        # import ipdb;ipdb.set_trace()
        # print(instances)
        L = sinput.size(2) * sinput.size(3) * sinput.size(4)
        instances=sinput.view(sinput.size(0),L,sinput.size(1))

        if encoder_hidden_state is not None:
            encoder_hidden_state=encoder_hidden_state.permute(1,0,2) 
        transformer_output = self.transformer_encoder(instances.permute(1,0,2),encoder_hidden_state=encoder_hidden_state)#mask取反，TransformerEncoder中mask为true才忽略

        out=transformer_output.permute(1,0,2).reshape(sinput.size())
        #把窗口结果去除padding并拼接回大batch
        # batch_output=transformer_output.permute(1,0,2)
        # batch_output=instances

        #方法1
        # final_outputs = []
        # for i in range(batch_output.shape[0]):
        #     output = batch_output[i]
        #     final_outputs.append(output[instance_masks[i]])

        # output=torch.cat(final_outputs, dim=0)

        #方法2
        # flat_output = batch_output.reshape(-1, batch_output.shape[-1])
        # flat_masks = instance_masks.reshape(-1)
        # # print(flat_masks.shape)
        # output = flat_output[flat_masks]
        # print(torch.all(output==sinput_F))
        # assert torch.all(output==sinput_F)

        # if self.res_attn:
        #     output=output+sinput_F

        # out = ME.SparseTensor(
        #     features=output,
        #     coordinate_map_key=sinput.coordinate_map_key,
        #     tensor_stride=sinput.tensor_stride,
        #     coordinate_manager=sinput.coordinate_manager,
        # )
        # out=self.output(out)
        return out