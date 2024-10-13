# Copyright (c) 2020 NVIDIA CORPORATION.
# Copyright (c) 2018-2020 Chris Choy (chrischoy@ai.stanford.edu).
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
from typing import Optional
import numpy as np
import torch
import torch.nn as nn
from torch.nn import Module
from torch.autograd import Function

from MinkowskiSparseTensor import SparseTensor
from MinkowskiTensorField import TensorField

from MinkowskiPooling import MinkowskiGlobalAvgPooling
from MinkowskiBroadcast import (
    MinkowskiBroadcastAddition,
    MinkowskiBroadcastMultiplication,
)
from MinkowskiEngineBackend._C import (
    CoordinateMapKey,
    BroadcastMode,
    PoolingMode,
)
from MinkowskiCoordinateManager import CoordinateManager

from MinkowskiCommon import (
    MinkowskiModuleBase,
    get_minkowski_function,
)
import torch.nn.init as init
import torch.nn.functional as F


class MinkowskiBatchNorm(Module):
    r"""A batch normalization layer for a sparse tensor.

    See the pytorch :attr:`torch.nn.BatchNorm1d` for more details.
    """

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(MinkowskiBatchNorm, self).__init__()
        self.bn = torch.nn.BatchNorm1d(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
        )

    def forward(self, input):
        output = self.bn(input.F)
        if isinstance(input, TensorField):
            return TensorField(
                output,
                coordinate_field_map_key=input.coordinate_field_map_key,
                coordinate_manager=input.coordinate_manager,
                quantization_mode=input.quantization_mode,
            )
        else:
            return SparseTensor(
                output,
                coordinate_map_key=input.coordinate_map_key,
                coordinate_manager=input.coordinate_manager,
            )

    def __repr__(self):
        s = "({}, eps={}, momentum={}, affine={}, track_running_stats={})".format(
            self.bn.num_features,
            self.bn.eps,
            self.bn.momentum,
            self.bn.affine,
            self.bn.track_running_stats,
        )
        return self.__class__.__name__ + s


class MinkowskiSyncBatchNorm(MinkowskiBatchNorm):
    r"""A batch normalization layer with multi GPU synchronization."""

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        process_group=None,
    ):
        Module.__init__(self)
        self.bn = torch.nn.SyncBatchNorm(
            num_features,
            eps=eps,
            momentum=momentum,
            affine=affine,
            track_running_stats=track_running_stats,
            process_group=process_group,
        )

    def forward(self, input):
        output = self.bn(input.F)
        if isinstance(input, TensorField):
            return TensorField(
                output,
                coordinate_field_map_key=input.coordinate_field_map_key,
                coordinate_manager=input.coordinate_manager,
                quantization_mode=input.quantization_mode,
            )
        else:
            return SparseTensor(
                output,
                coordinate_map_key=input.coordinate_map_key,
                coordinate_manager=input.coordinate_manager,
            )

    @classmethod
    def convert_sync_batchnorm(cls, module, process_group=None):
        r"""Helper function to convert
        :attr:`MinkowskiEngine.MinkowskiBatchNorm` layer in the model to
        :attr:`MinkowskiEngine.MinkowskiSyncBatchNorm` layer.

        Args:
            module (nn.Module): containing module
            process_group (optional): process group to scope synchronization,
            default is the whole world

        Returns:
            The original module with the converted
            :attr:`MinkowskiEngine.MinkowskiSyncBatchNorm` layer

        Example::

            >>> # Network with MinkowskiBatchNorm layer
            >>> module = torch.nn.Sequential(
            >>>            MinkowskiLinear(20, 100),
            >>>            MinkowskiBatchNorm1d(100)
            >>>          ).cuda()
            >>> # creating process group (optional)
            >>> # process_ids is a list of int identifying rank ids.
            >>> process_group = torch.distributed.new_group(process_ids)
            >>> sync_bn_module = convert_sync_batchnorm(module, process_group)

        """
        module_output = module
        if isinstance(module, MinkowskiBatchNorm):
            module_output = MinkowskiSyncBatchNorm(
                module.bn.num_features,
                module.bn.eps,
                module.bn.momentum,
                module.bn.affine,
                module.bn.track_running_stats,
                process_group,
            )
            if module.bn.affine:
                with torch.no_grad():
                    module_output.bn.weight = module.bn.weight
                    module_output.bn.bias = module.bn.bias
            module_output.bn.running_mean = module.bn.running_mean
            module_output.bn.running_var = module.bn.running_var
            module_output.bn.num_batches_tracked = module.bn.num_batches_tracked
            if hasattr(module, "qconfig"):
                module_output.bn.qconfig = module.bn.qconfig
        for name, child in module.named_children():
            module_output.add_module(
                name, cls.convert_sync_batchnorm(child, process_group)
            )
        del module
        return module_output


class MinkowskiInstanceNormFunction(Function):
    @staticmethod
    def forward(
        ctx,
        in_feat: torch.Tensor,
        in_coords_key: CoordinateMapKey,
        glob_coords_key: CoordinateMapKey = None,
        coords_manager: CoordinateManager = None,
        gpooling_mode=PoolingMode.GLOBAL_AVG_POOLING_KERNEL,
    ):
        if glob_coords_key is None:
            glob_coords_key = CoordinateMapKey(in_coords_key.get_coordinate_size())

        gpool_avg_forward = get_minkowski_function("GlobalPoolingForward", in_feat)
        broadcast_forward = get_minkowski_function("BroadcastForward", in_feat)

        mean, num_nonzero = gpool_avg_forward(
            in_feat,
            gpooling_mode,
            in_coords_key,
            glob_coords_key,
            coords_manager._manager,
        )

        # X - \mu
        centered_feat = broadcast_forward(
            in_feat,
            -mean,
            BroadcastMode.ELEMENTWISE_ADDITON,
            in_coords_key,
            glob_coords_key,
            coords_manager._manager,
        )

        # Variance = 1/N \sum (X - \mu) ** 2
        variance, num_nonzero = gpool_avg_forward(
            centered_feat ** 2,
            gpooling_mode,
            in_coords_key,
            glob_coords_key,
            coords_manager._manager,
        )

        # norm_feat = (X - \mu) / \sigma
        inv_std = 1 / (variance + 1e-8).sqrt()
        norm_feat = broadcast_forward(
            centered_feat,
            inv_std,
            BroadcastMode.ELEMENTWISE_MULTIPLICATION,
            in_coords_key,
            glob_coords_key,
            coords_manager._manager,
        )

        ctx.saved_vars = (in_coords_key, glob_coords_key, coords_manager, gpooling_mode)
        # For GPU tensors, must use save_for_backward.
        ctx.save_for_backward(inv_std, norm_feat)
        return norm_feat

    @staticmethod
    def backward(ctx, out_grad):
        # https://kevinzakka.github.io/2016/09/14/batch_normalization/
        in_coords_key, glob_coords_key, coords_manager, gpooling_mode = ctx.saved_vars

        # To prevent the memory leakage, compute the norm again
        inv_std, norm_feat = ctx.saved_tensors

        gpool_avg_forward = get_minkowski_function("GlobalPoolingForward", out_grad)
        broadcast_forward = get_minkowski_function("BroadcastForward", out_grad)

        # 1/N \sum dout
        mean_dout, num_nonzero = gpool_avg_forward(
            out_grad,
            gpooling_mode,
            in_coords_key,
            glob_coords_key,
            coords_manager._manager,
        )

        # 1/N \sum (dout * out)
        mean_dout_feat, num_nonzero = gpool_avg_forward(
            out_grad * norm_feat,
            gpooling_mode,
            in_coords_key,
            glob_coords_key,
            coords_manager._manager,
        )

        # out * 1/N \sum (dout * out)
        feat_mean_dout_feat = broadcast_forward(
            norm_feat,
            mean_dout_feat,
            BroadcastMode.ELEMENTWISE_MULTIPLICATION,
            in_coords_key,
            glob_coords_key,
            coords_manager._manager,
        )

        unnorm_din = broadcast_forward(
            out_grad - feat_mean_dout_feat,
            -mean_dout,
            BroadcastMode.ELEMENTWISE_ADDITON,
            in_coords_key,
            glob_coords_key,
            coords_manager._manager,
        )

        norm_din = broadcast_forward(
            unnorm_din,
            inv_std,
            BroadcastMode.ELEMENTWISE_MULTIPLICATION,
            in_coords_key,
            glob_coords_key,
            coords_manager._manager,
        )

        return norm_din, None, None, None, None


class MinkowskiStableInstanceNorm(MinkowskiModuleBase):
    def __init__(self, num_features,group=1):
        Module.__init__(self)
        self.num_features = num_features
        if group > num_features:
            print("group should be less than num_features, set group=num_features")
            group = num_features
        self.group= group
        assert self.num_features % self.group == 0
        # print(group)
        self.eps = 1e-6

        self.weight = nn.Parameter(torch.ones(1, num_features//self.group))
        self.bias = nn.Parameter(torch.zeros(1, num_features//self.group))

        self.mean_in = MinkowskiGlobalAvgPooling()
        self.glob_sum = MinkowskiBroadcastAddition()
        self.glob_sum2 = MinkowskiBroadcastAddition()
        self.glob_mean = MinkowskiGlobalAvgPooling()
        self.glob_times = MinkowskiBroadcastMultiplication()
        self.reset_parameters()

    def __repr__(self):
        s = f"(nchannels={self.num_features})"
        return self.__class__.__name__ + s

    def reset_parameters(self):
        self.weight.data.fill_(1)
        self.bias.data.zero_()

    def forward(self, x):
        # neg_mean_in = self.mean_in(
        #     SparseTensor(-x.F, coords_key=x.coords_key, coords_manager=x.coords_man)
        # )
        neg_mean_in = self.mean_in(
            SparseTensor(-x.F, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)
        )
        # print(neg_mean_in.F)

        # print(neg_mean_in/)
        neg_mean_in_F=neg_mean_in.F#.mean(dim=1,keepdim=True).repeat(1,self.num_features)
        neg_mean_in_F=neg_mean_in_F.view(-1,self.num_features//self.group,self.group).mean(dim=-1)
        neg_mean_in_F=torch.repeat_interleave(neg_mean_in_F,self.group,dim=-1)
        # import ipdb;ipdb.set_trace()
        neg_mean_in=SparseTensor(
            neg_mean_in_F,
            coordinate_map_key=neg_mean_in.coordinate_map_key,
            coordinate_manager=neg_mean_in.coordinate_manager,
        )
        


        # print(neg_mean_in.F.shape)#b,c
        centered_in = self.glob_sum(x, neg_mean_in)
        temp = SparseTensor(
            centered_in.F ** 2,
            coordinate_map_key=centered_in.coordinate_map_key,
            coordinate_manager=centered_in.coordinate_manager,
        )
        var_in = self.glob_mean(temp)

        var_in_F=var_in.F#.mean(dim=1,keepdim=True).repeat(1,self.num_features)
        var_in_F=var_in_F.view(-1,self.num_features//self.group,self.group).mean(dim=-1)
        var_in_F=torch.repeat_interleave(var_in_F,self.group,dim=-1)
        # print(var_in_F==var_in.F)
        var_in = SparseTensor(
            var_in_F,
            coordinate_manager=var_in.coordinate_manager,
            coordinate_map_key=var_in.coordinate_map_key,
        )
        instd_in = SparseTensor(
            1 / (var_in.F + self.eps).sqrt(),
            coordinate_map_key=var_in.coordinate_map_key,
            coordinate_manager=var_in.coordinate_manager,
        )

        x = self.glob_times(self.glob_sum2(x, neg_mean_in), instd_in)

        scale=torch.repeat_interleave(self.weight,self.group,dim=-1)
        shift=torch.repeat_interleave(self.bias,self.group,dim=-1)

        return SparseTensor(
            x.F * scale + shift,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )


class MinkowskiStableGroupNorm(MinkowskiModuleBase):
    def __init__(self, num_features,group=None):
        Module.__init__(self)
        self.num_features = num_features
        group= group if group is not None else num_features
        self.eps = 1e-6
        self.weight = nn.Parameter(torch.ones(1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, num_features))


        self.mean_in = MinkowskiGlobalAvgPooling()
        self.glob_sum = MinkowskiBroadcastAddition()
        self.glob_sum2 = MinkowskiBroadcastAddition()
        self.glob_mean = MinkowskiGlobalAvgPooling()
        self.glob_times = MinkowskiBroadcastMultiplication()
        self.reset_parameters()

    def __repr__(self):
        s = f"(nchannels={self.num_features})"
        return self.__class__.__name__ + s

    def reset_parameters(self):
        self.weight.data.fill_(1)
        self.bias.data.zero_()

    def forward(self, x):
        # neg_mean_in = self.mean_in(
        #     SparseTensor(-x.F, coords_key=x.coords_key, coords_manager=x.coords_man)
        # )

        #先平均池化,然后对通道求平均
        neg_mean_in = self.mean_in(
            SparseTensor(-x.F, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)
        )
        #然后对通道求平均
        # print(neg_mean_in.F.shape)  #b,c
        # print(neg_mean_in.F.mean(dim=1,keepdim=True).repeat(1,self.num_features).shape) #b,c
        neg_mean_in_F=neg_mean_in.F.mean(dim=1,keepdim=True).repeat(1,self.num_features)

        neg_mean_in=SparseTensor(
            neg_mean_in_F,
            coordinate_map_key=neg_mean_in.coordinate_map_key,
            coordinate_manager=neg_mean_in.coordinate_manager,
        )

        centered_in = self.glob_sum(x, neg_mean_in)

        temp = SparseTensor(
            centered_in.F ** 2,
            coordinate_map_key=centered_in.coordinate_map_key,
            coordinate_manager=centered_in.coordinate_manager,
        )
        var_in = self.glob_mean(temp)


        var_in_F=var_in.F.mean(dim=1,keepdim=True).repeat(1,self.num_features)
        var_in = SparseTensor(
            var_in_F,
            coordinate_map_key=var_in.coordinate_map_key,
            coordinate_manager=var_in.coordinate_manager,
        )

        instd_in = SparseTensor(
            1 / (var_in.F + self.eps).sqrt(),
            coordinate_map_key=var_in.coordinate_map_key,
            coordinate_manager=var_in.coordinate_manager,
        )

        x = self.glob_times(self.glob_sum2(x, neg_mean_in), instd_in)
        return SparseTensor(
            x.F * self.weight + self.bias,
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )

class MinkowskiInstanceNorm(MinkowskiModuleBase):
    r"""A instance normalization layer for a sparse tensor."""

    def __init__(self, num_features):
        r"""
        Args:

            num_features (int): the dimension of the input feautres.

            mode (GlobalPoolingModel, optional): The internal global pooling computation mode.
        """
        Module.__init__(self)
        self.num_features = num_features
        self.weight = nn.Parameter(torch.ones(1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, num_features))
        self.reset_parameters()
        self.inst_norm = MinkowskiInstanceNormFunction()

    def __repr__(self):
        s = f"(nchannels={self.num_features})"
        return self.__class__.__name__ + s

    def reset_parameters(self):
        self.weight.data.fill_(1)
        self.bias.data.zero_()

    def forward(self, input: SparseTensor):
        assert isinstance(input, SparseTensor)

        output = self.inst_norm.apply(
            input.F, input.coordinate_map_key, None, input.coordinate_manager
        )
        output = output * self.weight + self.bias

        return SparseTensor(
            output,
            coordinate_map_key=input.coordinate_map_key,
            coordinate_manager=input.coordinate_manager,
        )

class HjmInstanceNorm(MinkowskiBatchNorm):
    r"""把batch分开，对每个实例做MinkowskiBatchNorm，相当于实际上的instanceNorm"""
    def forward(self, input):
        # return super().forward(input)
        coords, features = input.C,input.F
        batch_indices = coords[:, 0].cpu().numpy()
        unique_batch_indices = np.unique(batch_indices)
        output=torch.zeros_like(features)
        for batch_idx in unique_batch_indices:
            batch_mask = (batch_indices == batch_idx)
            output[batch_mask]+=self.bn(features[batch_mask])
        
        if isinstance(input, TensorField):
            return TensorField(
                output,
                coordinate_field_map_key=input.coordinate_field_map_key,
                coordinate_manager=input.coordinate_manager,
                quantization_mode=input.quantization_mode,
            )
        else:
            return SparseTensor(
                output,
                coordinate_map_key=input.coordinate_map_key,
                coordinate_manager=input.coordinate_manager,
            )

class AdaStableInstanceNorm(MinkowskiModuleBase):
    r"""AdaIN层，对每个实例做instanceNorm"""
    def __init__(self, num_features,embedding_dim):
        Module.__init__(self)
        self.num_features = num_features
        self.eps = 1e-6
        self.weight = nn.Parameter(torch.ones(1, num_features))
        self.bias = nn.Parameter(torch.zeros(1, num_features))

        self.mean_in = MinkowskiGlobalAvgPooling()
        self.glob_sum = MinkowskiBroadcastAddition()
        self.glob_sum2 = MinkowskiBroadcastAddition()
        self.glob_mean = MinkowskiGlobalAvgPooling()
        self.glob_times = MinkowskiBroadcastMultiplication()

        self.linear=nn.Linear(embedding_dim,num_features*2)
        # self.linear_shift=nn.Linear(embedding_dim,num_features)

        self.reset_parameters()

    def __repr__(self):
        s = f"(nchannels={self.num_features})"
        return self.__class__.__name__ + s

    def reset_parameters(self):
        self.weight.data.fill_(1)
        self.bias.data.zero_()
        nn.init.normal_(self.linear.weight, 0, 0.01)
        nn.init.constant_(self.linear.bias, 0)                      

    def forward(self, x,emb):
        # scale=self.linear_scale(emb)
        # shift=self.linear_shift(emb)
        scale,shift=self.linear(emb).chunk(2,dim=1)

        # coords, features = x.C,x.F
        # batch_indices = coords[:, 0]
        # # unique_batch_indices = torch.unique(batch_indices)
        # scale_all=torch.zeros_like(features)
        # shift_all=torch.zeros_like(features)
        # scale_all = scale[x.C[:,0]].clone()
        # shift_all = shift[x.C[:,0]].clone()
        # for batch_idx in unique_batch_indices:
        #     batch_mask = (batch_indices == batch_idx)
        #     scale_all[batch_mask]=scale[batch_idx]
        #     shift_all[batch_mask]=shift[batch_idx]

        # neg_mean_in = self.mean_in(
        #     SparseTensor(-x.F, coords_key=x.coords_key, coords_manager=x.coords_man)
        # )
        neg_mean_in = self.mean_in(
            SparseTensor(-x.F, coordinate_map_key=x.coordinate_map_key, coordinate_manager=x.coordinate_manager)
        )
        # print(neg_mean_in.C.shape)
        centered_in = self.glob_sum(x, neg_mean_in) #已经中心化，均值为0
        temp = SparseTensor(
            centered_in.F ** 2,
            coordinate_map_key=centered_in.coordinate_map_key,
            coordinate_manager=centered_in.coordinate_manager,
        )
        var_in = self.glob_mean(temp) #方差
        instd_in = SparseTensor(
            1 / (var_in.F + self.eps).sqrt(),
            coordinate_map_key=var_in.coordinate_map_key,
            coordinate_manager=var_in.coordinate_manager,
        )

        x = self.glob_times(self.glob_sum2(x, neg_mean_in), instd_in) #标准化，每个instance的feature都减去均值除以标准差
        # import ipdb;ipdb.set_trace()
        return SparseTensor(
            (x.F * self.weight + self.bias)*(1+scale[x.C[:,0]])+shift[x.C[:,0]], #ada
            # x.F*(1+scale[x.C[:,0]])+shift[x.C[:,0]],
            coordinate_map_key=x.coordinate_map_key,
            coordinate_manager=x.coordinate_manager,
        )
        


class HjmGroupNorm(Module):
    r"""Applies Group Normalization over a mini-batch of inputs.

    This layer implements the operation as described in
    the paper `Group Normalization <https://arxiv.org/abs/1803.08494>`__

    .. math::
        y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta

    The input channels are separated into :attr:`num_groups` groups, each containing
    ``num_channels / num_groups`` channels. :attr:`num_channels` must be divisible by
    :attr:`num_groups`. The mean and standard-deviation are calculated
    separately over the each group. :math:`\gamma` and :math:`\beta` are learnable
    per-channel affine transform parameter vectors of size :attr:`num_channels` if
    :attr:`affine` is ``True``.
    The standard-deviation is calculated via the biased estimator, equivalent to
    `torch.var(input, unbiased=False)`.

    This layer uses statistics computed from input data in both training and
    evaluation modes.

    Args:
        num_groups (int): number of groups to separate the channels into
        num_channels (int): number of channels expected in input
        eps: a value added to the denominator for numerical stability. Default: 1e-5
        affine: a boolean value that when set to ``True``, this module
            has learnable per-channel affine parameters initialized to ones (for weights)
            and zeros (for biases). Default: ``True``.

    Shape:
        - Input: :math:`(N, C, *)` where :math:`C=\text{num\_channels}`
        - Output: :math:`(N, C, *)` (same shape as input)

    Examples::

        >>> input = torch.randn(20, 6, 10, 10)
        >>> # Separate 6 channels into 3 groups
        >>> m = nn.GroupNorm(3, 6)
        >>> # Separate 6 channels into 6 groups (equivalent with InstanceNorm)
        >>> m = nn.GroupNorm(6, 6)
        >>> # Put all 6 channels into a single group (equivalent with LayerNorm)
        >>> m = nn.GroupNorm(1, 6)
        >>> # Activating the module
        >>> output = m(input)
    """

    __constants__ = ['num_groups', 'num_channels', 'eps', 'affine']
    num_groups: int
    num_channels: int
    eps: float
    affine: bool

    def __init__(self, num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        if num_channels % num_groups != 0:
            raise ValueError('num_channels must be divisible by num_groups')

        self.num_groups = num_groups
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine
        if self.affine:
            self.weight = nn.Parameter(torch.empty(num_groups, **factory_kwargs))
            self.bias = nn.Parameter(torch.empty(num_groups, **factory_kwargs))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        weight=torch.repeat_interleave(self.weight,self.num_channels//self.num_groups,dim=-1)
        bias=torch.repeat_interleave(self.bias,self.num_channels//self.num_groups,dim=-1)

        return F.group_norm(
            input, self.num_groups, weight, bias, self.eps)

    def extra_repr(self) -> str:
        return '{num_groups}, {num_channels}, eps={eps}, ' \
            'affine={affine}'.format(**self.__dict__)

