
# of the code.
from typing import Optional
import torch

import MinkowskiEngine as ME
import MinkowskiEngine.MinkowskiFunctional as MF

from tests.python.common import data_loader

from MinkowskiEngine.modules.vae_block import ResNetBlock,ResNet2,ResNet3
import torch.nn as nn
from diffusers.models.embeddings import TimestepEmbedding, Timesteps
from torch.nn import TransformerEncoder, TransformerEncoderLayer




class UNet(nn.Module):

    # CHANNELS = [4,320, 640, 1280]

    def __init__(self,channels,config,time_embedding_type="positional",flip_sin_to_cos=True,freq_shift: int = 0,num_train_timesteps: Optional[int] = None,embedding_dim: Optional[int] = None):
        super(UNet, self).__init__()
        ch=channels
        self.config=config
        
        self.block1 =ResNet2(ch[0], ch[1],after='downsample',embedding_dim=embedding_dim)
        self.block2 =ResNet2(ch[1], ch[2],after='downsample',embedding_dim=embedding_dim)
        self.block3 =ResNet2(ch[2], ch[3],after='downsample',embedding_dim=embedding_dim)

        self.block3_tr = ResNet2(ch[3], ch[2],after='upsample_determine',embedding_dim=embedding_dim)
        self.block2_tr = ResNet2(ch[2]*2, ch[1],after='upsample_determine',embedding_dim=embedding_dim)
        # self.block1_tr = ResNet2(ch[1]*2, ch[0],after='upsample_determine',embedding_dim=embedding_dim)
        self.block1_tr = ResNet2(ch[1]*2, ch[0],after='upsample_determine')


        # if time_embedding_type == "positional":
        #     self.time_proj = Timesteps(ch[1], flip_sin_to_cos, freq_shift)
        # elif time_embedding_type == "learned":
        #     self.time_proj = nn.Embedding(num_train_timesteps, ch[1])
        #     timestep_input_dim = ch[1]


    def forward(self, x,temb=None):
        out_s1 = self.block1(x,temb)
        out = MF.silu(out_s1)

        out_s2 = self.block2(out,temb)
        out = MF.silu(out_s2)

        out_s4 = self.block3(out,temb)
        out = MF.silu(out_s4)

        out = MF.silu(self.block3_tr(out,temb))
        out = ME.cat(out, out_s2)

        out = MF.silu(self.block2_tr(out,temb))
        out = ME.cat(out, out_s1)

        # return self.block1_tr(out,temb)
        return self.block1_tr(out)



if __name__ == '__main__':
    # loss and network
    net = UNet(3, 5, D=2)
    print(net)

    # a data loader must return a tuple of coords, features, and labels.
    coords, feat, label = data_loader()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    net = net.to(device)
    input = ME.SparseTensor(feat, coords, device=device)

    # Forward
    output = net(input)
