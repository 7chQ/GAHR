import torch
from torch import nn
from .base_transformer import Transformer, LayerNorm
from typing import Tuple, Union

from collections import OrderedDict
from torch.nn import functional as F

from model import lorentz as L

class D_Block(nn.Module):
    def __init__(self, fin, fout, k, s, p, res, CLIP_feat):
        super(D_Block, self).__init__()
        self.res, self.CLIP_feat = res, CLIP_feat
        self.learned_shortcut = (fin != fout)
        self.conv_r = nn.Sequential(
            nn.Conv2d(fin, fout, k, s, p, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(fin, fout, k, s, p, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            )
        self.conv_s = nn.Conv2d(fin, fout, 1, stride=1, padding=0)
        if self.res==True:
            self.gamma = nn.Parameter(torch.zeros(1))
        if self.CLIP_feat==True:
            self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x, CLIP_feat=None):
        res = self.conv_r(x)
        if self.learned_shortcut:
            x = self.conv_s(x)
        if (self.res==True)and(self.CLIP_feat==True):
            return (x + self.gamma*res + self.beta*CLIP_feat)/(1+self.beta+self.gamma)
        elif (self.res==True)and(self.CLIP_feat!=True):
            return x + self.gamma*res
        elif (self.res!=True)and(self.CLIP_feat==True):
            return x + self.beta*CLIP_feat
        else:
            return x
        
class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x 

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x
        
class VisualTransformer(nn.Module):
    def __init__(self, input_resolution: Union[int, Tuple[int, int]], patch_size: int, width: int, layers: int, heads: int, embed_dim: int,
                 checkpoint: bool, dropout: float = 0, emb_dropout: float = 0):
        super().__init__()
        if isinstance(input_resolution, int):
            input_resolution = (input_resolution, input_resolution)
        self.input_resolution = input_resolution
        self.num_x = (input_resolution[1] - patch_size) // patch_size + 1
        self.num_y = (input_resolution[0] - patch_size) // patch_size + 1
        num_patches = self.num_x * self.num_y

        output_dim = embed_dim
        self.output_dim = output_dim
        self.freeze_conv1 = True
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width,
                               kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn(num_patches + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, checkpoint=checkpoint, dropout=dropout,
                                       emb_dropout=emb_dropout)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))
        self.patch_dropout = PatchDropout(0.3) 

        self.initialize_parameters()

        self.visual_alpha = nn.Parameter(torch.tensor(embed_dim**-0.5).log())

    def initialize_parameters(self):
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)

        if self.freeze_conv1:
            for layer in [self.conv1]:
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = False
        return self
    

    def forward(self, x: torch.Tensor, curv, return_dense=False, return_feature=False):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        # shape = [*, width, grid ** 2]
        x = x.reshape(x.shape[0], x.shape[1], -1)
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x = self.ln_post(x[:, 0, :])
        x = self.ln_post(x)
        dense_feat = x
        
        if self.proj is not None:
            dense_feat = x @ self.proj
            x = dense_feat[:, 0, :]

        self.visual_alpha.data = torch.clamp(self.visual_alpha.data, max=0.0)
        x= x * self.visual_alpha.exp()
        with torch.autocast("cuda", dtype=torch.float32):
            x = L.exp_map0(x, curv.exp())
            
        if return_dense:
            return x, dense_feat
        if return_feature:
            return dense_feat
        return x 


def visual_transformer(config):
    vision_width = 768
    vision_layers = 12
    vision_heads = vision_width // 64

    kwargs = {
        'layers': vision_layers,
        'heads': vision_heads,
        'input_resolution': config.experiment.input_resolution,
        'patch_size': 16,
        'width': vision_width,
        'checkpoint': False,
        'embed_dim': config.model.embed_dim,
    }

    model = VisualTransformer(**kwargs)
    return model
