import torch
from torch import nn
from torch.nn import Module
import clip
import math
import sys
from models.encoders.helpers import _upsample_add
from models.encoders.psp_encoders_features2 import GradualStyleBlock
from models.stylegan2.model import EqualLinear, PixelNorm
from torch.nn import Linear, LayerNorm, LeakyReLU, Sequential, InstanceNorm2d, Conv2d
from PIL import Image
import torchvision.transforms as transforms

class TextModulationModule(Module):
    def __init__(self, in_channels):
        super(TextModulationModule, self).__init__()
        self.conv = Conv2d(in_channels, in_channels, 3, stride=1, padding=1, bias=False)
        self.norm = InstanceNorm2d(in_channels)
        self.gamma_function = Sequential(Linear(512, 512), LayerNorm([512]), LeakyReLU(), Linear(512, in_channels))
        self.beta_function = Sequential(Linear(512, 512), LayerNorm([512]), LeakyReLU(), Linear(512, in_channels))
        self.leakyrelu = LeakyReLU()
        
    def forward(self, x, embedding):
        x = self.conv(x)
        x = self.norm(x)
        log_gamma = self.gamma_function(embedding.float())
        gamma = log_gamma.exp().unsqueeze(2).unsqueeze(3)
        beta = self.beta_function(embedding.float()).unsqueeze(2).unsqueeze(3)
        out = x * (1 + gamma) + beta
        out = self.leakyrelu(out)
        return out
        
class SubTextMapper(Module):
    def __init__(self, opts, in_channels):
        super(SubTextMapper, self).__init__()
        self.opts = opts
        self.pixelnorm = PixelNorm()
        self.modulation_module_list = nn.ModuleList([TextModulationModule(in_channels) for i in range(3)])
        
    def forward(self, x, embedding):
        x = self.pixelnorm(x)
        for modulation_module in self.modulation_module_list:
            x = modulation_module(x, embedding)
        return x

class HairMapper(Module): 
    def __init__(self, opts):
        super(HairMapper, self).__init__()
        self.opts = opts
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device="cuda")
        self.transform = transforms.Compose([transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
        self.face_pool = torch.nn.AdaptiveAvgPool2d((224, 224))
        self.hairstyle_cut_flag = 0
        self.color_cut_flag = 0

        if not opts.no_coarse_mapper: 
            self.coarse_mapping = SubTextMapper(opts, 512)
        if not opts.no_medium_mapper:
            self.medium_mapping = SubTextMapper(opts, 256)
        if not opts.no_fine_mapper:
            self.fine_mapping = SubTextMapper(opts, 128)
            
        # self.styles = nn.ModuleList()
        # log_size = int(math.log(opts.stylegan_size, 2))
        # self.style_count = 2 * log_size - 2
        # self.coarse_ind = 3
        # self.middle_ind = 7
        # for i in range(self.style_count):
        #     if i < self.coarse_ind:
        #         style = GradualStyleBlock(512, 512, 16)
        #     elif i < self.middle_ind:
        #         style = GradualStyleBlock(512, 512, 32)
        #     else:
        #         style = GradualStyleBlock(512, 512, 64)
        #     self.styles.append(style)
        # self.latlayer1 = Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        # self.latlayer2 = Conv2d(128, 512, kernel_size=1, stride=1, padding=0)


    def forward(self, features, txt_embed):
        txt_embed = txt_embed.detach()
        c1, c2, c3 = features

        if not self.opts.no_coarse_mapper:
            c3 = self.coarse_mapping(c3, txt_embed)
        if not self.opts.no_medium_mapper:
            c2 = self.medium_mapping(c2, txt_embed)
        if not self.opts.no_fine_mapper:
            c1 = self.fine_mapping(c1, txt_embed)
        return (c1,c2,c3)
            
#         latents = []
#         for j in range(self.coarse_ind):
#             latents.append(self.styles[j](c3))

#         p2 = _upsample_add(c3, self.latlayer1(c2))
#         for j in range(self.coarse_ind, self.middle_ind):
#             latents.append(self.styles[j](p2))

#         p1 = _upsample_add(p2, self.latlayer2(c1))
#         for j in range(self.middle_ind, self.style_count):
#             latents.append(self.styles[j](p1))

#         out = torch.stack(latents, dim=1)
        # return out