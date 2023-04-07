import torch
import torch.nn as nn
import torch.nn.functional as F
from SPDNorm import SPDNormResnetBlock
import numpy as np
class Config:
    pass

class PDGANGenerator(nn.Module):
    
    def __init__(self):
        super().__init__() 
        self.config = Config()
        setattr(self.config, 'norm_G', 'spectral')
        self.layer0 = nn.Linear(256, 16 * 64 * 4 * 4)
        self.layer1 = SPDNormResnetBlock(16 * 64, 16 * 64, 2, 3, self.config)
        self.layer2 = nn.ConvTranspose2d(16 * 64, 16 * 64, kernel_size=4, stride=2, padding=1)
      
        self.layer3= SPDNormResnetBlock(16 * 64, 16 * 64, 3, 3, self.config)
        self.layer4 = nn.ConvTranspose2d(16 * 64, 16 * 64, kernel_size=4, stride=2, padding=1)
      
        self.layer5 = SPDNormResnetBlock(16 * 64, 8 * 64, 4, 3, self.config)
        self.layer6 = nn.ConvTranspose2d(8 * 64, 8 * 64, kernel_size=4, stride=2, padding=1)
      
        self.layer7 = SPDNormResnetBlock(8 * 64, 4 * 64, 5, 3, self.config)
        self.layer8 = nn.ConvTranspose2d(4 * 64, 4 * 64, kernel_size=4, stride=2, padding=1)
      
        self.layer9 = SPDNormResnetBlock(4 * 64, 2 * 64, 6, 5, self.config)
        self.layer10 = nn.ConvTranspose2d(2 * 64, 2 * 64, kernel_size=4, stride=2, padding=1)
      
        self.layer11 = SPDNormResnetBlock(2 * 64, 64, 7, 5, self.config)
        self.layer12 = nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1)
        self.layer13 = nn.Conv2d(64, 3, 3, padding=1)
        
    def forward(self, x, img, mask):
        x = self.layer0(x)
        out = x.view(-1, 16 * 64, 4, 4)
        in_mask = mask[:, 0, :, :].unsqueeze(1)
        out = self.layer1(out, img, in_mask)
        out = self.layer2(out)
        out = self.layer3(out, img, in_mask)
        out = self.layer4(out)
        out = self.layer5(out, img, in_mask)
        out = self.layer6(out)
        out = self.layer7(out, img, in_mask)
        out = self.layer8(out)
        out = self.layer9(out, img, in_mask)
        out = self.layer10(out)
        out = self.layer11(out, img, in_mask)
        out = self.layer12(out)
        out = F.leaky_relu(out, 2e-1)
        out = self.layer13(out)
        out = torch.tanh(out)
        return out

class PDGANDiscriminator(nn.Module):
    def __init__(self):
        super().__init__() 
        self.discriminator0 = NLayerDiscriminator()
        self.discriminator1 = NLayerDiscriminator()
    
    def forward(self, x):
        sampled_x = F.avg_pool2d(x, kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)
        return [self.discriminator0(x), self.discriminator1(sampled_x)]

    
# Adapted from: https://github.com/yuan-yin/UNISST and
# https://github.com/KumapowerLIU/PD-GAN/blob/main/models/network/Discriminator.py
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))
        nf = 64
        input_nc = 6

        norm_layer = get_nonspade_norm_layer()
        sequence = [
            [
                nn.Conv2d(input_nc, nf, kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, False),
            ]
        ]

        for n in range(1, 4):
            nf_prev = nf
            nf = min(nf * 2, 512)
            stride = 1 if n == 3 else 2
            sequence += [
                [
                    norm_layer(
                        nn.Conv2d(
                            nf_prev, nf, kernel_size=kw, stride=stride, padding=padw
                        )
                    ),
                    nn.LeakyReLU(0.2, False),
                ]
            ]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        # We divide the layers into groups to extract intermediate layer outputs
        for n in range(len(sequence)):
            self.add_module("model" + str(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        return results[1:]

# Adapted from: https://github.com/yuan-yin/UNISST and 
# https://github.com/KumapowerLIU/PD-GAN/blob/main/models/network/Discriminator.py
def get_nonspade_norm_layer(norm_type="spectralinstance"):
    # helper function to get # output channels of the previous layer
    def get_out_channel(layer):
        if hasattr(layer, "out_channels"):
            return getattr(layer, "out_channels")
        return layer.weight.size(0)

    # this function will be returned
    def add_norm_layer(layer):
        nonlocal norm_type
        if norm_type.startswith("spectral"):
            layer = nn.utils.spectral_norm(layer)
            subnorm_type = norm_type[len("spectral") :]

        if subnorm_type == "none" or len(subnorm_type) == 0:
            return layer

        # remove bias in the previous layer, which is meaningless
        # since it has no effect after normalization
        if getattr(layer, "bias", None) is not None:
            delattr(layer, "bias")
            layer.register_parameter("bias", None)

        if subnorm_type == "batch":
            norm_layer = nn.BatchNorm2d(get_out_channel(layer), affine=True)
        elif subnorm_type == "instance":
            norm_layer = nn.InstanceNorm2d(get_out_channel(layer), affine=False)
        else:
            raise ValueError("normalization layer %s is not recognized" % subnorm_type)

        return nn.Sequential(layer, norm_layer)

    return add_norm_layer