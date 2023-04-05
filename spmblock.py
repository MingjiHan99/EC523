import torch.nn as nn
import torch.nn.functional as F
import torch
class SPDNormSoftBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SPDNormSoftBlock, self).__init__()
        self.layer0 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.layer1 = nn.InstanceNorm2d(128)
        self.layer2 = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.layer3 = nn.Conv2d(2 * in_channel, in_channel, kernel_size=3, padding=1)
        self.layer4 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.layer5 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1)

    def forward(self, x, img, mask):
        img = F.interpolate(img, size=x.size()[2:])
        mask = F.interpolate(mask, size=mask.size()[2:])
        out = self.layer0(img)
        out = self.layer1(out)
        out = F.relu(out)
        features = self.layer2(out)
        out = self.layer3(torch.cat([features, img], 1))
        pd_map = out * (1 - mask) + mask
        
        gamma = self.layer4(img)
        beta = self.layer5(img)
        # Residual Structure
        return x + pd_map * (gamma * x +  beta)
    

class SPDNormHardBlock(nn.Module):
    def __init__(self, in_channel, out_channel, k, mask_num):
        super(SPDNormHardBlock, self).__init__()
        self.conv0 = nn.Conv2d(in_channel, in_channel, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.mask_num = mask_num
        
    def forward(self):
        img = F.interpolate(img, size=x.size()[2:])
        mask = F.interpolate(mask, size=mask.size()[2:])
        '''
        pd_map = 0.0
        for i in range(self.mask_number):
            pd_map = self.mask_conv(mask_pre)
            mask_generate = (mask_out - mask_pre) * (
                1 / (torch.exp(torch.tensor(i + 1).cuda()))
            )
            mask_pre = mask_out
            pd_map = pd_map + mask_generate
        hard_map_inner = (1 - mask_out) * (1 / (torch.exp(torch.tensor(i + 1).cuda())))
        hard_map = hard_map + mask_resize + hard_map_inner
        hard_out = self.conv_0(self.actvn(self.norm_0(x, prior_image_resize, hard_map)))
        hard_out = self.conv_1(
            self.actvn(self.norm_1(hard_out, prior_image_resize, hard_map))
        )
        '''
        pass

    
class SPDNormResnetBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SPDNormResnetBlock, self).__init__()
        self.soft = SPDNormSoftBlock()
        self.hard0 = SPDNormHardBlock()
        self.hard1 = SPDNormHardBlock()
        
        
    def forward(self, x, img, mask):
        pass
    

# Adapted from: https://github.com/Boyiliee/Positional-Normalization
def PositionalNorm2d(x, epsilon=1e-5):
    # x: B*C*W*H normalize in C dim
    mean = x.mean(dim=1, keepdim=True)
    std = x.var(dim=1, keepdim=True).add(epsilon).sqrt()
    output = (x - mean) / std
    return output
