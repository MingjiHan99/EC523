import torch
import torch.nn as nn
import torch.functional as F
from spmblock import SPDNormResnetBlock
class PDGAN:
    pass

class PDGANGenerator(nn.Module):
    
    def __init__(self):
        super().__init__() 
        self.layer0 = nn.Linear(256, 16 * 64 * 4 * 4)
        self.layer1 = SPDNormResnetBlock()
        self.layer2 = nn.ConvTranspose2d(16 * 64, 16 * 64, kernel_size=4, stride=2, padding=1)
        self.layer3= SPDNormResnetBlock()
        self.layer4 = nn.ConvTranspose2d(16 * 64, 16 * 64, kernel_size=4, stride=2, padding=1)
        self.layer5 = SPDNormResnetBlock()
        self.layer6 = nn.ConvTranspose2d(8 * 64, 8 * 64, kernel_size=4, stride=2, padding=1)
        self.layer7 = SPDNormResnetBlock()
        self.layer8 = nn.ConvTranspose2d(4 * 64, 4 * 64, kernel_size=4, stride=2, padding=1)
        self.layer9 = SPDNormResnetBlock()
        self.layer10 = nn.ConvTranspose2d(4 * 64, 2 * 64, kernel_size=4, stride=2, padding=1)
        self.layer11 = SPDNormResnetBlock()
        self.layer12 = nn.ConvTranspose2d(2 * 64, 64, kernel_size=4, stride=2, padding=1)
        self.layer13 = nn.Conv2d(64, 4, 4, padding=1)
        
    def forward(self, vec, img, mask):
        out = self.layer0(vec)
        out = out.view(-1, 16 * 64, 4, 4)
        mask = mask[:, 0, :, :].unsqueeze(1)
        out = self.layer1(out, img, mask)
        out = self.layer2(out)
        out = self.layer3(out, img, mask)
        out = self.layer4(out)
        out = self.layer5(out, img, mask)
        out = self.layer6(out)
        out = self.layer7(out, img, mask)
        out = self.layer8(out)
        out = self.layer9(out, img, mask)
        out = self.layer10(out)
        out = self.layer11(out, img, mask)
        out = self.layer12(out)
        out = F.leak_relu(out, 2e-1)
        out = self.layer13(out)
        out = F.tanh(out)
        return out

class PDGANDiscriminator:
    def __init__(self):
        pass