from blocks import PCBActiv, ResnetBlock, UnetSkipConnectionDBlock
import torch.nn as nn
import torch

# Pretrained Model for PD-GAN Training
class PretrainedModel:
    def __init__(self, encoder_path, decoder_path):
        self.encoder = Encoder(input_nc=3, norm_layer="instance")
        self.decoder = Decoder(output_nc=3, norm_layer="instance")
        encoder_state = torch.load(encoder_path)
        decoder_state = torch.load(decoder_path)
        self.encoder.load_state_dict(encoder_state['net'])
        self.decoder.load_state_dict(decoder_state['net'])
        self.encoder = self.encoder.cuda()   
        self.decoder = self.decoder.cuda()
        
    def forward(self, broken_imgs):
        with torch.no_grad():
            out_7, out_5, out_4, out_3, out_2, out_1 = self.encoder.forward(broken_imgs)
            output = self.decoder.forward(out_1, out_2, out_3, out_4, out_5, out_7)
        return output

# Adapted from: https://github.com/KumapowerLIU/PD-GAN/blob/main/models/network/pconv.py
class Encoder(nn.Module):
    def __init__(self, input_nc, ngf=64, res_num=4, norm_layer="instance"):
        super(Encoder, self).__init__()

        # construct unet structure
        Encoder_1 = PCBActiv(
            input_nc, ngf, norm_layer=None, activ=None, outer=True
        )  # 128
        Encoder_2 = PCBActiv(ngf, ngf * 2, norm_layer=norm_layer)  # 64
        Encoder_3 = PCBActiv(ngf * 2, ngf * 4, norm_layer=norm_layer)  # 32
        Encoder_4 = PCBActiv(ngf * 4, ngf * 8, norm_layer=norm_layer)  # 16
        Encoder_5 = PCBActiv(ngf * 8, ngf * 8, norm_layer=norm_layer)  # 8
        Encoder_6 = PCBActiv(ngf * 8, ngf * 8, norm_layer=None, inner=True)  # 4

        blocks = []
        for _ in range(res_num):
            block = ResnetBlock(ngf * 8)
            blocks.append(block)

        self.middle = nn.Sequential(*blocks)

        self.Encoder_1 = Encoder_1
        self.Encoder_2 = Encoder_2
        self.Encoder_3 = Encoder_3
        self.Encoder_4 = Encoder_4
        self.Encoder_5 = Encoder_5
        self.Encoder_6 = Encoder_6

    def forward(self, x):
        out_1 = self.Encoder_1(x)
        out_2 = self.Encoder_2(out_1)
        out_3 = self.Encoder_3(out_2)
        out_4 = self.Encoder_4(out_3)
        out_5 = self.Encoder_5(out_4)
        out_6 = self.Encoder_6(out_5)
        out_7 = self.middle(out_6)
        return out_7, out_5, out_4, out_3, out_2, out_1

# Code is from https://github.com/KumapowerLIU/PD-GAN/blob/main/models/network/pconv.py
class Decoder(nn.Module):
    def __init__(self, output_nc, ngf=64, norm_layer=nn.BatchNorm2d):
        super(Decoder, self).__init__()

        # construct unet structure
        Decoder_1 = UnetSkipConnectionDBlock(
            ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True
        )
        Decoder_2 = UnetSkipConnectionDBlock(ngf * 16, ngf * 8, norm_layer=norm_layer)
        Decoder_3 = UnetSkipConnectionDBlock(ngf * 16, ngf * 4, norm_layer=norm_layer)
        Decoder_4 = UnetSkipConnectionDBlock(ngf * 8, ngf * 2, norm_layer=norm_layer)
        Decoder_5 = UnetSkipConnectionDBlock(ngf * 4, ngf, norm_layer=norm_layer)
        Decoder_6 = UnetSkipConnectionDBlock(
            ngf * 2, output_nc, norm_layer=norm_layer, outermost=True
        )

        self.Decoder_1 = Decoder_1
        self.Decoder_2 = Decoder_2
        self.Decoder_3 = Decoder_3
        self.Decoder_4 = Decoder_4
        self.Decoder_5 = Decoder_5
        self.Decoder_6 = Decoder_6

    def forward(self, input_1, input_2, input_3, input_4, input_5, input_6):
        y_1 = self.Decoder_1(input_6[0])
        y_2 = self.Decoder_2(torch.cat([y_1, input_5[0]], 1))
        y_3 = self.Decoder_3(torch.cat([y_2, input_4[0]], 1))
        y_4 = self.Decoder_4(torch.cat([y_3, input_3[0]], 1))
        y_5 = self.Decoder_5(torch.cat([y_4, input_2[0]], 1))
        y_6 = self.Decoder_6(torch.cat([y_5, input_1[0]], 1))
        out = y_6
        return out
    
if __name__ == "__main__":
    encoder_path = './Face/2_net_EN.pth'
    decoder_path = './Face/2_net_DE.pth'    
    pretrained_cnn = PretrainedModel('./Face/2_net_EN.pth', './Face/2_net_DE.pth')
    epoch = 20
    for i in range(0, 20):
        pass
    