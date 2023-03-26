import torch.nn as nn
import torch

# Adapted from: https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/blob/master/net.py
class PartialConv(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
    ):
        super().__init__()

        self.input_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )
        self.mask_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
        )

        torch.nn.init.constant_(self.mask_conv.weight, 1.0)

        # mask is not updated
        for param in self.mask_conv.parameters():
            param.requires_grad = False

    def forward(self, x):
        # http://masc.cs.gmu.edu/wiki/partialconv
        # C(X) = W^T * X + b, C(0) = b, D(M) = 1 * M + 0 = sum(M)
        # W^T* (M .* X) / sum(M) + b = [C(M .* X) – C(0)] / D(M) + C(0)
        input = x[0]
        mask = x[1].float().cuda()

        output = self.input_conv(input * mask)
        if self.input_conv.bias is not None:
            output_bias = self.input_conv.bias.view(1, -1, 1, 1).expand_as(output)
        else:
            output_bias = torch.zeros_like(output)

        with torch.no_grad():
            output_mask = self.mask_conv(mask)

        no_update_holes = output_mask == 0
        mask_sum = output_mask.masked_fill_(no_update_holes.bool(), 1.0)
        output_pre = (output - output_bias) / mask_sum + output_bias
        output = output_pre.masked_fill_(no_update_holes.bool(), 0.0)
        new_mask = torch.ones_like(output)
        new_mask = new_mask.masked_fill_(no_update_holes.bool(), 0.0)
        out = [output, new_mask]
        return out

# Adapted from: https://github.com/naoto0804/pytorch-inpainting-with-partial-conv/blob/master/net.py
class PCBActiv(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        norm_layer="instance",
        sample="down-4",
        activ="leaky",
        conv_bias=False,
        inner=False,
        outer=False,
    ):
        super().__init__()
        if sample == "same-5":
            self.conv = PartialConv(in_ch, out_ch, 5, 1, 2, bias=conv_bias)
        elif sample == "same-7":
            self.conv = PartialConv(in_ch, out_ch, 7, 1, 3, bias=conv_bias)
        elif sample == "down-4":
            self.conv = PartialConv(in_ch, out_ch, 4, 2, 1, bias=conv_bias)
        else:
            self.conv = PartialConv(in_ch, out_ch, 3, 1, 1, bias=conv_bias)

        if norm_layer == "instance":
            self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        elif norm_layer == "batch":
            self.norm = nn.BatchNorm2d(out_ch, affine=True)
        else:
            pass

        if activ == "relu":
            self.activation = nn.ReLU()
        elif activ == "leaky":
            self.activation = nn.LeakyReLU(negative_slope=0.2)
        else:
            pass
        self.inner = inner
        self.outer = outer

    def forward(self, input):
        out = input
        if self.inner:
            out[0] = self.activation(out[0])
            out = self.conv(out)
        elif self.outer:
            out = self.conv(out)
        else:
            out[0] = self.activation(out[0])
            out = self.conv(out)
            out[0] = self.norm(out[0])
        return out

# Adapted from https://github.com/KumapowerLIU/PD-GAN/blob/main/models/blocks/pconvblocks.py
class UnetSkipConnectionDBlock(nn.Module):
    def __init__(
        self,
        inner_nc,
        outer_nc,
        outermost=False,
        innermost=False,
        norm_layer="instance",
    ):
        super(UnetSkipConnectionDBlock, self).__init__()
        uprelu = nn.ReLU()
        upconv = nn.ConvTranspose2d(
            inner_nc, outer_nc, kernel_size=4, stride=2, padding=1
        )
        if norm_layer == "instance":
            upnorm = nn.InstanceNorm2d(outer_nc, affine=True)
        elif norm_layer == "batch":
            upnorm = nn.BatchNorm2d(outer_nc, affine=True)
        else:
            pass

        if outermost:
            up = [uprelu, upconv, nn.Tanh()]
            model = up
        elif innermost:
            up = [uprelu, upconv, upnorm]
            model = up
        else:
            up = [uprelu, upconv, upnorm]
            model = up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)
    
# Code is from https://github.com/KumapowerLIU/PD-GAN/blob/main/models/network/pconv.py
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