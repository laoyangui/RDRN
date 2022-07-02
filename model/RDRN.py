import torch
import torch.nn as nn


def make_model(args, parent=False):
    model = RDRN()
    return model

class RDRN(nn.Module):
    def __init__(self, in_nc=3, nf=52, num_modules=6, out_nc=3, upscale=4):
        super(RDRN, self).__init__()

        self.fea_conv = conv_layer(in_nc, nf, kernel_size=3)

        self.B1 = DRB(in_channels=nf)
        self.B2 = DRB(in_channels=nf)
        self.B3 = DRB(in_channels=nf)
        self.B4 = DRB(in_channels=nf)
        self.B5 = DRB(in_channels=nf)
        self.B6 = DRB(in_channels=nf)
        self.c = conv_block(nf * num_modules, nf, kernel_size=1, act_type='lrelu')

        self.LR_conv = conv_layer(nf, nf, kernel_size=3)

        upsample_block = pixelshuffle_block
        self.upsampler = upsample_block(nf, out_nc, upscale_factor=upscale)
        self.scale_idx = 0


    def forward(self, input):
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #
        # input = input.to(device)
        out_fea = self.fea_conv(input)
        b, c, h, w = input.size()
        device = torch.device('cuda:0')

        ind1 = torch.arange(w - 1, -1, -1).to(device)
        ind2 = torch.arange(h - 1, -1, -1).to(device)

        out_B1 = self.B1(out_fea)

        input2 = torch.transpose(out_B1, dim0=2, dim1=3)
        out_B2 = self.B2(input2)
        out_B2 = torch.transpose(out_B2, dim0=2, dim1=3)

        #90counterclockwise
        input3 = torch.transpose(out_B2, dim0=2, dim1=3)
        input3 = input3.index_select(2, ind1).clone()
        out_B3 = self.B3(input3)
        out_B3 = out_B3.index_select(2, ind1).clone()
        out_B3 = torch.transpose(out_B3, dim0=2, dim1=3)

        #180counterclockwise
        input4 = torch.transpose(out_B3, dim0=2, dim1=3)
        input4 = input4.index_select(2, ind1).clone()
        input4 = torch.transpose(input4, dim0=2, dim1=3)
        input4 = input4.index_select(2, ind2).clone()
        out_B4 = self.B4(input4)
        out_B4 = out_B4.index_select(2, ind2).clone()
        out_B4 = torch.transpose(out_B4, dim0=2, dim1=3)
        out_B4 = out_B4.index_select(2, ind1).clone()
        out_B4 = torch.transpose(out_B4, dim0=2, dim1=3)

        #270counterclockwise
        input5 = torch.transpose(out_B4, dim0=2, dim1=3)
        input5 = input5.index_select(3, ind2).clone()
        out_B5 = self.B5(input5)
        out_B5 = out_B5.index_select(3, ind2).clone()
        out_B5 = torch.transpose(out_B5, dim0=2, dim1=3)

        #mirroring
        input6 = out_B5.index_select(3, ind1).clone()
        out_B6 = self.B6(input6)
        out_B6 = out_B6.index_select(3, ind1).clone()

        out_B = self.c(torch.cat([out_B1, out_B2, out_B3, out_B4, out_B5, out_B6], dim=1))
        out_lr = self.LR_conv(out_B) + out_fea

        output = self.upsampler(out_lr)

        return output

    def load_state_dict1(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx


def conv_layer(in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1):
    padding = int((kernel_size - 1) / 2) * dilation
    return nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, bias=True, dilation=dilation,
                     groups=groups)


def norm(norm_type, nc):
    norm_type = norm_type.lower()
    if norm_type == 'batch':
        layer = nn.BatchNorm2d(nc, affine=True)
    elif norm_type == 'instance':
        layer = nn.InstanceNorm2d(nc, affine=False)
    else:
        raise NotImplementedError('normalization layer [{:s}] is not found'.format(norm_type))
    return layer


def pad(pad_type, padding):
    pad_type = pad_type.lower()
    if padding == 0:
        return None
    if pad_type == 'reflect':
        layer = nn.ReflectionPad2d(padding)
    elif pad_type == 'replicate':
        layer = nn.ReplicationPad2d(padding)
    else:
        raise NotImplementedError('padding layer [{:s}] is not implemented'.format(pad_type))
    return layer


def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding


def conv_block(in_nc, out_nc, kernel_size, stride=1, dilation=1, groups=1, bias=True,
               pad_type='zero', norm_type=None, act_type='relu'):
    padding = get_valid_padding(kernel_size, dilation)
    p = pad(pad_type, padding) if pad_type and pad_type != 'zero' else None
    padding = padding if pad_type == 'zero' else 0

    c = nn.Conv2d(in_nc, out_nc, kernel_size=kernel_size, stride=stride, padding=padding,
                  dilation=dilation, bias=bias, groups=groups)
    a = activation(act_type) if act_type else None
    n = norm(norm_type, out_nc) if norm_type else None
    return sequential(p, c, n, a)


def activation(act_type, inplace=True, neg_slope=0.05, n_prelu=1):
    act_type = act_type.lower()
    if act_type == 'relu':
        layer = nn.ReLU(inplace)
    elif act_type == 'lrelu':
        layer = nn.LeakyReLU(neg_slope, inplace)
    elif act_type == 'prelu':
        layer = nn.PReLU(num_parameters=n_prelu, init=neg_slope)
    else:
        raise NotImplementedError('activation layer [{:s}] is not found'.format(act_type))
    return layer


class ShortcutBlock(nn.Module):
    def __init__(self, submodule):
        super(ShortcutBlock, self).__init__()
        self.sub = submodule

    def forward(self, x):
        output = x + self.sub(x)
        return output


def mean_channels(F):
    assert (F.dim() == 4)
    spatial_sum = F.sum(3, keepdim=True).sum(2, keepdim=True)
    return spatial_sum / (F.size(2) * F.size(3))


def stdv_channels(F):
    assert (F.dim() == 4)
    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)


def sequential(*args):
    if len(args) == 1:
        if isinstance(args[0], OrderedDict):
            raise NotImplementedError('sequential does not support OrderedDict input.')
        return args[0]
    modules = []
    for module in args:
        if isinstance(module, nn.Sequential):
            for submodule in module.children():
                modules.append(submodule)
        elif isinstance(module, nn.Module):
            modules.append(module)
    return nn.Sequential(*modules)

class DRB(nn.Module):
    def __init__(self, in_channels, distillation_rate=0.25):
        super(DRB, self).__init__()
        self.dc = self.distilled_channels = in_channels // 2
        self.rc = self.remaining_channels = in_channels
        self.c1_d = conv_layer(in_channels, self.dc, 1)
        # self.c1_r = conv_layer(in_channels, self.rc, 3)
        self.branch1 = nn.Sequential(
            BasicConv(self.rc, self.rc, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv(self.rc, self.rc, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            # BasicSepConv(self.rc, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        )
        self.c2_d = conv_layer(self.remaining_channels, self.dc, 1)
        # self.c2_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.branch2 = nn.Sequential(
            BasicConv(self.rc, self.rc, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv(self.rc, self.rc, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            # BasicSepConv(self.rc, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        )
        self.c3_d = conv_layer(self.remaining_channels, self.dc, 1)
        # self.c3_r = conv_layer(self.remaining_channels, self.rc, 3)
        self.branch3 = nn.Sequential(
            BasicConv(self.rc, self.rc, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv(self.rc, self.rc, kernel_size=(3, 1), stride=1, padding=(1, 0)),
            BasicConv(self.rc, self.rc, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv(self.rc, self.rc, kernel_size=(3, 1), stride=1, padding=(1, 0)),

            # BasicSepConv(self.rc, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
        )
        self.c4 = conv_layer(self.remaining_channels, self.dc, 3)
        self.act = activation('lrelu', neg_slope=0.05)
        self.c5 = conv_layer(self.dc * 4, in_channels, 1)
        self.eca = eca_layer(in_channels)


    def forward(self, input):
        distilled_c1 = self.act(self.c1_d(input))
        r_c1 = (self.branch1(input))
        r_c1 = self.act(r_c1 + input)

        distilled_c2 = self.act(self.c2_d(r_c1))
        r_c2 = (self.branch2(r_c1))
        r_c2 = self.act(r_c2 + r_c1)

        distilled_c3 = self.act(self.c3_d(r_c2))
        r_c3 = (self.branch3(r_c2))
        r_c3 = self.act(r_c3 + r_c2)

        r_c4 = self.act(self.c4(r_c3))

        out = torch.cat([distilled_c1, distilled_c2, distilled_c3, r_c4], dim=1)
        out_fused = self.eca(self.c5(out))

        return out_fused

def pixelshuffle_block(in_channels, out_channels, upscale_factor=2, kernel_size=3, stride=1):
    conv = conv_layer(in_channels, out_channels * (upscale_factor ** 2), kernel_size, stride)
    pixel_shuffle = nn.PixelShuffle(upscale_factor)
    return sequential(conv, pixel_shuffle)

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class BasicSepConv(nn.Module):

    def __init__(self, in_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicSepConv, self).__init__()
        self.out_channels = in_planes
        self.conv = nn.Conv2d(in_planes, in_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups = in_planes, bias=bias)
        self.bn = nn.BatchNorm2d(in_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

def stdv_channels(F):
    assert(F.dim() == 4)
    u = F.dim()

    F_mean = mean_channels(F)
    F_variance = (F - F_mean).pow(2).sum(3, keepdim=True).sum(2, keepdim=True) / (F.size(2) * F.size(3))
    return F_variance.pow(0.5)

class eca_layer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(eca_layer, self).__init__()
        self.contrast = stdv_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: input features with shape [b, c, h, w]
        b, c, h, w = x.size()

        # feature descriptor on the global spatial information
        y = self.avg_pool(x) + self.contrast(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        ye = y.expand_as(x)

        return x * ye