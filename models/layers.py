import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_, xavier_normal_


def weights_init(conv_weights, init, activate):
    if init is None:
        return
    if init == 'kaiming':
        if hasattr(activate, 'negative_slope'):
            kaiming_normal_(conv_weights, a=activate.negative_slope)
        else:
            kaiming_normal_(conv_weights, a=0)
    elif init == 'xavier_normal':
        xavier_normal_(conv_weights)


class Self_Attention(nn.Module):
    def __init__(self, in_dim):
        super(Self_Attention, self).__init__()
        self.in_dim = in_dim
        self.query_conv = conv(in_dim, in_dim // 8, 1, 1)
        self.key_conv = conv(in_dim, in_dim // 8, 1, 1)
        self.value_conv = conv(in_dim, in_dim, 1, 1)
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, fea_map):
        B, C, H, W = fea_map.size()
        query = self.query_conv(fea_map).view(B, -1, H*W).permute(0, 2, 1)  # BxNxC
        key = self.key_conv(fea_map).view(B, -1, H*W)
        energy = torch.bmm(query, key)  # BxNxN 相似度矩阵
        attention = self.softmax(energy)
        value = self.value_conv(fea_map).view(B, -1, H*W)

        out = torch.bmm(value, attention.permute(0, 2, 1))
        self_attention_map = out.view(B, C, H, W)
        out = self.gamma * self_attention_map + fea_map
        print(f'..............gamma的值为： {self.gamma.item()}')
        return out, attention

def conv(in_channel, out_channel, kernel, stride, pad=0, init='kaiming', norm='', active=nn.ReLU()):
    conv2d = nn.Conv2d(in_channel, out_channel, kernel, stride, pad)
    layers = [conv2d]
    weights_init(layers[0].weight, init, active)
    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channel))
    elif norm == 'layer':
        layers.append(nn.LayerNorm(out_channel))
    if active is not None:
        layers.append(active)
    return nn.Sequential(*layers)


def deconv(in_channel, out_channel, kernel, stride, pad=0, op=0, init='kaiming', norm='', active=nn.ReLU()):
    deconv2d = nn.ConvTranspose2d(in_channel, out_channel, kernel, stride, pad, op)
    layers = [deconv2d]
    weights_init(layers[0].weight, init, active)
    if norm == 'batch':
        layers.append(nn.BatchNorm2d(out_channel))
    elif norm == 'layer':
        layers.append(nn.LayerNorm(out_channel))
    if active is not None:
        layers.append(active)
    return nn.Sequential(*layers)


class Residual_Block(nn.Module):
    def __init__(self, in_channel, out_channel, kernel, stride, pad,
                 init='kaiming', batch_norm='', active=nn.ReLU()):
        super(Residual_Block, self).__init__()

        self.activate = active
        if out_channel is None:
            out_channel = in_channel // stride
        if stride == 1:
            self.shortcut = nn.Identity()
        layers = [
            conv(in_channel, out_channel, kernel, 1,
                 pad if pad is not None else (kernel-1)//2, init,
                 norm=batch_norm, active=active),
            conv(in_channel, out_channel, kernel, stride,
                 pad if pad is not None else (kernel-1)//2, init,
                 norm=batch_norm, active=None)
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        short_cut = self.shortcut(x)
        out = self.layers(x)

        return self.activate(short_cut + out)



