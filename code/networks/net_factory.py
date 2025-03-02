from networks.unet import UNet_HL
from networks.VNet import VNet_HL
import torch.nn as nn

def net_factory(net_type="unet", in_chns=1, class_num=2, mode = "train", tsne=0):
    if net_type == "unet" and mode == "train":
        net = UNet_HL(in_chns=in_chns, class_num=class_num).cuda()
    if net_type == "VNet_HL" and mode == "train" and tsne==0:
        net = VNet_HL(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=True).cuda()
    if net_type == "VNet_HL" and mode == "test" and tsne==0:
        net = VNet_HL(n_channels=in_chns, n_classes=class_num, normalization='batchnorm', has_dropout=False).cuda()
    return net


def M3HL_net(in_chns=1, class_num=2, ema=False):
    net = UNet_HL(in_chns=in_chns, class_num=class_num).cuda()
    if ema:
        for param in net.parameters():
            param.detach_()
    return net
