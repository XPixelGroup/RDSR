import math
from PIL import Image
import torch, cv2, os, sys, numpy as np, matplotlib.pyplot as plt

from ModelZoo.NN.srresnet_arch import *
from collections import OrderedDict


def load_model_feature(model_name):

    first_net = MSRResNet_details_firstob()
    b1_net = MSRResNet_details_b1ob()
    b2_net = MSRResNet_details_b2ob()
    b3_net = MSRResNet_details_b3ob()
    b4_net = MSRResNet_details_b4ob()
    b5_net = MSRResNet_details_b5ob()
    b6_net = MSRResNet_details_b6ob()
    b7_net = MSRResNet_details_b7ob()
    b8_net = MSRResNet_details_b8ob()
    b9_net = MSRResNet_details_b9ob()
    b10_net = MSRResNet_details_b10ob()
    b11_net = MSRResNet_details_b11ob()
    b12_net = MSRResNet_details_b12ob()
    b13_net = MSRResNet_details_b13ob()
    b14_net = MSRResNet_details_b14ob()
    b15_net = MSRResNet_details_b15ob()
    b16_net = MSRResNet_details_b16ob()
    hr_net = MSRResNet_details_hrob()
    last_net = MSRResNet_details_lastob()

    load_net = torch.load(model_name)
    load_net = load_net['params']
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'

    for k, v in load_net.items():
        # print(k)
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v

    first_net.load_state_dict(load_net_clean, strict=True)
    b1_net.load_state_dict(load_net_clean, strict=True)
    b2_net.load_state_dict(load_net_clean, strict=True)
    b3_net.load_state_dict(load_net_clean, strict=True)
    b4_net.load_state_dict(load_net_clean, strict=True)
    b5_net.load_state_dict(load_net_clean, strict=True)
    b6_net.load_state_dict(load_net_clean, strict=True)
    b7_net.load_state_dict(load_net_clean, strict=True)
    b8_net.load_state_dict(load_net_clean, strict=True)
    b9_net.load_state_dict(load_net_clean, strict=True)
    b10_net.load_state_dict(load_net_clean, strict=True)
    b11_net.load_state_dict(load_net_clean, strict=True)
    b12_net.load_state_dict(load_net_clean, strict=True)
    b13_net.load_state_dict(load_net_clean, strict=True)
    b14_net.load_state_dict(load_net_clean, strict=True)
    b15_net.load_state_dict(load_net_clean, strict=True)
    b16_net.load_state_dict(load_net_clean, strict=True)
    hr_net.load_state_dict(load_net_clean, strict=True)
    last_net.load_state_dict(load_net_clean, strict=True)

    return first_net, b1_net, b2_net, b3_net, b4_net, b5_net, b6_net, b7_net, \
           b8_net, b9_net, b10_net, b11_net,b12_net, b13_net, b14_net, b15_net, b16_net, hr_net, last_net
