import os
import torch
from collections import OrderedDict


NN_LIST = [
    'RCAN',
    'CARN',
    'RRDBNet',
    'RNAN', 
    'SAN',
]


MODEL_LIST = {
    'RCAN': {
        'Base': 'RCAN.pt',
    },
    'CARN': {
        'Base': 'CARN_7400.pth',
    },
    'RRDBNet': {
        'Base': 'RRDBNet_PSNR_SRx4_DF2K_official-150ff491.pth',
    },
    'SAN': {
        'Base': 'SAN_BI4X.pt',
    },
    'RNAN': {
        'Base': 'RNAN_SR_F64G10P48BIX4.pt',
    },
}

def print_network(model, model_name):
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print('Network [%s] was created. Total number of parameters: %.1f kelo. '
          'To see the architecture, do print(network).'
          % (model_name, num_params / 1000))


def get_model(model_name, factor=4, num_channels=3):
    """
    All the models are defaulted to be X4 models, the Channels is defaulted to be RGB 3 channels.
    :param model_name:
    :param factor:
    :param num_channels:
    :return:
    """
    print(f'Getting SR Network {model_name}')
    #if model_name.split('-')[0] in NN_LIST:

    if 'RCAN' in model_name:
        from .NN.rcan import RCAN
        net = RCAN(factor=factor, num_channels=num_channels)

    elif model_name == 'CARN':
        from .CARN.carn import CARNet
        net = CARNet(factor=factor, num_channels=num_channels)

    elif model_name == 'RRDBNet':
        from .NN.rrdbnet import RRDBNet
        net = RRDBNet(num_in_ch=num_channels, num_out_ch=num_channels)

    elif model_name == 'SAN':
        from .NN.san import SAN
        net = SAN(factor=factor, num_channels=num_channels)

    elif model_name == 'RNAN':
        from .NN.rnan import RNAN
        net = RNAN(factor=factor, num_channels=num_channels)

    # elif 'SRResNet' in model_name:
    #     from .NN.srresnet_arch import MSRResNet
    #     net=MSRResNet()

    elif 'PAN' in model_name:
        from .NN.pan import PAN
        net=PAN()

    elif 'SRCNN' in model_name:
        from .NN.srcnn import SRCNN
        net=SRCNN()

    elif ('MSRResNet' in model_name) and ('ob' not in model_name):
        from .NN.srresnet_arch import MSRResNet_wGR_details
        net=MSRResNet_wGR_details()

    elif ('RealSRResNet' in model_name) and ('ob' not in model_name):
        from .NN.srresnet_arch import MSRResNet_wGR_details
        net=MSRResNet_wGR_details()

    # elif 'firstob' in model_name:
    #     from .NN.srresnet_arch import MSRResNet_details_firstob
    #     net=MSRResNet_details_firstob()
    #
    # elif 'b1ob' in model_name:
    #     from .NN.srresnet_arch import MSRResNet_details_b1ob
    #     net=MSRResNet_details_b1ob()
    #
    # elif 'b2ob' in model_name:
    #     from .NN.srresnet_arch import MSRResNet_details_b2ob
    #     net=MSRResNet_details_b2ob()
    #
    # elif 'b3ob' in model_name:
    #     from .NN.srresnet_arch import MSRResNet_details_b3ob
    #     net = MSRResNet_details_b3ob()
    #
    # elif 'b4ob' in model_name:
    #     from .NN.srresnet_arch import MSRResNet_details_b4ob
    #     net = MSRResNet_details_b4ob()
    #
    # elif 'b5ob' in model_name:
    #     from .NN.srresnet_arch import MSRResNet_details_b5ob
    #     net = MSRResNet_details_b5ob()
    #
    # elif 'b6ob' in model_name:
    #     from .NN.srresnet_arch import MSRResNet_details_b6ob
    #     net = MSRResNet_details_b6ob()
    #
    # elif 'b7ob' in model_name:
    #     from .NN.srresnet_arch import MSRResNet_details_b7ob
    #     net = MSRResNet_details_b7ob()
    #
    # elif 'b8ob' in model_name:
    #     from .NN.srresnet_arch import MSRResNet_details_b8ob
    #     net = MSRResNet_details_b8ob()
    #
    # elif 'b9ob' in model_name:
    #     from .NN.srresnet_arch import MSRResNet_details_b9ob
    #     net = MSRResNet_details_b9ob()
    #
    # elif 'b10ob' in model_name:
    #     from .NN.srresnet_arch import MSRResNet_details_b10ob
    #     net = MSRResNet_details_b10ob()
    #
    # elif 'b11ob' in model_name:
    #     from .NN.srresnet_arch import MSRResNet_details_b11ob
    #     net = MSRResNet_details_b11ob()
    #
    # elif 'b12ob' in model_name:
    #     from .NN.srresnet_arch import MSRResNet_details_b12ob
    #     net = MSRResNet_details_b12ob()
    #
    # elif 'b13ob' in model_name:
    #     from .NN.srresnet_arch import MSRResNet_details_b13ob
    #     net = MSRResNet_details_b13ob()
    #
    # elif 'b14ob' in model_name:
    #     from .NN.srresnet_arch import MSRResNet_details_b14ob
    #     net = MSRResNet_details_b14ob()
    #
    # elif 'b15ob' in model_name:
    #     from .NN.srresnet_arch import MSRResNet_details_b15ob
    #     net = MSRResNet_details_b15ob()
    #
    # elif 'b16ob' in model_name:
    #     from .NN.srresnet_arch import MSRResNet_details_b16ob
    #     net = MSRResNet_details_b16ob()
    #
    # elif 'hrob' in model_name:
    #     from .NN.srresnet_arch import MSRResNet_details_hrob
    #     net = MSRResNet_details_hrob()
    #
    # elif 'lastob' in model_name:
    #     from .NN.srresnet_arch import MSRResNet_details_lastob
    #     net = MSRResNet_details_lastob()

    elif 'lastob_onetoall' in model_name:
        from .NN.srresnet_arch import MSRResNet_details_lastob_onetoall
        net = MSRResNet_details_lastob_onetoall()

    elif 'lastob_range_onetoall' in model_name:
        from .NN.srresnet_arch import MSRResNet_details_lastob_range_onetoall
        net = MSRResNet_details_lastob_range_onetoall()

    else:
        raise NotImplementedError()

    print_network(net, model_name) #统计模型参数量
    return net
    # else:
    #     raise NotImplementedError()


def load_model_RCAN(model_loading_name):
    """
    :param model_loading_name: model_name-training_name
    :return:
    """

    model_name='RCAN'
    net = get_model(model_name)
    #state_dict_path='/data0/xtkong/ClassSR/experiments/RCAN_20b_64c_10g_x4_alltype/models/100000_G.pth'
    print(f'Loading model {model_loading_name} for {model_name} network.')
    state_dict = torch.load(model_loading_name, map_location='cpu')
    net.load_state_dict(state_dict)
    return net


def load_model_now(model_name,model_loading_name):
    """
    :param model_loading_name: model_name-training_name
    :return:
    """

    net = get_model(model_name)
    #state_dict_path='/data0/xtkong/ClassSR/experiments/RCAN_20b_64c_10g_x4_alltype/models/100000_G.pth'
    print(f'Loading model {model_loading_name} for {model_name} network.')

    load_net = torch.load(model_loading_name)

    load_net = load_net['params']
    # print(load_net['params'])
    #print(load_net)
    load_net_clean = OrderedDict()  # remove unnecessary 'module.'


    for k, v in load_net.items():
        # print(k)
        if k.startswith('module.'):
            load_net_clean[k[7:]] = v
        else:
            load_net_clean[k] = v


    net.load_state_dict(load_net_clean, strict=True)
    return net






