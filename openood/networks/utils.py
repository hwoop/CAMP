import torch.backends.cudnn as cudnn

from .clip_vanilla import VanillaCLIP


def get_network(
    network_config,
):

    if network_config.name == 'vanilla_clip':
        net = VanillaCLIP(network_config)
    else:
        raise Exception('Unexpected Network Architecture!')

    cudnn.benchmark = True
    return net
