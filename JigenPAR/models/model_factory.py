from models import caffenet

nets_map = {
    'caffenet': caffenet.caffenet,
    'parnet': caffenet.parnet,
    'parnet_H': caffenet.parnet_H,
    'parnet_B': caffenet.parnet_B,
    'parnet_M': caffenet.parnet_M,
}


def get_network(name):
    if name not in nets_map:
        raise ValueError('Name of network unknown %s' % name)

    def get_network_fn(**kwargs):
        return nets_map[name](**kwargs)

    return get_network_fn
