num_layers = None
weights = None
edge_index = None
_current_layer = None


def create(layers):
    global num_layers, weights, edge_index, _current_layer
    num_layers = layers
    weights = [[] for _ in range(num_layers)]
    edge_index = []
    _current_layer = 0


def append_weights(w):
    global num_layers, weights, _current_layer
    weights[_current_layer].append(w)
    _current_layer = (_current_layer + 1) % num_layers


def append_idx(idx):
    global edge_index
    edge_index.append(idx)
