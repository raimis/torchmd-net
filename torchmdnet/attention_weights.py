import torch
from torch_sparse import spspmm


num_layers = None
weights = None
edge_index = None
rollout_index = None
rollout_weights = None
_current_layer = None


def create(layers):
    global num_layers, weights, edge_index, rollout_index, rollout_weights, _current_layer
    # initialize everything
    num_layers = layers
    weights = [[] for _ in range(num_layers)]
    edge_index = []
    rollout_index = []
    rollout_weights = []
    _current_layer = 0


def append_weights(w):
    global num_layers, weights, edge_index, _current_layer
    weights[_current_layer].append(w)
    assert len(weights[_current_layer]) == len(edge_index), 'Mismatch between stored attention weights and index'
    
    # increment layer counter
    _current_layer += 1
    if _current_layer == num_layers:
        # perform attention rollout at last layer
        _rollout()
        _current_layer = 0


def _rollout():
    # attention rollout (https://arxiv.org/pdf/2005.00928.pdf)
    global num_layers, weights, edge_index, rollout_index, rollout_weights, _current_layer
    assert _current_layer == num_layers, 'Rollout has to happen at last layer'

    index = edge_index[-1]
    n_atoms = len(index.unique())
    
    # initialize cumulator with first layer
    cumulator_index = index
    cumulator_weights = weights[0][-1]

    # rollout over layers
    for layer in range(1, num_layers):
        cw = weights[layer][-1]
        # multiply attention matrix with cumulator
        heads = []
        for head in range(cw.size(1)):
            inew, wnew = spspmm(index, cw[:,head], cumulator_index, cumulator_weights[:,head],
                                n_atoms, n_atoms, n_atoms, coalesced=True)
            heads.append(wnew)
        # update cumulator
        cumulator_index = inew
        cumulator_weights = torch.stack(heads, dim=1)
    # store rolled out attention weights
    rollout_index.append(cumulator_index)
    rollout_weights.append(cumulator_weights)


def store_idx(idx):
    global edge_index
    edge_index.append(idx)
