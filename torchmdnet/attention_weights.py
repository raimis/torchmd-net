from torch_sparse import spspmm


num_layers = None
weights = None
edge_index = None
rollout_index = []
rollout_weights = []
_current_layer = None


def reset(layers):
    global num_layers, weights, edge_index, rollout_index, rollout_weights, _current_layer
    # initialize everything
    num_layers = layers
    weights = [[] for _ in range(num_layers)]
    edge_index = []
    _current_layer = 0


def append_weights(w):
    global num_layers, weights, edge_index, _current_layer
    weights[_current_layer].append(w.detach().cpu())
    assert len(weights[_current_layer]) == len(
        edge_index
    ), "Mismatch between stored attention weights and index"

    # increment layer counter
    _current_layer += 1
    if _current_layer == num_layers:
        # perform attention rollout at last layer
        _rollout()
        reset(num_layers)


def _rollout():
    # attention rollout (https://arxiv.org/pdf/2005.00928.pdf)
    global num_layers, weights, edge_index, rollout_index, rollout_weights, _current_layer
    assert _current_layer == num_layers, "Rollout has to happen at last layer"

    index = edge_index[-1]
    n_atoms = len(index.unique())

    # initialize cumulator with first layer
    cumulator_index = index
    cumulator_weights = weights[0][-1].mean(dim=1)

    # rollout over layers
    for layer in range(1, num_layers):
        # multiply attention matrix with cumulator
        cumulator_index, cumulator_weights = spspmm(
            index,
            weights[layer][-1].mean(dim=1),
            cumulator_index,
            cumulator_weights,
            n_atoms,
            n_atoms,
            n_atoms,
            coalesced=True,
        )
    # store rolled out attention weights
    rollout_index.append(cumulator_index)
    rollout_weights.append(cumulator_weights)


def store_idx(idx):
    global edge_index
    edge_index.append(idx.detach().cpu())
