import torch
from torch import nn
from torch_geometric.nn import radius_graph, knn_graph, MessagePassing
from torch_scatter import scatter
from torchmdnet.models.utils import (NeighborEmbedding, CosineCutoff, Distance,
                                     rbf_class_mapping, act_class_mapping)


class ElementTransformer(nn.Module):
    r"""
    Args:
        hidden_channels (int, optional): Hidden embedding size.
            (default: :obj:`128`)
        num_filters (int, optional): The number of filters to use.
            (default: :obj:`128`)
        num_layers (int, optional): The number of interaction blocks.
            (default: :obj:`6`)
        num_rbf (int, optional): The number of radial basis functions :math:`\mu`.
            (default: :obj:`50`)
        rbf_type (string, optional): The type of radial basis function to use.
            (default: :obj:`"expnorm"`)
        trainable_rbf (bool, optional): Whether to train RBF parameters with
            backpropagation. (default: :obj:`True`)
        activation (string, optional): The type of activation function to use.
            (default: :obj:`"silu"`)
        attn_activation (string, optional): The type of activation function to use
            inside the attention mechanism. (default: :obj:`"silu"`)
        neighbor_embedding (bool, optional): Whether to perform an initial neighbor
            embedding step. (default: :obj:`True`)
        num_heads (int, optional): Number of attention heads.
            (default: :obj:`8`)
        cutoff_lower (float, optional): Lower cutoff distance for interatomic interactions.
            (default: :obj:`0.0`)
        cutoff_upper (float, optional): Upper cutoff distance for interatomic interactions.
            (default: :obj:`5.0`)
        max_z (int, optional): Maximum atomic number. Used for initializing embeddings.
            (default: :obj:`100`)
        max_num_neighbors (int, optional): Maximum number of neighbors to return for a
            given node/atom when constructing the molecular graph during forward passes.
            This attribute is passed to the torch_cluster radius_graph routine keyword
            max_num_neighbors, which normally defaults to 32. Users should set this to
            higher values if they are using higher upper distance cutoffs and expect more
            than 32 neighbors per node/atom.
            (default: :obj:`32`)
    """

    def __init__(self, hidden_channels=128, num_filters=128, num_layers=6, num_rbf=50,
                 rbf_type='expnorm', trainable_rbf=True, activation='silu',
                 attn_activation='silu', neighbor_embedding=True, num_heads=8,
                 cutoff_lower=0.0, cutoff_upper=5.0, max_z=100, max_num_neighbors=32):
        super(ElementTransformer, self).__init__()

        assert rbf_type in rbf_class_mapping, (f'Unknown RBF type "{rbf_type}". '
                                               f'Choose from {", ".join(rbf_class_mapping.keys())}.')
        assert activation in act_class_mapping, (f'Unknown activation function "{activation}". '
                                                 f'Choose from {", ".join(act_class_mapping.keys())}.')

        self.hidden_channels = hidden_channels
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.num_rbf = num_rbf
        self.rbf_type = rbf_type
        self.trainable_rbf = trainable_rbf
        self.activation = activation
        self.attn_activation = attn_activation
        self.neighbor_embedding = neighbor_embedding
        self.num_heads = num_heads
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.max_z = max_z

        act_class = act_class_mapping[activation]
        attn_act_class = act_class_mapping[attn_activation]

        self.embedding = nn.Embedding(self.max_z, hidden_channels)

        self.distance = Distance(cutoff_lower, cutoff_upper,
                                 max_num_neighbors=max_num_neighbors)
        self.distance_expansion = rbf_class_mapping[rbf_type](
            cutoff_lower, cutoff_upper, num_rbf, trainable_rbf
        )
        self.neighbor_embedding = NeighborEmbedding(
            hidden_channels, num_rbf, cutoff_lower, cutoff_upper, self.max_z
        ) if neighbor_embedding else None

        self.interactions = nn.ModuleList()
        for _ in range(num_layers):
            block = InteractionBlock(hidden_channels, num_rbf, num_filters, num_heads,
                                     act_class, attn_act_class, cutoff_lower, cutoff_upper)
            self.interactions.append(block)

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()

    def forward(self, z, pos, batch=None):
        x = self.embedding(z)

        edge_index, edge_weight = self.distance(pos, batch)
        edge_attr = self.distance_expansion(edge_weight)

        if self.neighbor_embedding:
            x = self.neighbor_embedding(z, x, edge_index, edge_weight, edge_attr)

        for interaction in self.interactions:
            x = x + interaction(x, z, edge_index, edge_weight, edge_attr)
        
        return x, z, pos, batch

    def __repr__(self):
        return (f'{self.__class__.__name__}('
                f'hidden_channels={self.hidden_channels}, '
                f'num_filters={self.num_filters}, '
                f'num_layers={self.num_layers}, '
                f'num_rbf={self.num_rbf}, '
                f'rbf_type={self.rbf_type}, '
                f'trainable_rbf={self.trainable_rbf}, '
                f'activation={self.activation}, '
                f'neighbor_embedding={self.neighbor_embedding}, '
                f'cutoff_lower={self.cutoff_lower}, '
                f'cutoff_upper={self.cutoff_upper})')


class InteractionBlock(nn.Module):
    def __init__(self, hidden_channels, num_rbf, num_filters, num_heads,
                 activation, attn_activation, cutoff_lower, cutoff_upper):
        super(InteractionBlock, self).__init__()

        self.conv = ElemConv(hidden_channels, num_filters, num_rbf,
                             activation, cutoff_lower, cutoff_upper)
        self.attn = ElemAttn(hidden_channels, num_filters, num_heads,
                             activation, attn_activation)

    def reset_parameters(self):
        self.conv.reset_parameters()
        self.attn.reset_parameters()

    def forward(self, x, z, edge_index, edge_weight, edge_attr):
        y, elem_index = self.conv(x, z, edge_index, edge_weight, edge_attr)
        y = self.attn(y, elem_index)
        return y


class ElemConv(MessagePassing):
    def __init__(self, in_channels, num_filters, num_rbf, activation,
                 cutoff_lower, cutoff_upper, max_z=100):
        super(ElemConv, self).__init__(aggr='add', node_dim=0)
        self.max_z = max_z

        self.lin1 = nn.Linear(in_channels, num_filters, bias=False)
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.filter_net = nn.Sequential(
            nn.Linear(num_rbf, num_filters),
            activation(),
            nn.Linear(num_filters, num_filters),
        )

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin1.weight)
        nn.init.xavier_uniform_(self.filter_net[0].weight)
        self.filter_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.filter_net[2].weight)
        self.filter_net[2].bias.data.fill_(0)

    def forward(self, x, z, edge_index, edge_weight, edge_attr):
        C = self.cutoff(edge_weight)
        W = self.filter_net(edge_attr) * C.view(-1, 1)

        x = self.lin1(x)
        y, elem_index = self.propagate(edge_index, x=x, z=z, W=W)
        return y, elem_index

    def message(self, x_j, W):
        return x_j * W

    def aggregate(self, inputs, index, dim_size, z_j):
        elem_index = torch.stack([z_j, index])

        # aggregate atoms of each element
        elem_index, unique_indices = elem_index.unique(return_inverse=True, dim=1)
        y = scatter(inputs, unique_indices, dim=0)

        # sort indices for later
        idx_sort = elem_index[1].argsort()
        return y[idx_sort], elem_index[:,idx_sort]


class ElemAttn(nn.Module):
    def __init__(self, out_channels, num_filters, num_heads, activation,
                 attn_activation, bias=True):
        super(ElemAttn, self).__init__()
        assert num_filters % num_heads == 0, (f'The number of filters ({num_filters}) '
                                              f'must be evenly divisible by the number '
                                              f'of attention heads ({num_heads})')
        self.num_heads = num_heads
        self.bias = bias

        self.q_proj = nn.Linear(num_filters, num_filters, bias=bias)
        self.k_proj = nn.Linear(num_filters, num_filters, bias=bias)
        self.v_proj = nn.Linear(num_filters, num_filters, bias=bias)
        self.o_proj1 = nn.Linear(num_filters, num_filters, bias=bias)
        self.o_proj2 = nn.Linear(num_filters, num_filters, bias=bias)
        
        self.act = activation()
        self.attn_act = activation()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj1.weight)
        nn.init.xavier_uniform_(self.o_proj2.weight)
        if self.bias:
            self.q_proj.bias.data.fill_(0)
            self.k_proj.bias.data.fill_(0)
            self.v_proj.bias.data.fill_(0)
            self.o_proj1.bias.data.fill_(0)
            self.o_proj2.bias.data.fill_(0)

    def forward(self, x, elem_index):
        bs = x.size(0)

        q = self.q_proj(x).reshape(bs, self.num_heads, -1).transpose(0, 1)
        k = self.k_proj(x).reshape(bs, self.num_heads, -1).transpose(0, 1)
        v = self.v_proj(x).reshape(bs, self.num_heads, -1).transpose(0, 1)

        # This computes way more than it should. Ideally, attention weights should only
        # be calculated inside each molecule (i.e. inside elem_index[1]). Currently, attention
        # weights are computed for each pair of atoms in the whole batch. Maybe a sparse matrix
        # multiplication would work here. Otherwise, this operation resembles "scatter_matmul",
        # which does not exist
        attn = torch.matmul(q, k.transpose(1, 2))
        attn = self.attn_act(attn)

        # Mask out attention weights outside of the molecule. This way of creating
        # the block diagonal matrix is not also ideal
        counts = elem_index[1].unique(return_counts=True)[1]
        blocks = torch.block_diag(*[torch.ones(n, n, device=attn.device) for n in counts])
        attn = attn * blocks

        # weight values by attention weights
        out = torch.matmul(attn, v)

        # slow way of computing attention weights sequentially for each molecule:
        # out = elementwise_attention(q, k, v, elem_index[1], self.attn_act)

        out = out.transpose(0, 1).reshape(bs, -1)
        out = self.o_proj1(out)
        # TODO: activation here?

        # aggregate atom types
        out = scatter(out, elem_index[1], dim=0)
        out = self.o_proj2(out)
        return self.act(out)


def elementwise_attention(q, k, v, index, act):
    # this function might be quick when implemented as a CUDA kernel,
    # currently it is very slow due looping over each molecule in the batch
    result = torch.empty_like(q)
    for i in torch.unique(index):
        mask = index == i
        attn = torch.matmul(q[:,mask], k[:,mask].transpose(1, 2))
        attn = act(attn)
        result[:,mask] = torch.matmul(attn, v[:,mask])
    return result
