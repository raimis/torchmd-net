import torch
from torch import nn
from torch_geometric.nn import radius_graph, knn_graph, MessagePassing
from torch_geometric.utils import softmax
from torch_scatter import scatter
from torch_sparse import coalesce
from torchmdnet.models.utils import (NeighborEmbedding, CosineCutoff,
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
    """

    def __init__(self, hidden_channels=128, num_filters=128,
                 num_layers=6, num_rbf=50, rbf_type='expnorm',
                 trainable_rbf=True, activation='silu', neighbor_embedding=True,
                 num_heads=8, cutoff_lower=0.0, cutoff_upper=5.0, max_z=100):
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
        self.neighbor_embedding = neighbor_embedding
        self.num_heads = num_heads
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.max_z = max_z

        act_class = act_class_mapping[activation]

        self.embedding = nn.Embedding(self.max_z, hidden_channels)

        self.distance_expansion = rbf_class_mapping[rbf_type](
            cutoff_lower, cutoff_upper, num_rbf, trainable_rbf
        )
        self.neighbor_embedding = NeighborEmbedding(
            hidden_channels, num_rbf, cutoff_lower, cutoff_upper, self.max_z
        ) if neighbor_embedding else None

        self.interactions = nn.ModuleList()
        for _ in range(num_layers):
            block = InteractionBlock(hidden_channels, num_rbf, num_filters, num_heads,
                                     act_class, cutoff_lower, cutoff_upper)
            self.interactions.append(block)

    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.interactions:
            interaction.reset_parameters()

    def forward(self, z, pos, batch=None):
        x = self.embedding(z)

        edge_index = radius_graph(pos, r=self.cutoff_upper, batch=batch)
        row, col = edge_index
        edge_weight = (pos[row] - pos[col]).norm(dim=-1)
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
                 activation, cutoff_lower, cutoff_upper):
        super(InteractionBlock, self).__init__()

        self.conv = ElemConv(hidden_channels, num_filters, num_rbf,
                             activation, cutoff_lower, cutoff_upper)
        self.attn = ElemAttn(hidden_channels, num_filters, num_heads, activation)

        self.act = activation()
        self.lin = nn.Linear(hidden_channels, hidden_channels)

    def reset_parameters(self):
        self.conv.reset_parameters()
        nn.init.xavier_uniform_(self.lin.weight)
        self.lin.bias.data.fill_(0)

    def forward(self, x, z, edge_index, edge_weight, edge_attr):
        y, elem_index = self.conv(x, z, edge_index, edge_weight, edge_attr)
        y = self.attn(y, elem_index)

        y = self.act(y)
        y = self.lin(y)
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
        elem_index, y = coalesce(elem_index, inputs, m=self.max_z, n=dim_size,
                                 op=self.aggr)
        idx_sort = elem_index[1].argsort()
        return y[idx_sort], elem_index[:,idx_sort]


class ElemAttn(MessagePassing):
    def __init__(self, out_channels, num_filters, num_heads, activation,
                 bias=True, max_z=100):
        super(ElemAttn, self).__init__(aggr='add', node_dim=0)
        assert num_filters % num_heads == 0, (f'The number of filters ({num_filters}) '
                                              f'must be evenly divisible by the number '
                                              f'of attention heads ({num_heads})')
        self.num_heads = num_heads
        self.bias = bias
        self.max_z = max_z

        self.q_proj = nn.Linear(num_filters, num_filters, bias=bias)
        self.k_proj = nn.Linear(num_filters, num_filters, bias=bias)
        self.v_proj = nn.Linear(num_filters, num_filters, bias=bias)
        self.o_proj = nn.Linear(num_filters, num_filters, bias=bias)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.o_proj.weight)
        if self.bias:
            self.q_proj.bias.data.fill_(0)
            self.k_proj.bias.data.fill_(0)
            self.v_proj.bias.data.fill_(0)
            self.o_proj.bias.data.fill_(0)

    def forward(self, x, elem_index):
        bs = x.size(0)

        q = self.q_proj(x).reshape(bs, self.num_heads, -1)
        k = self.k_proj(x).reshape(bs, self.num_heads, -1)
        v = self.v_proj(x).reshape(bs, self.num_heads, -1)
        
        edge_index = knn_graph(x, self.max_z, batch=elem_index[1], loop=True)
        out = self.propagate(edge_index, q=q, k=k, v=v, index=elem_index[1])

        out = self.o_proj(out.reshape(bs, -1))
        out = scatter(out, elem_index[1], dim=0)
        return out

    def message(self, q_i, k_j, v_j, index):
        attn = (q_i * k_j).sum(dim=-1)
        attn = softmax(attn, index)
        out = v_j * attn.unsqueeze(2)
        return out
