import math
import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import radius_graph, MessagePassing


class NeighborEmbedding(MessagePassing):
    def __init__(self, hidden_channels, num_rbf, cutoff_lower,
                 cutoff_upper, max_z=100):
        super(NeighborEmbedding, self).__init__(aggr='add')
        self.embedding = nn.Embedding(max_z, hidden_channels)
        self.distance_proj = nn.Linear(num_rbf, hidden_channels)
        self.combine = nn.Linear(hidden_channels * 2, hidden_channels)
        self.cutoff = CosineCutoff(cutoff_lower, cutoff_upper)

        self.reset_parameters()

    def reset_parameters(self):
        self.embedding.reset_parameters()
        nn.init.xavier_uniform_(self.distance_proj.weight)
        nn.init.xavier_uniform_(self.combine.weight)
        self.distance_proj.bias.data.fill_(0)
        self.combine.bias.data.fill_(0)

    def forward(self, z, x, edge_index, edge_weight, edge_attr):
        C = self.cutoff(edge_weight)
        W = self.distance_proj(edge_attr) * C.view(-1, 1)

        x_neighbors = self.embedding(z)
        x_neighbors = self.propagate(edge_index, x=x_neighbors, W=W)
        x_neighbors = self.combine(torch.cat([x, x_neighbors], dim=1))
        return x_neighbors

    def message(self, x_j, W):
        return x_j * W


class GaussianSmearing(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=True):
        super(GaussianSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        offset, coeff = self._initial_params()
        if trainable:
            self.register_parameter('coeff', nn.Parameter(coeff))
            self.register_parameter('offset', nn.Parameter(offset))
        else:
            self.register_buffer('coeff', coeff)
            self.register_buffer('offset', offset)

    def _initial_params(self):
        offset = torch.linspace(self.cutoff_lower, self.cutoff_upper, self.num_rbf)
        coeff = -0.5 / (offset[1] - offset[0]) ** 2
        return offset, coeff

    def reset_parameters(self):
        offset, coeff = self._initial_params()
        self.offset.data.copy_(offset)
        self.coeff.data.copy_(coeff)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))


class ExpNormalSmearing(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=True):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(0, cutoff_upper)

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter('means', nn.Parameter(means))
            self.register_parameter('betas', nn.Parameter(betas))
        else:
            self.register_buffer('means', means)
            self.register_buffer('betas', betas)

    def _initial_params(self):
        # initialize means and betas according to the default values in PhysNet
        # https://pubs.acs.org/doi/10.1021/acs.jctc.9b00181
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff_upper + self.cutoff_lower))
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(-self.betas * (torch.exp(-dist + self.cutoff_lower) - self.means) ** 2)


class BesselSmearing(nn.Module):
    """
    Implementation adapted from PaiNN

    Sine for radial basis expansion with coulomb decay. (0th order Bessel from DimeNet)
    """

    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0, num_rbf=50, trainable=True):
        """
        Args:
            cutoff_upper: radial cutoff
            num_rbf: number of basis functions.
        """
        super(BesselSmearing, self).__init__()
        assert cutoff_lower == 0, 'Lower cutoff not implemented for Bessel RBF'

        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.num_rbf = num_rbf
        self.trainable = trainable

        # compute offset and width of Gaussian functions
        freqs = self._initial_params()
        if trainable:
            self.register_parameter('freqs', nn.Parameter(freqs))
        else:
            self.register_buffer('freqs', freqs)

    def _initial_params(self):
        return torch.arange(1, self.num_rbf + 1) * math.pi / self.cutoff_upper

    def reset_parameters(self):
        freqs = self._initial_params()
        self.freqs.data.copy_(freqs)

    def forward(self, inputs):
        ax = inputs.unsqueeze(-1) * self.freqs
        sinax = torch.sin(ax)

        norm = torch.where(inputs == 0, torch.tensor(1.0, device=inputs.device), inputs)
        y = sinax / norm.unsqueeze(1)

        return y


class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class CosineCutoff(nn.Module):
    def __init__(self, cutoff_lower=0.0, cutoff_upper=5.0):
        super(CosineCutoff, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper

    def forward(self, distances):
        if self.cutoff_lower > 0:
            cutoffs = 0.5 * (torch.cos(math.pi * (2 * (distances - self.cutoff_lower) / (self.cutoff_upper - self.cutoff_lower) + 1.0)) + 1.0)
            # remove contributions below the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            cutoffs = cutoffs * (distances > self.cutoff_lower).float()
            return cutoffs
        else:
            cutoffs = 0.5 * (torch.cos(distances * math.pi / self.cutoff_upper) + 1.0)
            # remove contributions beyond the cutoff radius
            cutoffs = cutoffs * (distances < self.cutoff_upper).float()
            return cutoffs


class Distance(nn.Module):
    def __init__(self, cutoff_lower, cutoff_upper, return_vec=False, max_num_neighbors=32):
        super(Distance, self).__init__()
        self.cutoff_lower = cutoff_lower
        self.cutoff_upper = cutoff_upper
        self.return_vec = return_vec
        self.max_num_neighbors = max_num_neighbors

    def forward(self, pos, batch):
        edge_index = radius_graph(pos, r=self.cutoff_upper, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors))
        edge_vec = (pos[edge_index[0]] - pos[edge_index[1]])
        edge_weight = edge_vec.norm(dim=-1)

        lower_mask = edge_weight >= self.cutoff_lower
        edge_index = edge_index[:,lower_mask]
        edge_weight = edge_weight[lower_mask]

        if self.return_vec:
            edge_vec = edge_vec[lower_mask]
            return edge_index, edge_weight, edge_vec
        return edge_index, edge_weight


rbf_class_mapping = {
    'gauss': GaussianSmearing,
    'expnorm': ExpNormalSmearing,
    'bessel': BesselSmearing
}

act_class_mapping = {
    'ssp': ShiftedSoftplus,
    'silu': nn.SiLU,
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid
}
