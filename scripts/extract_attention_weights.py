from os.path import dirname, join, exists
import pickle
from tqdm import tqdm
import torch
from torchmdnet.datasets import QM9
from torchmdnet.models import load_model
from torchmdnet.utils import make_splits
from torchmdnet.data import Subset
from torchmdnet import attention_weights
from torch_geometric.data import DataLoader
from torch_scatter import scatter


n_elements = 5
model_path = 'models/et/epoch=1129-val_loss=0.0002-test_loss=0.0064.ckpt'
dataset_path = '/home/philipp/Documents/data/qm9'
dataset_arg = 'energy_U0'
batch_size = 64

z2idx = {
    1: 0,
    6: 1,
    7: 2,
    8: 3,
    9: 4
}

torch.set_grad_enabled(False)

# load data
splits_path = join(dirname(model_path), 'splits.npz')
assert exists(splits_path), f'Missing splits.npz in {dirname(model_path)}.'
_, _, test_split = make_splits(None, None, None, None, None, splits=splits_path)
data = DataLoader(Subset(QM9(dataset_path, dataset_arg=dataset_arg), test_split), batch_size=batch_size)
# load model
model = load_model(model_path)
# initialize attention weight collector
attention_weights.create(model.representation_model.num_layers)

zs_0, zs_1 = [], []
zs_0_ref, zs_1_ref = [], []
# extract attention weights from model
for batch in tqdm(data):
    model(batch.z, batch.pos, batch.batch)
    # TODO: is this the correct ordering?
    zs_0.append(batch.z[attention_weights.rollout_index[-1][0]])
    zs_1.append(batch.z[attention_weights.rollout_index[-1][1]])
    zs_0_ref.append(batch.z[batch.edge_index[0]])
    zs_1_ref.append(batch.z[batch.edge_index[1]])

# compute attention weight scatter indices
zs = torch.stack([torch.cat(zs_0), torch.cat(zs_1)])
zs, index = torch.unique(zs, dim=1, return_inverse=True)
zs = zs.reshape(2, n_elements, n_elements)

# reduce attention weights to elemental interactions
attn = torch.cat(attention_weights.rollout_weights, dim=0)
attn = scatter(attn, index=index, dim=0, reduce='mean')
attn = attn.reshape(n_elements, n_elements, -1)

# compute bonding probabilities from the data
zs_ref = torch.stack([torch.cat(zs_0_ref), torch.cat(zs_1_ref)])
zs_ref, counts_ref = torch.unique(zs_ref, dim=1, return_counts=True)
counts_ref = counts_ref.float()
for elem in zs_ref.unique():
    counts_ref[zs_ref[0] == elem] /= counts_ref[zs_ref[0] == elem].sum()
index_ref = zs_ref.clone().apply_(lambda z: z2idx[z])
counts_ref_square = torch.zeros(n_elements, n_elements)
counts_ref_square[index_ref[0],index_ref[1]] = counts_ref

# save data
with open(join(dirname(model_path), 'attn_weights.pkl'), 'wb') as f:
    pickle.dump((zs[1,0], attn, zs_ref[0].unique(), counts_ref_square), f)
