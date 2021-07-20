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
batch_size = 128

torch.set_grad_enabled(False)

splits_path = join(dirname(model_path), 'splits.npz')
assert exists(splits_path), f'Missing splits.npz in {dirname(model_path)}.'
_, _, test_split = make_splits(None, None, None, None, None, splits=splits_path)
data = DataLoader(Subset(QM9(dataset_path, dataset_arg=dataset_arg), test_split), batch_size=batch_size)
model = load_model(model_path)

attention_weights.create(model.representation_model.num_layers)

zs_0 = []
zs_1 = []
for batch in tqdm(data):
    model(batch.z, batch.pos, batch.batch)
    # TODO: is this the correct ordering?
    zs_0.append(batch.z[attention_weights.edge_index[-1][0]])
    zs_1.append(batch.z[attention_weights.edge_index[-1][1]])

zs = torch.stack([torch.cat(zs_0), torch.cat(zs_1)])
zs, index = torch.unique(zs, dim=1, return_inverse=True)
zs = zs.reshape(2, n_elements, n_elements)

attn = []
for layer in attention_weights.weights:
    current = torch.cat(layer, dim=0)
    reduced = scatter(current, index=index, dim=0, reduce='mean')
    attn.append(reduced.reshape(n_elements, n_elements, -1))

with open(join(dirname(model_path), 'attn_weights.pkl'), 'wb') as f:
    pickle.dump((zs[1,0], attn), f)
