from os.path import join
import pickle
from matplotlib import pyplot as plt
import torch


model_folder = 'models/et'

num2elem = {
    1: 'H',
    6: 'C',
    7: 'N',
    8: 'O',
    9: 'F',
}

# load data
with open(join(model_folder, 'attn_weights.pkl'), 'rb') as f:
    zs, weights = pickle.load(f)
elements = [num2elem[int(num)] for num in zs]
weights = [w.permute(2, 0, 1) for w in weights]

# attention rollout (https://arxiv.org/pdf/2005.00928.pdf)
# TODO: this should be done with raw attention matrices
cum = weights[0]
for w in weights[1:]:
    cum = torch.matmul(w, cum)
weights = cum.permute(1, 2, 0)

# plot attention weights
fig, axes = plt.subplots(ncols=weights.size(-1), sharex=True, sharey=True)
for i, ax in enumerate(axes):
    ax.matshow(weights[...,i], cmap='Reds')
    ax.set(
        title=f'Head {i + 1}',
        xticks=range(len(elements)),
        yticks=range(len(elements)),
        xticklabels=elements,
        yticklabels=elements
    )
plt.show()
