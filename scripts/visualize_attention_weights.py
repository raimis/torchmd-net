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
    zs, weights, zs_ref, probs_ref = pickle.load(f)
elements = [num2elem[int(num)] for num in zs]
elements_ref = [num2elem[int(num)] for num in zs_ref]

# plot attention weights
fig, axes = plt.subplots(ncols=weights.size(-1) + 1, sharex=True, sharey=True)

axes[0].matshow(probs_ref, cmap='Blues', vmin=0, vmax=1)
axes[0].set(
    title='Bond probabilities',
    xticks=range(len(elements_ref)),
    yticks=range(len(elements_ref)),
    xticklabels=elements_ref,
    yticklabels=elements_ref
)

for i, ax in enumerate(axes[1:]):
    ax.matshow(weights[...,i], cmap='Reds')
    ax.set(
        title=f'Head {i + 1}',
        xticks=range(len(elements)),
        yticks=range(len(elements)),
        xticklabels=elements,
        yticklabels=elements
    )
plt.show()
