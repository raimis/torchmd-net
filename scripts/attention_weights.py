from os.path import dirname, join, exists
import pickle
import argparse
from tqdm import tqdm
import torch
from torchmdnet.datasets import QM9
from torchmdnet.models import load_model
from torchmdnet.utils import make_splits
from torchmdnet.data import Subset
from torchmdnet import attention_weights
from torch_geometric.data import DataLoader
from torch_scatter import scatter
from matplotlib import pyplot as plt


z2idx = {
    1: 0,
    6: 1,
    7: 2,
    8: 3,
    9: 4
}

num2elem = {
    1: 'H',
    6: 'C',
    7: 'N',
    8: 'O',
    9: 'F',
}

n_elements = len(z2idx)

def extract_data(model_path, dataset_path, dataset_arg, batch_size=64, plot_molecules=False):
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

        if plot_molecules:
            for mol_idx in batch.batch.unique():
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                # edges
                for idx1, idx2 in batch.edge_index.T[batch.batch[batch.edge_index[0]] == mol_idx]:
                    # attention weights
                    max_attn = attention_weights.rollout_weights[-1][batch.batch[attention_weights.rollout_index[-1][0]] == mol_idx].sum(dim=-1).max()
                    attn_idx = torch.where((attention_weights.rollout_index[-1][0] == idx1) & (attention_weights.rollout_index[-1][1] == idx2))[0]
                    attn_weight = max(0, min(1, attention_weights.rollout_weights[-1][attn_idx].sum() / max_attn))
                    ax.quiver(*batch.pos[idx1], *(batch.pos[idx2] - batch.pos[idx1]), alpha=float(attn_weight), colors='0', lw=2)
                    # bonds
                    ax.plot(*torch.stack([batch.pos[idx1], batch.pos[idx2]], dim=1), alpha=0.1, c='0', linestyle='dashed')
                # nodes
                for atom_type in z2idx.keys():
                    if ((batch.batch == mol_idx) & (batch.z == atom_type)).sum() == 0:
                        continue
                    colors = [f'C{z2idx[int(z)]}' for z in batch.z[(batch.batch == mol_idx) & (batch.z == atom_type)]]
                    ax.scatter(*batch.pos[(batch.batch == mol_idx) & (batch.z == atom_type)].T, c=colors, label=num2elem[atom_type], s=100)
                plt.legend()
                plt.axis('off')
                plt.show()

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


def visualize(weights_directory):
    # load data
    with open(join(weights_directory, 'attn_weights.pkl'), 'rb') as f:
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze Attention Weights')
    parser.add_argument('--extract-data', type=bool, help='Whether to extract the attention weights or use previously stored data')
    parser.add_argument('--model-path', type=str, help='Path to a model checkpoint with corresponding splits.npz in the same directory')
    parser.add_argument('--dataset-path', type=str, help='Path to the directory containing QM9 dataset files')
    parser.add_argument('--dataset-arg', type=str, help='Additional argument to the dataset class (e.g. target property for QM9)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for the attention weight extraction')
    parser.add_argument('--plot-molecules', type=bool, help='If True, draws all processed molecules with associated attention weights during extraction')

    args = parser.parse_args()

    if args.extract_data:
        extract_data(args.model_path, args.dataset_path, args.dataset_arg, args.batch_size, args.plot_molecules)
    visualize(dirname(args.model_path))
