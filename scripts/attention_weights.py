import os
from os.path import dirname, join, exists
import pickle
import time
import glob
import argparse
from tqdm import tqdm
import torch
from torchmdnet import datasets, attention_weights
from torchmdnet.models.model import load_model
from torchmdnet.utils import make_splits
from torchmdnet.data import Subset
from torch_geometric.data import DataLoader
from torch_scatter import scatter
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from moleculekit.molecule import Molecule
from moleculekit.vmdgraphics import VMDCylinder
from moleculekit.vmdviewer import getCurrentViewer


num2elem = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F'}
z2idx = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}
n_elements = len(num2elem)
dset_arg2name = {
    'energy_U0': '$U_0$'
}

torch.manual_seed(1234)


def extract_data(model_path, dataset_path, dataset_name, dataset_arg, batch_size=64, plot_molecules=False, device='cpu'):
    torch.set_grad_enabled(False)

    # load data
    print('loading data')
    splits_path = join(dirname(model_path), 'splits.npz')
    assert exists(splits_path), f'Missing splits.npz in {dirname(model_path)}.'
    _, _, test_split = make_splits(None, None, None, None, None, splits=splits_path)
    data = DataLoader(Subset(getattr(datasets, dataset_name)(dataset_path, dataset_arg=dataset_arg), test_split),
                      batch_size=batch_size, shuffle=True, num_workers=2)
    # load model
    print('loading model')
    model = load_model(model_path).to(device)
    # initialize attention weight collector
    attention_weights.create(model.representation_model.num_layers)

    zs_0, zs_1 = [], []
    zs_0_ref, zs_1_ref = [], []
    atoms_per_elem = {z: 0 for z in z2idx.keys()}
    distances = []
    # extract attention weights from model
    for batch in tqdm(data, desc='extracting attention weights'):
        model(batch.z.to(device), batch.pos.to(device), batch.batch.to(device))

        if batch.edge_index is None:
            # guess bonds
            idx_offset = 0
            edge_index = []
            for mol_idx in batch.batch.unique():
                mask = batch.batch == mol_idx
                mol = Molecule().empty(mask.sum())
                mol.coords = batch.pos[mask].unsqueeze(2).numpy()
                mol.element[:] = [num2elem[num] for num in batch.z[mask].numpy()]
                mol.name[:] = [num2elem[num] for num in batch.z[mask].numpy()]

                idxs = torch.from_numpy(mol._guessBonds().T.astype(np.int64)) + idx_offset
                edge_index.append(torch.cat([idxs, idxs.flip(dims=(0,))], dim=1))

                idx_offset += mask.sum()
            batch.edge_index = torch.cat(edge_index, dim=1)

        if plot_molecules != 'off':
            idx_offset = 0
            for mol_idx in batch.batch.unique():
                rollout_batch = batch.batch[attention_weights.rollout_index[-1][0]]
                if plot_molecules == 'VMD':
                    vmd = getCurrentViewer()

                    # visualize using VMD
                    mask = batch.batch == mol_idx
                    mol = Molecule().empty(mask.sum())
                    mol.coords = batch.pos[mask].unsqueeze(2).numpy()
                    mol.element[:] = [num2elem[num] for num in batch.z[mask].numpy()]
                    mol.name[:] = [num2elem[num] for num in batch.z[mask].numpy()]

                    max_attn = attention_weights.rollout_weights[-1][rollout_batch == mol_idx].abs().max()
                    for idx1, idx2 in attention_weights.rollout_index[-1].T[rollout_batch == mol_idx]:
                        attn_idx = torch.where((attention_weights.rollout_index[-1][0] == idx1) & (attention_weights.rollout_index[-1][1] == idx2))[0]
                        strength = float(attention_weights.rollout_weights[-1][attn_idx] / max_attn)

                        draw_thresholds = (0.1, -0.1)
                        if strength > draw_thresholds[0] or strength < draw_thresholds[1]:
                            vmd.send(f'mol new')
                            vmd.send(f'material add mat-{idx1}-{idx2}')
                            vmd.send(f'material change opacity mat-{idx1}-{idx2} {abs(strength) ** 2}')
                            vmd.send(f'material change specular mat-{idx1}-{idx2} 0')
                            vmd.send(f'draw material mat-{idx1}-{idx2}')

                            if strength > draw_thresholds[0]:
                                VMDCylinder(mol.coords[idx1 - idx_offset].flatten(), mol.coords[idx2 - idx_offset].flatten(), color='red', radius=0.05 * abs(strength))
                            elif strength < draw_thresholds[1]:
                                VMDCylinder(mol.coords[idx1 - idx_offset].flatten(), mol.coords[idx2 - idx_offset].flatten(), color='blue', radius=0.05 * abs(strength))

                    mol.view(style='Licorice', viewerhandle=vmd)
                    vmd.send('mol modstyle 0 top Licorice 0.1 12 12')
                    vmd.send('mol modmaterial 0 top Transparent')
                    vmd.send(f'material change opacity Transparent 0.25')
                    vmd.send(f'material change opacity Diffuse 1')
                    vmd.send(f'material change opacity Specular 0')
                    vmd.send('display rendermode GLSL')

                    while not vmd.completed():
                        time.sleep(0.1)
                elif plot_molecules == 'matplotlib':
                    # visualize using matplotlib
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    # edges
                    max_attn = attention_weights.rollout_weights[-1][rollout_batch == mol_idx].max()
                    for idx1, idx2 in attention_weights.rollout_index[-1].T[rollout_batch == mol_idx]:
                        # attention weights
                        attn_idx = torch.where((attention_weights.rollout_index[-1][0] == idx1) & (attention_weights.rollout_index[-1][1] == idx2))[0]
                        attn_weight = max(0, min(1, attention_weights.rollout_weights[-1][attn_idx] / max_attn))
                        ax.quiver(*batch.pos[idx1], *(batch.pos[idx2] - batch.pos[idx1]), alpha=float(attn_weight), colors='red', lw=1, arrow_length_ratio=0.1)
                        if batch.edge_index is not None and ((batch.edge_index[0] == idx1) & (batch.edge_index[1] == idx2)).any() and idx1 != idx2:
                            # bonds
                            ax.plot(*torch.stack([batch.pos[idx1], batch.pos[idx2]], dim=1), alpha=1, c='0', linestyle='dotted')

                    # nodes
                    for atom_type in num2elem.keys():
                        if ((batch.batch == mol_idx) & (batch.z == atom_type)).sum() == 0:
                            continue
                        colors = [f'C{int(z)}' for z in batch.z[(batch.batch == mol_idx) & (batch.z == atom_type)]]
                        ax.scatter(*batch.pos[(batch.batch == mol_idx) & (batch.z == atom_type)].T, c=colors, label=num2elem[atom_type], s=100)
                    plt.legend()
                    plt.axis('off')
                    plt.show()
                idx_offset += (batch.batch == mol_idx).sum()

        zs_0.append(batch.z[attention_weights.rollout_index[-1][0]])
        zs_1.append(batch.z[attention_weights.rollout_index[-1][1]])

        zs_0_ref.append(batch.z[batch.edge_index[0]])
        zs_1_ref.append(batch.z[batch.edge_index[1]])

        for elem in batch.z.unique().numpy():
            atoms_per_elem[elem] += (batch.z == elem).sum().numpy()

        distances.append(((batch.pos[attention_weights.rollout_index[-1][0]] - batch.pos[attention_weights.rollout_index[-1][1]]) ** 2).sum(dim=-1).sqrt())
        
    print('processing data')

    # compute attention weight scatter indices
    zs_full = torch.stack([torch.cat(zs_0), torch.cat(zs_1)])
    zs, index = torch.unique(zs_full, dim=1, return_inverse=True)
    z_idxs = zs.clone().apply_(lambda z: z2idx[z])
    tmp = torch.zeros(2, n_elements, n_elements).long()
    tmp[:,z_idxs[0],z_idxs[1]] = zs
    zs = tmp[1,0]

    # reduce attention weights to elemental interactions
    attn_full = torch.cat(attention_weights.rollout_weights, dim=0)
    attn = scatter(attn_full, index=index, dim=0, reduce='mean')
    tmp = torch.full((n_elements, n_elements), float('nan'))
    tmp[z_idxs[0],z_idxs[1]] = attn
    attn = tmp

    # compute bond probabilities from the data
    zs_ref = torch.stack([torch.cat(zs_0_ref), torch.cat(zs_1_ref)])
    zs_ref, counts_ref = torch.unique(zs_ref, dim=1, return_counts=True)
    counts_ref = counts_ref.float()
    for elem in zs_ref.unique():
        counts_ref[zs_ref[0] == elem] /= counts_ref[zs_ref[0] == elem].sum()
    index_ref = zs_ref.clone().apply_(lambda z: z2idx[z])
    counts_ref_square = torch.full((n_elements, n_elements), float('nan'))
    counts_ref_square[index_ref[0],index_ref[1]] = counts_ref
    zs_ref = zs_ref[0].unique()

    dist = torch.cat(distances)

    # save data
    print('saving data')
    with open(join(dirname(model_path), 'attn_weights.pkl'), 'wb') as f:
        pickle.dump((zs, attn, zs_ref, counts_ref_square, atoms_per_elem, zs_full, attn_full, dist), f)
    print('done')


def visualize(basedir, normalize_attention):
    plt.rcParams['mathtext.fontset'] = 'cm'

    paths = sorted(glob.glob(join(basedir, '**', 'attn_weights.pkl'), recursive=True))[::-1]

    # plot attention weights
    print(f'creating attention plot with {len(paths)} datasets')
    fig, axes_all = plt.subplots(nrows=len(paths), ncols=4, sharex=False, sharey=True, figsize=(8, 4.8),
                                 gridspec_kw=dict(width_ratios=[0.5, 1, 1, 1], hspace=0), squeeze=False)

    for dataset_idx, (path, axes) in enumerate(zip(paths, axes_all[:,1:])):
        axes_all[dataset_idx,0].axis('off')
        names = [dset_arg2name[name] if name in dset_arg2name else name[0].upper() + name[1:]
                 for name in path.split(os.sep)[-2].split('-')[:-1]]
        axes_all[dataset_idx,0].text(0.6, 0.5, '\n'.join(names), ha='right', va='center',
                                     transform=axes_all[dataset_idx,0].transAxes, fontsize=15)

        # load data
        with open(path, 'rb') as f:
            zs, weights, zs_ref, probs_ref, atoms_per_elem, _, _, _ = pickle.load(f)
        elements = num2elem.values()

        # subplot 0
        axes[0].imshow(probs_ref, cmap='Reds', vmin=0, vmax=1)
        axes[0].set(
            xticks=range(len(elements)),
            yticks=range(len(elements)),
            xticklabels=elements,
            yticklabels=elements,
        )
        axes[0].set_ylabel('$z_i$', fontsize=15)
        if dataset_idx == 0:
            axes[0].set_title('Bond Probabilities', fontsize=12)
            axes[0].tick_params(labelleft=True, labelbottom=False)
        else:
            axes[0].tick_params(labelleft=True, top=True)
        if dataset_idx == len(paths) - 1:
            axes[0].set_xlabel('$z_j$', fontsize=15)

        # subplot 1
        if normalize_attention:
            mask = ~weights.isnan().all(dim=1)
            weights[mask] = weights[mask] / weights[mask].nansum(dim=1, keepdim=True)
        axes[1].imshow(weights, cmap='Blues')
        axes[1].set(
            xticks=range(len(elements)),
            yticks=range(len(elements)),
            xticklabels=elements,
            yticklabels=elements,
        )
        if dataset_idx == 0:
            axes[1].set_title('Attention Scores', fontsize=12)
            axes[1].tick_params(labelbottom=False)
        else:
            axes[1].tick_params(top=True)
        if dataset_idx == len(paths) - 1:
            axes[1].set_xlabel('$z_j$', fontsize=15)

        # subplot 2
        axes[2].barh(range(len(atoms_per_elem.keys())), atoms_per_elem.values(), color='forestgreen')
        for i, v in enumerate(atoms_per_elem.values()):
            is_max = v >= max(atoms_per_elem.values()) * 0.85
            offset = max(atoms_per_elem.values()) * 0.025
            axes[2].text(v - offset if is_max else v + offset, i, str(v), va='center', ha='right' if is_max else 'left', color='1' if is_max else '0')
        axes[2].set_box_aspect(1)
        axes[2].set_xticks([])
        if dataset_idx == 0:
            axes[2].set_title('Total', fontsize=12)
        axes[2].tick_params(labelright=True)

        for ax in axes:
            ax.tick_params(color='0.5', right=True)
            for spine in ax.spines.values():
                spine.set_edgecolor('0.5')

    plt.savefig(join(basedir, 'attn_weights.pdf'), bbox_inches='tight')


    for path_idx, path in enumerate(paths):
        print(f'creating dist-attention plot ({path_idx + 1}/{len(paths)})')
        # load data
        with open(path, 'rb') as f:
            _, _, _, _, _, zs_full, attn_full, dist = pickle.load(f)

        # visualize attention by distance
        z1, z2 = 1, 6
        ma_width = 1000
        mask = ((zs_full[0] == z1) & (zs_full[1] == z2)) | ((zs_full[0] == z2) & (zs_full[1] == z1))
        fig, ax = plt.subplots()
        ax.grid(True)
        ax.hist(dist[mask].numpy(), bins=70, color='C1', alpha=0.3)
        ax.set_ylabel('Number of interactions', color='C1')
        ax = ax.twinx()
        averaged = pd.Series(attn_full[mask][dist[mask].argsort()]).rolling(ma_width).mean().shift(-ma_width).values
        ax.scatter(dist[mask].sort().values, averaged, marker='.', color='C0')
        ax.set_xlabel('Distance ($\AA$)')
        ax.set_ylabel('Attention score', color='C0')
        ax.set_title('Attention scores by distance for Hydrogen-Carbon interactions')
        ax.set_xlim(0)
        plt.savefig(join(dirname(path), 'attn-dist.pdf'), bbox_inches='tight')
    print('done')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze Attention Weights')
    parser.add_argument('--extract-data', type=bool, help='Whether to extract the attention weights or use previously stored data')
    parser.add_argument('--model-path', type=str, help='Path to a model checkpoint with corresponding splits.npz in the same directory')
    parser.add_argument('--dataset-path', type=str, help='Path to the directory containing the dataset')
    parser.add_argument('--dataset-name', type=str, choices=datasets.__all__, help='Name of the dataset')
    parser.add_argument('--dataset-arg', type=str, help='Additional argument to the dataset class (e.g. target property for QM9)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for the attention weight extraction')
    parser.add_argument('--plot-molecules', type=str, default='off', choices=['off', 'VMD', 'matplotlib'], help='If True, draws all processed molecules with associated attention weights during extraction')
    parser.add_argument('--normalize-attention', type=bool, help='Whether to normalize the attention scores such that each row adds up to one')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the extraction on')

    args = parser.parse_args()

    if args.extract_data:
        extract_data(args.model_path, args.dataset_path, args.dataset_name,
                     args.dataset_arg, args.batch_size, args.plot_molecules,
                     args.device)
    visualize(dirname(dirname(args.model_path)), args.normalize_attention)
