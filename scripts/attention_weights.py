import os
from os.path import basename, dirname, join, exists
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
from PIL import Image


render_rate = 0.01
num2elem = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}
z2idx = {1: 0, 6: 1, 7: 2, 8: 3, 9: 4}
n_elements = len(num2elem)
dset_arg2name = {
    "energy_U0": "$U_0$",
    "salicylic_acid": "Salicylic Acid",
    "malonaldehyde": "Malondialdehyde",
}
kcalmol2ev = 0.043363

torch.manual_seed(1234)


def extract_data(
    model_path,
    dataset_path,
    dataset_name,
    dataset_arg,
    batch_size=64,
    plot_molecules=False,
    device="cpu",
):
    torch.set_grad_enabled(False)

    # load data
    print("loading data")
    splits_path = join(dirname(model_path), "splits.npz")
    data = getattr(datasets, dataset_name)(dataset_path, dataset_arg=dataset_arg)
    has_dy = hasattr(data[0], "dy")
    if exists(splits_path):
        _, _, test_split = make_splits(None, None, None, None, None, splits=splits_path)
    else:
        print("Warning: couldn't find splits.npz, using whole dataset")
        _, _, test_split = make_splits(len(data), 0, 0, None, 10)
    data = DataLoader(
        Subset(data, test_split), batch_size=batch_size, shuffle=True, num_workers=6
    )

    # load model
    print("loading model")
    model = load_model(model_path, device=device, derivative=has_dy).eval()
    # initialize attention weight collector
    attention_weights.reset(model.representation_model.num_layers)

    losses_y = []
    losses_dy = []
    zs_0, zs_1 = [], []
    zs_0_ref, zs_1_ref = [], []
    atoms_per_elem = {z: 0 for z in z2idx.keys()}
    distances = []
    mol_save_idx = 0
    # extract attention weights from model
    progress = tqdm(data, desc="extracting attention weights")
    for batch in progress:
        batch = batch.to(device)

        if model.derivative:
            torch.set_grad_enabled(True)
        pred, deriv = model(
            batch.z.to(device), batch.pos.to(device), batch.batch.to(device)
        )
        torch.set_grad_enabled(False)

        model.zero_grad()
        batch.pos = batch.pos.detach()
        pred = pred.detach()
        if deriv is not None:
            deriv = deriv.detach()

        losses_y.append(torch.nn.functional.l1_loss(pred, batch.y).cpu().numpy())
        if deriv is not None and hasattr(batch, "dy"):
            losses_dy.append(torch.nn.functional.l1_loss(deriv, batch.dy).cpu().numpy())
            progress.set_postfix(
                dict(loss_y=np.mean(losses_y), loss_dy=np.mean(losses_dy))
            )
        else:
            progress.set_postfix(dict(loss=np.mean(losses_y)))

        batch = batch.cpu()

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

                idxs = (
                    torch.from_numpy(mol._guessBonds().T.astype(np.int64)) + idx_offset
                )
                edge_index.append(torch.cat([idxs, idxs.flip(dims=(0,))], dim=1))

                idx_offset += mask.sum()
            batch.edge_index = torch.cat(edge_index, dim=1)

        if plot_molecules:
            idx_offset = 0
            for mol_idx in batch.batch.unique():
                if torch.rand(1) < render_rate:
                    rollout_batch = batch.batch[attention_weights.rollout_index[-1][0]]
                    vmd = getCurrentViewer()

                    # visualize using VMD
                    mask = batch.batch == mol_idx
                    mol = Molecule().empty(mask.sum())
                    mol.coords = batch.pos[mask].unsqueeze(2).numpy()
                    mol.element[:] = [num2elem[num] for num in batch.z[mask].numpy()]
                    mol.name[:] = [num2elem[num] for num in batch.z[mask].numpy()]

                    max_attn = (
                        attention_weights.rollout_weights[-1][rollout_batch == mol_idx]
                        .abs()
                        .max()
                    )
                    for idx1, idx2 in attention_weights.rollout_index[-1].T[
                        rollout_batch == mol_idx
                    ]:
                        attn_idx = torch.where(
                            (attention_weights.rollout_index[-1][0] == idx1)
                            & (attention_weights.rollout_index[-1][1] == idx2)
                        )[0]
                        strength = float(
                            attention_weights.rollout_weights[-1][attn_idx] / max_attn
                        )

                        draw_thresholds = (0.2, -0.2)
                        if (
                            strength > draw_thresholds[0]
                            or strength < draw_thresholds[1]
                        ):
                            vmd.send("mol new")
                            vmd.send(f"material add mat-{idx1}-{idx2}")
                            vmd.send(
                                f"material change opacity mat-{idx1}-{idx2} {abs(strength) ** 2}"
                            )
                            vmd.send(f"material change specular mat-{idx1}-{idx2} 0")
                            vmd.send(f"draw material mat-{idx1}-{idx2}")

                            if strength > draw_thresholds[0]:
                                VMDCylinder(
                                    mol.coords[idx1 - idx_offset].flatten(),
                                    mol.coords[idx2 - idx_offset].flatten(),
                                    color="red",
                                    radius=0.05 * abs(strength),
                                )
                            elif strength < draw_thresholds[1]:
                                VMDCylinder(
                                    mol.coords[idx1 - idx_offset].flatten(),
                                    mol.coords[idx2 - idx_offset].flatten(),
                                    color="blue",
                                    radius=0.05 * abs(strength),
                                )

                    mol.view(style="Licorice", viewerhandle=vmd)
                    vmd.send("mol modstyle 0 top Licorice 0.1 12 12")
                    vmd.send("mol modmaterial 0 top Transparent")
                    vmd.send("material change opacity Transparent 0.4")
                    vmd.send("material change opacity Diffuse 1")
                    vmd.send("material change opacity Specular 0")
                    vmd.send("mol addrep")

                    vmd.send("display rendermode GLSL")
                    vmd.send("color Display Background white")
                    vmd.send("display ambientocclusion on")
                    vmd.send("display aoambient 1")
                    vmd.send("display depthcue off")

                    vmd.send("mol representation VDW 0.125 12")
                    vmd.send("mol selection all")
                    vmd.send("mol material AOChalky")
                    vmd.send("mol addrep top")

                    img_name = f"{dataset_name}-{dataset_arg}-{mol_save_idx}.png"
                    vmd.send(f"render TachyonLOSPRayInternal {img_name} save %s")
                    mol_save_idx += 1
                    vmd.close()

                    img = np.array(Image.open(img_name))
                    mask = (img != 254).any(axis=(1, 2)) | (img != 254).any(axis=(0, 2))
                    Image.fromarray(img[mask][:, mask]).save(img_name)

                    while not vmd.completed():
                        time.sleep(0.1)
                idx_offset += (batch.batch == mol_idx).sum()

        zs_0.append(batch.z[attention_weights.rollout_index[-1][0]].int())
        zs_1.append(batch.z[attention_weights.rollout_index[-1][1]].int())

        zs_0_ref.append(batch.z[batch.edge_index[0]].int())
        zs_1_ref.append(batch.z[batch.edge_index[1]].int())

        for elem in batch.z.unique().numpy():
            atoms_per_elem[elem] += (batch.z == elem).sum().numpy()

        distances.append(
            (
                (
                    batch.pos[attention_weights.rollout_index[-1][0]]
                    - batch.pos[attention_weights.rollout_index[-1][1]]
                )
                ** 2
            )
            .sum(dim=-1)
            .sqrt()
        )

    print("final MAE:", end="")
    if len(losses_dy) > 0:
        print(
            f"\n\tloss y:  {np.mean(losses_y):.3f}\n\tloss dy: {np.mean(losses_dy):.3f}"
        )
    else:
        print(f" {np.mean(losses_y):.3f}")

    print("processing data")

    # compute attention weight scatter indices
    zs_full = torch.stack([torch.cat(zs_0), torch.cat(zs_1)])
    zs, index = torch.unique(zs_full, dim=1, return_inverse=True)
    zs = zs.long()
    z_idxs = zs.clone().apply_(lambda z: z2idx[z])
    tmp = torch.zeros(2, n_elements, n_elements).long()
    tmp[:, z_idxs[0], z_idxs[1]] = zs
    zs = tmp[1, 0]

    # reduce attention weights to elemental interactions
    attn_full = torch.cat(attention_weights.rollout_weights, dim=0)
    attn = scatter(attn_full, index=index, dim=0, reduce="mean")
    tmp = torch.full((n_elements, n_elements), float("nan"))
    tmp[z_idxs[0], z_idxs[1]] = attn
    attn = tmp

    # compute bond probabilities from the data
    zs_ref = torch.stack([torch.cat(zs_0_ref), torch.cat(zs_1_ref)])
    zs_ref, counts_ref = torch.unique(zs_ref, dim=1, return_counts=True)
    zs_ref = zs_ref.long()
    counts_ref = counts_ref.float()
    for elem in zs_ref.unique():
        counts_ref[zs_ref[0] == elem] /= counts_ref[zs_ref[0] == elem].sum()
    index_ref = zs_ref.clone().apply_(lambda z: z2idx[z])
    counts_ref_square = torch.full((n_elements, n_elements), float("nan"))
    counts_ref_square[index_ref[0], index_ref[1]] = counts_ref
    zs_ref = zs_ref[0].unique()

    dist = torch.cat(distances)

    # save data
    print("saving data")
    with open(join(dirname(model_path), "attn_weights.pkl"), "wb") as f:
        pickle.dump(
            (
                zs,
                attn,
                zs_ref,
                counts_ref_square,
                atoms_per_elem,
                zs_full,
                attn_full,
                dist,
            ),
            f,
        )
    print("done")


def visualize(
    basedir, normalize_attention, distance_plots, combine_dataset, ignore_datasets=[]
):
    plt.rcParams["mathtext.fontset"] = "cm"

    paths = sorted(glob.glob(join(basedir, "**", "attn_weights.pkl"), recursive=True))
    ignore_datasets = [name.lower() for name in ignore_datasets]
    paths = [
        path
        for path in paths
        if basename(dirname(path)).split("-")[0].lower() not in ignore_datasets
    ]

    dataset_paths = dict()
    if combine_dataset:
        print("combining datasets")
        for path in paths:
            dset_name = basename(dirname(path)).split("-")[0]
            if dset_name in dataset_paths:
                dataset_paths[dset_name].append(path)
            else:
                dataset_paths[dset_name] = [path]
        paths = dataset_paths

    # plot attention weights
    print(f"creating attention plot with {len(paths)} datasets")
    fig, axes_all = plt.subplots(
        nrows=len(paths),
        ncols=4,
        sharex=False,
        sharey=True,
        figsize=(8, 2.4 * len(paths)),
        gridspec_kw=dict(width_ratios=[0.5, 1, 1, 1], hspace=0),
        squeeze=False,
    )

    for dataset_idx, (path, axes) in enumerate(zip(paths, axes_all[:, 1:])):
        if combine_dataset:
            dset_name = path
        else:
            dset_name = [
                dset_arg2name[name]
                if name in dset_arg2name
                else name[0].upper() + name[1:]
                for name in path.split(os.sep)[-2].split("-")[:-1]
            ]
            dset_name = "\n".join(dset_name)

        axes_all[dataset_idx, 0].axis("off")
        axes_all[dataset_idx, 0].text(
            0.6,
            0.5,
            dset_name,
            ha="right",
            va="center",
            transform=axes_all[dataset_idx, 0].transAxes,
            fontsize=12,
        )

        elements = num2elem.values()

        # load data
        if combine_dataset:
            weights_counts = torch.zeros(len(num2elem), len(num2elem), dtype=torch.int)
            weights = torch.zeros(len(num2elem), len(num2elem))
            probs_ref_counts = weights_counts.clone()
            probs_ref = weights.clone()
            atoms_per_elem = dict()
            for p in tqdm(paths[path], desc=f"combining data for {path}"):
                with open(p, "rb") as f:
                    zs, _weights, _, _probs_ref, _atoms_per_elem, _, _, _ = pickle.load(
                        f
                    )
                weights_counts += (~_weights.isnan()).int()
                weights += _weights.nan_to_num()
                probs_ref_counts += (~_probs_ref.isnan()).int()
                probs_ref += _probs_ref.nan_to_num()
                for elem in _atoms_per_elem.keys():
                    if elem in atoms_per_elem:
                        atoms_per_elem[elem] += _atoms_per_elem[elem]
                    else:
                        atoms_per_elem[elem] = _atoms_per_elem[elem]
            weights /= weights_counts
            probs_ref /= probs_ref_counts
        else:
            with open(path, "rb") as f:
                zs, weights, _, probs_ref, atoms_per_elem, _, _, _ = pickle.load(f)

            n_elements = len(elements)
            zs = zs[:n_elements]
            weights = weights[:n_elements, :n_elements]
            probs_ref = probs_ref[:n_elements, :n_elements]
            atoms_per_elem = {
                k: v for k, v in atoms_per_elem.items() if k in num2elem.keys()
            }

        # subplot 0
        axes[0].imshow(probs_ref, cmap="Reds", vmin=0, vmax=1)
        axes[0].set(
            xticks=range(len(elements)),
            yticks=range(len(elements)),
            xticklabels=elements,
            yticklabels=elements,
        )
        axes[0].set_ylabel("$z_i$", fontsize=15)
        if dataset_idx == 0:
            axes[0].set_title("Bond Probabilities", fontsize=12)
            axes[0].tick_params(labelleft=True)
        else:
            axes[0].tick_params(labelleft=True, top=True)
        if dataset_idx == len(paths) - 1:
            axes[0].set_xlabel("$z_j$", fontsize=15)

        # subplot 1
        if normalize_attention:
            mask = ~weights.isnan().all(dim=1)
            weights[mask] = (
                weights[mask] / weights[mask].nansum(dim=1, keepdim=True).abs()
            )
        axes[1].imshow(weights, cmap="Blues")
        axes[1].set(
            xticks=range(len(elements)),
            yticks=range(len(elements)),
            xticklabels=elements,
            yticklabels=elements,
        )
        if dataset_idx == 0:
            axes[1].set_title("Attention Scores", fontsize=12)
        else:
            axes[1].tick_params(top=True)
        if dataset_idx == len(paths) - 1:
            axes[1].set_xlabel("$z_j$", fontsize=15)

        # subplot 2
        axes[2].barh(
            range(len(atoms_per_elem.keys())),
            atoms_per_elem.values(),
            color="forestgreen",
        )
        for i, v in enumerate(atoms_per_elem.values()):
            is_max = v >= max(atoms_per_elem.values()) * 0.70
            offset = max(atoms_per_elem.values()) * 0.025
            axes[2].text(
                v - offset if is_max else v + offset,
                i,
                str(v),
                va="center",
                ha="right" if is_max else "left",
                color="1" if is_max else "0",
            )
        axes[2].set_box_aspect(1)
        axes[2].set_xticks([])
        if dataset_idx == 0:
            axes[2].set_title("Total", fontsize=12)
        axes[2].tick_params(labelright=True)

        for ax in axes:
            ax.tick_params(color="0.5", right=True)
            for spine in ax.spines.values():
                spine.set_edgecolor("0.5")

    plt.savefig(join(basedir, "attn_weights.pdf"), bbox_inches="tight")

    if distance_plots:
        for path_idx, path in enumerate(paths):
            print(f"creating dist-attention plot ({path_idx + 1}/{len(paths)})")
            # load data
            with open(path, "rb") as f:
                _, _, _, _, _, zs_full, attn_full, dist = pickle.load(f)

            # visualize attention by distance
            z1, z2 = 1, 6
            ma_width = 1000
            mask = ((zs_full[0] == z1) & (zs_full[1] == z2)) | (
                (zs_full[0] == z2) & (zs_full[1] == z1)
            )
            fig, ax = plt.subplots()
            ax.grid(True)
            ax.hist(dist[mask].numpy(), bins=70, color="C1", alpha=0.3)
            ax.set_ylabel("Number of interactions", color="C1")
            ax = ax.twinx()
            averaged = (
                pd.Series(attn_full[mask][dist[mask].argsort()])
                .rolling(ma_width)
                .mean()
                .shift(-ma_width)
                .values
            )
            ax.scatter(dist[mask].sort().values, averaged, marker=".", color="C0")
            ax.set_xlabel("Distance ($\AA$)")
            ax.set_ylabel("Attention score", color="C0")
            ax.set_title(
                "Attention scores by distance for Hydrogen-Carbon interactions"
            )
            ax.set_xlim(0)
            plt.savefig(join(dirname(path), "attn-dist.pdf"), bbox_inches="tight")
    print("done")


if __name__ == "__main__":
    # fmt: off
    parser = argparse.ArgumentParser(description='Analyze Attention Weights')
    parser.add_argument('--extract-data', type=bool, help='Whether to extract the attention weights or use previously stored data')
    parser.add_argument('--model-path', type=str, help='Path to a model checkpoint with corresponding splits.npz in the same directory')
    parser.add_argument('--dataset-path', type=str, help='Path to the directory containing the dataset')
    parser.add_argument('--dataset-name', type=str, choices=datasets.__all__, help='Name of the dataset')
    parser.add_argument('--dataset-arg', type=str, help='Additional argument to the dataset class (e.g. target property for QM9)')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for the attention weight extraction')
    parser.add_argument('--plot-molecules', type=bool, help='The visualization system for molecules')
    parser.add_argument('--distance-plots', type=bool, help='If True, create distance-attention plots')
    parser.add_argument('--normalize-attention', type=bool, help='Whether to normalize the attention scores such that each row adds up to one')
    parser.add_argument('--combine-dataset', type=bool, help='Whether to combine all data from the same dataset')
    parser.add_argument('--ignore-datasets', type=str, default='', help='Comma separated names of datasets not to include in the plots')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the extraction on')
    # fmt: on

    args = parser.parse_args()

    if args.extract_data:
        extract_data(
            args.model_path,
            args.dataset_path,
            args.dataset_name,
            args.dataset_arg,
            args.batch_size,
            args.plot_molecules,
            args.device,
        )

    visualize(
        dirname(dirname(args.model_path)),
        args.normalize_attention,
        args.distance_plots,
        args.combine_dataset,
        args.ignore_datasets.split(","),
    )
