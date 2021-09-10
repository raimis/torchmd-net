import time
import os
from os.path import exists, join, dirname, basename
import argparse
import pickle
import numpy as np
import torch
import glob
from torchmdnet import datasets, attention_weights
from torchmdnet.models.model import load_model
from tqdm import tqdm
from matplotlib import pyplot as plt


# fmt: off
parser = argparse.ArgumentParser(description='Displacement attention weights')
parser.add_argument('--model-path', type=str, help='Path to a model checkpoint')
parser.add_argument('--splits-path', type=str, help='Path to a splits.npz file for the dataset')
parser.add_argument('--qm9-path', type=str, help='Path to the directory containing the dataset')
parser.add_argument('--disp-strength', type=float, default=0.4, help='Displacement strength in Angström')
parser.add_argument('--extract-data', type=bool, help='Whether to extract the data or just create the figure')
parser.add_argument('--device', type=str, default='cpu', help='Device to run the extraction on')
# fmt: on

args = parser.parse_args()

torch.manual_seed(1234)
np.random.seed(1234)
torch.set_grad_enabled(False)

num2elem = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}

# extract data
if args.extract_data:
    # load data
    print("loading data")
    data = datasets.QM9(args.qm9_path, dataset_arg="energy_U0")

    if exists(args.splits_path):
        splits = np.load(args.splits_path)
        data = torch.utils.data.Subset(data, splits["idx_test"])
        print(f"found {len(data)} samples in the test set")
    else:
        print(f"found {len(data)} samples")

    # load model
    print("loading model")
    model = load_model(args.model_path, device=args.device).eval()
    # initialize attention weight collector
    attention_weights.reset(model.representation_model.num_layers)

    displaced = {1: [], 6: [], 8: []}
    normal = {1: [], 6: [], 8: []}

    # for mol_idx in tqdm(range(len(data))):
    for mol_idx in tqdm(range(10)):
        if 7 in data[mol_idx].z or 9 in data[mol_idx].z:
            continue

        # displace single atom
        for i in range(len(data[mol_idx].z)):
            x = data[mol_idx].clone()
            offset = np.random.normal(0, 1, 3)
            offset /= np.linalg.norm(offset) / args.disp_strength
            x.pos[i] += offset
            # extract attention weights
            model(x.z.to(args.device), x.pos.to(args.device))

            attn_idx_displaced = torch.where(attention_weights.rollout_index[-1] == i)[
                1
            ]
            attn_idx_normal = torch.where(attention_weights.rollout_index[-1] != i)[1]
            displaced[int(x.z[i])].append(
                attention_weights.rollout_weights[-1][attn_idx_displaced]
                .abs()
                .mean()
                .numpy()
                / attention_weights.rollout_weights[-1].abs().max().numpy()
            )
            normal[int(x.z[i])].append(
                attention_weights.rollout_weights[-1][attn_idx_normal]
                .abs()
                .mean()
                .numpy()
                / attention_weights.rollout_weights[-1].abs().max().numpy()
            )

    model_name = basename(dirname(args.model_path)).split("-")[0]
    with open(
        join(dirname(dirname(args.model_path)), f"{model_name}-displacement-attn.pkl"),
        "wb",
    ) as f:
        pickle.dump((normal, displaced), f)

# load data
data_paths = glob.glob(
    join(dirname(dirname(args.model_path)), "*-displacement-attn.pkl")
)
data = dict()
for path in data_paths:
    model_name = path.split(os.sep)[-1].split("-")[0]
    with open(path, "rb") as f:
        data[model_name] = pickle.load(f)

# create plot
title_size = 15

plt.style.use("seaborn-dark")
fig, axes_grid = plt.subplots(
    nrows=len(data),
    ncols=4,
    squeeze=False,
    sharex=True,
    sharey=True,
    gridspec_kw=dict(width_ratios=[0.3, 1, 1, 1], wspace=0.2, hspace=0.2),
)
fig.text(
    0.05, 0.5, "model trained on", va="center", rotation="vertical", fontsize=title_size
)

for i, (axes, model_name) in enumerate(zip(axes_grid, data.keys())):
    axes[0].axis("off")
    axes[0].text(
        0,
        0.5,
        model_name,
        ha="right",
        va="center",
        transform=axes[0].transAxes,
        fontsize=title_size,
        rotation=90,
    )

    for j, (ax, (z, disp), (_, norm)) in enumerate(
        zip(axes[1:], data[model_name][1].items(), data[model_name][0].items())
    ):
        if i == 0:
            ax.set_title(num2elem[z], fontsize=title_size)
        ax.bar(
            range(2),
            [np.mean(norm), np.mean(disp)],
            color=["C0", "C1"],
            yerr=[np.std(norm), np.std(disp)],
            error_kw=dict(elinewidth=2, capsize=4, markeredgewidth=1),
        )
        ax.set_xticks(range(2))
        ax.set_xticklabels(
            ["equilibrium", "displaced"], rotation=30, ha="right", fontsize=title_size
        )
        if j == 0:
            ax.tick_params("y", labelleft=True, labelsize=title_size)
plt.tight_layout()
plt.savefig(
    join(dirname(dirname(args.model_path)), "displacement.pdf"),
    dpi=400,
    bbox_inches="tight",
)
