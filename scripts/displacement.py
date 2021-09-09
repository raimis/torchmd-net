import time
from os import path
import argparse
import numpy as np
import torch
from torchmdnet import datasets, attention_weights
from torchmdnet.models.model import load_model
from tqdm import tqdm
from matplotlib import pyplot as plt


# fmt: off
parser = argparse.ArgumentParser(description='Displacement attention weights')
parser.add_argument('--model-path', type=str, help='Path to a model checkpoint')
parser.add_argument('--splits-path', type=str, help='Path to a splits.npz file for the dataset')
parser.add_argument('--qm9-path', type=str, help='Path to the directory containing the dataset')
parser.add_argument('--top-n', type=int, default=10, help='Number of attention weights to visualize')
parser.add_argument('--device', type=str, default='cpu', help='Device to run the extraction on')
# fmt: on

args = parser.parse_args()

torch.manual_seed(1234)
np.random.seed(1234)
torch.set_grad_enabled(False)

num2elem = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}

# load data
print("loading data")
data = datasets.QM9(args.qm9_path, dataset_arg="energy_U0")

if path.exists(args.splits_path):
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

mol_idx = 0
total = 100
progress = tqdm(total=total)
while progress.n < total:
    if 7 in data[mol_idx].z or 9 in data[mol_idx].z:
        mol_idx += 1
        continue

    # displace single atom
    for i in range(len(data[mol_idx].z)):
        x = data[mol_idx].clone()
        offset = np.random.normal(0, 1, 3)
        offset /= np.linalg.norm(offset) / 0.4
        x.pos[i] += offset
        # extract attention weights
        model(x.z.to(args.device), x.pos.to(args.device))

        attn_idx_displaced = torch.where(attention_weights.rollout_index[-1] == i)[1]
        attn_idx_normal = torch.where(attention_weights.rollout_index[-1] != i)[1]
        displaced[int(x.z[i])].append(
            attention_weights.rollout_weights[-1][attn_idx_displaced]
            .abs()
            .mean()
            .numpy()
            / attention_weights.rollout_weights[-1].abs().max().numpy()
        )
        normal[int(x.z[i])].append(
            attention_weights.rollout_weights[-1][attn_idx_normal].abs().mean().numpy()
            / attention_weights.rollout_weights[-1].abs().max().numpy()
        )
    mol_idx += 1
    progress.update()

plt.style.use("seaborn-dark")
fig, axes = plt.subplots(ncols=3)
for ax, (z, disp), (_, norm) in zip(axes, displaced.items(), normal.items()):
    ax.set_title(num2elem[z])
    ax.bar(
        range(2),
        [np.mean(norm), np.mean(disp)],
        yerr=[np.std(norm), np.std(disp)],
    )
    ax.set_xticks(range(2))
    ax.set_xticklabels(["energy-minimized", "displaced"])
plt.tight_layout()
plt.show()
