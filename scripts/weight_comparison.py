import os
from os.path import join, basename, splitext, dirname
import sys
from glob import glob
import subprocess
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt


zoom = 1.5
num_mols = 3
visualize = False
img_dir = "/home/philipp/Desktop/imgs/"

model_paths = [
    "/home/philipp/Documents/models/QM9-energy_U0-et/epoch*",
    "/home/philipp/Documents/models/MD17-uracil-et/epoch*",
    "/home/philipp/Documents/models/ANI1-energy-et/epoch*",
]
dataset_name = "QM9"

dset2name = {"ANI1": "ANI-1", "QM9": "QM9", "MD17": "MD17 (_mol_)"}
dset2splits = {
    "QM9": "/home/philipp/Documents/models/QM9-energy_U0-et/splits.npz",
    "MD17": "/home/philipp/Documents/models/MD17-uracil-et/splits.npz",
    "ANI1": "/home/philipp/Documents/models/ANI1-energy-et/splits.npz",
}

# create images
if visualize:
    for mol_idx in np.random.permutation(100)[:num_mols]:
        for model_path in model_paths:
            model_path_full = glob(model_path)[0]
            subprocess.check_call(
                [
                    sys.executable,
                    "scripts/visualize_attention.py",
                    "--molecule-idx",
                    str(mol_idx),
                    "--model-path",
                    model_path_full,
                    "--splits-path",
                    dset2splits[dataset_name],
                    "--dataset-name",
                    dataset_name,
                    "--dataset-path",
                    f"/home/philipp/Documents/data/{dataset_name.lower()}/",
                    "--render",
                    "true",
                    "--zoom",
                    str(zoom),
                ]
            )
            img_path = f"{dataset_name}-{mol_idx}.png"
            model_name = model_path.split(os.sep)[-2].replace("-", "_")
            os.rename(img_path, join(img_dir, f"{model_name}-{img_path}"))

# load images
images = dict()
for path in sorted(glob(join(img_dir, f"*-{dataset_name}-*.png"))):
    model_name = splitext(basename(path))[0].split("-")[0]
    if model_name in images:
        images[model_name].append(Image.open(path))
    else:
        images[model_name] = [Image.open(path)]

plt.rcParams["mathtext.fontset"] = "cm"

text_size = 12
title_size = 15

# create plot
plt.style.use("seaborn-dark")
fig, axes = plt.subplots(
    len(model_paths),
    num_mols + 1,
    gridspec_kw=dict(width_ratios=[0.1] + [1] * num_mols, wspace=0, hspace=0),
)
fig.suptitle(
    f"Molecules from {dset2name[dataset_name].replace('_mol_', 'aspirin')}",
    fontsize=title_size,
)
fig.text(
    0.08, 0.5, "model trained on", va="center", rotation="vertical", fontsize=title_size
)

for i, model_name in enumerate(images.keys()):
    axes[i, 0].axis("off")
    axes[i, 0].text(
        0.5,
        0.5,
        dset2name[model_name.split("_")[0]].replace("_mol_", "uracil"),
        rotation=90,
        ha="center",
        va="center",
        fontsize=text_size,
    )

    for mol_idx in range(num_mols):
        if i == 0:
            axes[i, mol_idx + 1].set_title(
                f"molecule {mol_idx + 1}", fontsize=text_size
            )
        axes[i, mol_idx + 1].imshow(images[model_name][mol_idx])
        axes[i, mol_idx + 1].axis("off")
plt.savefig(f"/tmp/{dataset_name.lower()}-attention.pdf", bbox_inches="tight", dpi=400)
