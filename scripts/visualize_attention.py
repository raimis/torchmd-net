import time
from os import path
import argparse
import numpy as np
import torch
from torchmdnet import datasets, attention_weights
from torchmdnet.models.model import load_model
from moleculekit.molecule import Molecule
from moleculekit.vmdgraphics import VMDCylinder
from moleculekit.vmdviewer import getCurrentViewer
from PIL import Image


# fmt: off
parser = argparse.ArgumentParser(description='Visualize Attention Weights')
parser.add_argument('--molecule-idx', type=int, default=0, help='Index of the molecule to visualize')
parser.add_argument('--model-path', type=str, help='Path to a model checkpoint with corresponding splits.npz in the same directory')
parser.add_argument('--dataset-path', type=str, help='Path to the directory containing the dataset')
parser.add_argument('--dataset-name', type=str, choices=datasets.__all__, help='Name of the dataset')
parser.add_argument('--top-n', type=int, default=10, help='Number of attention weights to visualize')
parser.add_argument('--render', type=bool, help='Whether to render and save the molecule')
parser.add_argument('--zoom', type=float, default=1.0, help='The zoom factor when rendering the molecule')
parser.add_argument('--device', type=str, default='cpu', help='Device to run the extraction on')
# fmt: on

args = parser.parse_args()

torch.manual_seed(1234)
np.random.seed(1234)
torch.set_grad_enabled(False)

num2elem = {1: "H", 6: "C", 7: "N", 8: "O", 9: "F"}

# load data
print("loading data")
dataset = getattr(datasets, args.dataset_name)
try:
    dataset_arg = dataset.available_dataset_args[0]
except IndexError:
    dataset_arg = ""
data = dataset(args.dataset_path, dataset_arg=dataset_arg)

splits_path = path.join(path.dirname(args.model_path), "splits.npz")
if path.exists(splits_path):
    splits = np.load(splits_path)
    if len(splits["idx_train"]) + len(splits["idx_val"]) + len(
        splits["idx_test"]
    ) == len(data):
        data = torch.utils.data.Subset(data, splits["idx_test"])
        print(f"found {len(data)} samples in the test set")
    else:
        print(f"found {len(data)} samples")
else:
    print("Couldn't find splits.npz, using the whole dataset")
    print(f"found {len(data)} samples")

# load model
print("loading model")
model = load_model(args.model_path, device=args.device).eval()
# initialize attention weight collector
attention_weights.reset(model.representation_model.num_layers)

# extract attention weights from model
x = data[np.random.permutation(len(data))[args.molecule_idx]]
model(x.z.to(args.device), x.pos.to(args.device))

vmd = getCurrentViewer()

mol = Molecule().empty(len(x.z))
mol.coords = x.pos.unsqueeze(2).numpy()
mol.element[:] = [num2elem[num] for num in x.z.numpy()]
mol.name[:] = [num2elem[num] for num in x.z.numpy()]

if x.edge_index is None:
    # guess bonds
    idxs = torch.from_numpy(mol._guessBonds().T.astype(np.int64))
    x.edge_index = torch.cat([idxs, idxs.flip(dims=(0,))], dim=1)

# visualize using VMD
max_attn = attention_weights.rollout_weights[-1].abs().max()
for idx1, idx2 in attention_weights.rollout_index[-1].T:
    attn_idx = torch.where(
        (attention_weights.rollout_index[-1][0] == idx1)
        & (attention_weights.rollout_index[-1][1] == idx2)
    )[0]
    strength = float(attention_weights.rollout_weights[-1][attn_idx])
    scaled_strength = float(strength / max_attn)

    draw_threshold = sorted(attention_weights.rollout_weights[-1].abs())[-args.top_n]
    if abs(strength) >= draw_threshold:
        vmd.send("mol new")
        vmd.send(f"material add mat-{idx1}-{idx2}")
        vmd.send(
            f"material change opacity mat-{idx1}-{idx2} {abs(scaled_strength) ** 2}"
        )
        vmd.send(f"material change specular mat-{idx1}-{idx2} 0")
        vmd.send(f"draw material mat-{idx1}-{idx2}")

        if strength >= draw_threshold:
            VMDCylinder(
                mol.coords[idx1].flatten(),
                mol.coords[idx2].flatten(),
                color="red",
                radius=0.05 * abs(scaled_strength),
            )
        elif strength <= -draw_threshold:
            VMDCylinder(
                mol.coords[idx1].flatten(),
                mol.coords[idx2].flatten(),
                color="blue",
                radius=0.05 * abs(scaled_strength),
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
vmd.send(f"scale by {args.zoom}")

if args.render:
    img_name = f"{args.dataset_name}-{args.molecule_idx}.png"
    vmd.send(f"render TachyonLOSPRayInternal {img_name}")
    vmd.close()

    img = np.array(Image.open(img_name))
    mask = (img != 254).any(axis=(1, 2)) | (img != 254).any(axis=(0, 2))
    Image.fromarray(img[mask][:, mask]).save(img_name)

while not vmd.completed():
    time.sleep(0.1)
