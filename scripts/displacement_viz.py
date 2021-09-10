import time
from os import path
import argparse
import numpy as np
from copy import deepcopy
import torch
from torchmdnet import datasets, attention_weights
from torchmdnet.models.model import load_model
from moleculekit.molecule import Molecule
from moleculekit.vmdgraphics import VMDCylinder
from moleculekit.vmdviewer import getCurrentViewer
from PIL import Image


# fmt: off
parser = argparse.ArgumentParser(description='Displacement attention weights')
parser.add_argument('--mol-idx', type=int, default=0, help='Index of the molecule')
parser.add_argument('--disp-strength', type=float, default=0.4, help='Displacement strength in Angström')
parser.add_argument('--model-path', type=str, help='Path to a model checkpoint')
parser.add_argument('--splits-path', type=str, help='Path to a splits.npz file for the dataset')
parser.add_argument('--qm9-path', type=str, help='Path to the directory containing the dataset')
parser.add_argument('--top-n', type=int, default=10, help='Number of attention weights to visualize')
parser.add_argument('--render', type=bool, help='Whether to render and save the molecule')
parser.add_argument('--zoom', type=float, default=1.0, help='The zoom factor when rendering the molecule')
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

mol_idx = np.random.permutation(len(data))[args.mol_idx]

# load model
print("loading model")
model = load_model(args.model_path).eval()
# initialize attention weight collector
attention_weights.reset(model.representation_model.num_layers)

x = data[mol_idx]
while 7 in x.z or 9 in x.z:
    mol_idx += 1
    x = data[mol_idx]
atom_idx = torch.where(x.z == 6)[0][3]

mol_ref = Molecule().empty(len(x.z))
mol_ref.coords = x.pos.unsqueeze(2).numpy()
mol_ref.element[:] = [num2elem[num] for num in x.z.numpy()]
mol_ref.name[:] = [num2elem[num] for num in x.z.numpy()]

x = x.clone()
offset = np.random.normal(0, 1, 3)
offset /= np.linalg.norm(offset) / args.disp_strength
x.pos[atom_idx] += offset

mol = deepcopy(mol_ref)
mol.coords = x.pos.unsqueeze(2).numpy()

# extract attention weights
model(x.z, x.pos)

vmd = getCurrentViewer()

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
# vmd.send("display projection Perspective")

vmd.send("mol representation VDW 0.125 12")
vmd.send("mol selection all")
vmd.send("mol material AOChalky")
vmd.send("mol addrep top")
vmd.send(f"scale by {args.zoom}")

# reference molecule
bonds = mol_ref._guessBonds()
bonded = np.argwhere(bonds == atom_idx.numpy())
bonded[:, 1] = 1 - bonded[:, 1]
bonded_idxs = "".join([f" or index {i}" for i in bonds[bonded[:, 0], bonded[:, 1]]])

mol_ref.name[:] = ["R"] * len(mol_ref.name)
mol_ref.view(style="Licorice", sel=f"index {atom_idx}{bonded_idxs}", viewerhandle=vmd)
vmd.send("color Name R yellow")
vmd.send("color change rgb yellow 1 0 1")

vmd.send("material add reference")
vmd.send("material change ambient reference 0.5")
vmd.send("material change diffuse reference 1")
vmd.send("material change specular reference 0")
vmd.send("material change opacity reference 1")

vmd.send("mol modstyle 0 top Licorice 0.1 12 12")
vmd.send("mol modmaterial 0 top reference")

vmd.send("mol representation VDW 0.12 12")
vmd.send("mol material reference")
vmd.send("mol addrep top")

if args.render:
    img_name = f"QM9-displaced.png"
    vmd.send(f"render TachyonLOSPRayInternal {img_name}")
    vmd.close()

    img = np.array(Image.open(img_name))
    mask = (img != 254).any(axis=(1, 2)) | (img != 254).any(axis=(0, 2))
    Image.fromarray(img[mask][:, mask]).save(img_name)

while not vmd.completed():
    time.sleep(0.1)
