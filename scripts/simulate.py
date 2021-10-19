from mkl import get_max_threads,set_num_threads
set_num_threads(16)

import argparse

import sys, os
import numpy as np
import pyemma as pe
import pyemma
import torch  # pytorch
import matplotlib.pyplot as plt

sys.path.insert(0,'/home/musil/git/torchmd-net/')
from torchmdnet2.dataset import ChignolinDataset, DataModule
from torchmdnet2.models import LNNP, SchNet, MLPModel, CGnet
from torchmdnet2.utils import LoadFromFile, save_argparse
from torchmdnet2.simulation import Simulation

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from pytorch_lightning.plugins import DDPPlugin

from torch_geometric.data import DataLoader

from torch.nn import Embedding, Sequential, Linear, ModuleList

from os.path import join
from time import ctime

# def get_args():
#     # fmt: off
#     parser = argparse.ArgumentParser(description='Simulate')
#     parser.add_argument('--dataset_root', type=str, help='root path to the dataset')
#     parser.add_argument('--checkpoint_fn', type=str, help='path to the checkpoint to load the model from')
#     parser.add_argument('--device', default='cuda' type=str, help='specify the device to run the simulation on')
#     parser.add_argument('--model', )

#     args = parser.parse_args()

#     save_argparse(args, os.path.join(args.log_dir, 'simulation_input.yaml'), exclude=['conf'])

#     return args

if __name__ == "__main__":
    device = torch.device('cuda:3')
    print(f'Simulation starts on: {ctime()}')

    # basefn = '/local_scratch/musil/chign/soap_mlp/rs_diff_species_0'
    # prefix = 'ep99_dt5'
    # ckpt = 'epoch=99-validation_loss=27.2705-test_loss=0.0000.ckpt'
    # basefn = '/local_scratch/musil/chign/soap_mlp/diff_species_1'
    # prefix = 'traj-ep_41'
    basefn = '/local_scratch/musil/chign/soap_mlp/diff_species_2/'
    ckpt_id = 49
    ckpt = {
        9 : 'epoch=9-validation_loss=27.2639-test_loss=0.0000.ckpt',
        19 : 'epoch=19-validation_loss=27.2561-test_loss=0.0000.ckpt',
        29 : 'epoch=29-validation_loss=27.2540-test_loss=0.0000.ckpt',
        49 : 'epoch=49-validation_loss=27.2462-test_loss=0.0000.ckpt'
    }

    n_sims = 400
    simulation_time = 2e3 # ps

    save_interval = 20
    export_interval = 100000
    log_interval = 100000
    dt = 0.005 # ps
    friction = 5 # ps ^-1

    n_timesteps = int(simulation_time / dt)
    print(f'of {n_sims} trajectories for {simulation_time} [ps] ({n_timesteps} timesteps)')

    temperature = 350 # K
    k_b = 0.001985875 #   kcal/mol / K
    beta = 1 / (k_b * temperature) #   kcal/mol ^-1

    masses = np.array([
            12, 12, 12, 12, 12,
            12, 12, 12, 12, 12,
        ])
    mass_scale = 418.4 #

    prefix = f'traj-e_{ckpt_id:d}-dt_{dt*1000:.0f}fs'
    outfn = join(basefn,prefix,'traj')
    if not os.path.exists(join(basefn,prefix)):
        print(f'Making {join(basefn,prefix)}')
        os.mkdir(join(basefn,prefix))

    chignolin_dataset = ChignolinDataset('/local_scratch/musil/datasets/chignolin/')

    baseline_model = chignolin_dataset.get_baseline_model(n_beads=10)  # doesn't work without specifying n_beads

    model = MLPModel.load_from_checkpoint(
        join(basefn, ckpt[ckpt_id])
    )


    ids = np.arange(0, len(chignolin_dataset),len(chignolin_dataset)//n_sims).tolist()
    init = chignolin_dataset[ids]
    initial_coords = torch.cat([init[i].pos.reshape((1,-1,3)) for i in range(len(init))], dim=0).to(device=device)
    initial_coords.requires_grad_()

    sim_embeddings = torch.cat([init[i].z.reshape((1,-1)) for i in range(len(init))], dim=0).to(device=device)

    # initializing the Net with the learned weights and biases and preparing it for evaluation
    chignolin_net = CGnet(model, baseline_model).eval().to(device=device)


    sim = Simulation(chignolin_net, initial_coords, sim_embeddings,
                    length=n_timesteps, dt=dt,
                    friction=friction,masses=masses/mass_scale,
                    save_interval=save_interval, beta=beta,
                    save_potential=True, device=device,
                    log_interval=log_interval, log_type='print',
                    batch_size=600, export_interval=export_interval,
                    filename=outfn,
    )


    traj = sim.simulate()

    print(f'Simulation ended on: {ctime()}')
