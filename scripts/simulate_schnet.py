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
from time import ctime
from os.path import join

if __name__ == "__main__":
    device = torch.device('cuda:2')
    print(f'Simulation starts on: {ctime()}')

    n_sims = 500
    simulation_time = 5e3 # ps

    save_interval = 10
    export_interval = 100000
    log_interval = 100000
    dt = 0.005 # ps
    friction = 1 # ps ^-1


    temperature = 350 # K
    k_b = 0.001985875 #   kcal/mol / K
    beta = 1 / (k_b * temperature) #   kcal/mol ^-1
    masses = np.array([
            12, 12, 12, 12, 12,
            12, 12, 12, 12, 12,
        ])
    mass_scale = 418.4 #

    ckpt = {
        49 : 'epoch=49-validation_loss=739.7512-test_loss=0.0000.ckpt',
        25 : 'epoch=25-validation_loss=740.0948-test_loss=0.0000.ckpt',
        15 : 'epoch=15-validation_loss=740.3742-test_loss=0.0000.ckpt',
    }


    basefn = '/local_scratch/musil/chign/schnet/test_1/'

    chignolin_dataset = ChignolinDataset('/local_scratch/musil/datasets/chignolin/')

    n_timesteps = int(simulation_time / dt)
    print(f'of {n_sims} trajectories for {simulation_time} [ps] ({n_timesteps} timesteps)')

    baseline_model = chignolin_dataset.get_baseline_model(n_beads=10)  # doesn't work without specifying n_beads


    prefix = f'traj-e_{ckpt[15]}-dt_{dt*1000:d0}fs'

    model = SchNet.load_from_checkpoint(
        join(basefn, ckpt[15])
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
                    filename=join(basefn,prefix)
    )


    traj = sim.simulate()

    print(f'Simulation ended on: {ctime()}')
    # np.save('/local_scratch/musil/chign/test_4/traj.npy', traj)
    # with torch.profiler.profile(
    #     on_trace_ready=torch.profiler.tensorboard_trace_handler('/local_scratch/musil/chign/test_5'),
    #     record_shapes=False,
    #     with_stack=True
    #     ) as prof:

    #     traj = sim.simulate()

    # torch.save(traj, '/local_scratch/musil/chign/traj.pt')

    # fig,_, tica = plot_tica(baseline_model, chignolin_dataset, lag=10)
    # plt.savefig('/local_scratch/musil/chign/ref_traj.png', dpi=300, bbox_inches='tight')

    # fig,_,_ = plot_tica(baseline_model, traj, tica=tica)
    # plt.savefig('/local_scratch/musil/chign/simulated_traj.png', dpi=300, bbox_inches='tight')