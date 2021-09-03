#!/usr/bin/env python
# coding: utf-8
#!/bin/bash 
#SBATCH -J SchNet_AA_sim 
#SBATCH -D /data/scratch/schreibef98/projects
#SBATCH -o SchNet_AA_sim.%j.out 
#SBATCH --partition=gpu 
#SBATCH --nodes=1 
#SBATCH --gres=gpu:4 
#SBATCH --mem=10M 
#SBATCH --time=20:00:00 
#SBATCH --mail-type=end 
#SBATCH --mail-user= franz.josef.schreiber@fu-berlin.de

#from mkl import get_max_threads,set_num_threads
#set_num_threads(16)

import argparse

import sys, os
import numpy as np
import torch  # pytorch


sys.path.insert(0,'home/schreibef98/projects/torchmd-net/')
from torchmdnet2.dataset import ChignolinDataset, DataModule
from torchmdnet2.models import LNNP, SchNet, MLPModel, CGnet
from torchmdnet2.utils import LoadFromFile, save_argparse
from torchmdnet2.simulation import Simulation, PTSimulation
device = torch.device('cuda')


def main():
    # # Utils
    from torch_geometric.data.data import size_repr
    
    from argparse import Namespace
    class Args(Namespace):
        def __init__(self,**kwargs):
            for key, item in kwargs.items():
                self[key] = item
                
        def __getitem__(self, key):
            r"""Gets the data of the attribute :obj:`key`."""
            return getattr(self, key, None)
    
        def __setitem__(self, key, value):
            """Sets the attribute :obj:`key` to :obj:`value`."""
            setattr(self, key, value)
    
        @property
        def keys(self):
            r"""Returns all names of graph attributes."""
            keys = [key for key in self.__dict__.keys() if self[key] is not None]
            keys = [key for key in keys if key[:2] != '__' and key[-2:] != '__']
            return keys
    
        def __len__(self):
            r"""Returns the number of all present attributes."""
            return len(self.keys)
    
        def __contains__(self, key):
            r"""Returns :obj:`True`, if the attribute :obj:`key` is present in the
            data."""
            return key in self.keys
    
        def __iter__(self):
            r"""Iterates over all present attributes in the data, yielding their
            attribute names and content."""
            for key in sorted(self.keys):
                yield key, self[key]
    
        def __call__(self, *keys):
            r"""Iterates over all attributes :obj:`*keys` in the data, yielding
            their attribute names and content.
            If :obj:`*keys` is not given this method will iterative over all
            present attributes."""
            for key in sorted(self.keys) if not keys else keys:
                if key in self:
                    yield key, self[key]
                    
        def __repr__(self):
            cls = str(self.__class__.__name__)
            has_dict = any([isinstance(item, dict) for _, item in self])
    
            if not has_dict:
                info = [size_repr(key, item) for key, item in self]
                return '{}({})'.format(cls, ', '.join(info))
            else:
                info = [size_repr(key, item, indent=2) for key, item in self]
                return '{}(\n{}\n)'.format(cls, ',\n'.join(info))
    
    
    # # Load Model
    chignolin_dataset = ChignolinDataset('/home/schreibef98/projects/torchmd-net/datasets/chignolin_AA/')
    
    args = Args(**{
        
        'batch_size': 512,
     
        'load_model': None,
        'log_dir': '/home/schreibef98/projects/torchmd-net/notebooks/chignolin_logs/test_03_AA',
        
        'dataset_name': 'chignolin',
        'dataset_root': chignolin_dataset.root,
        'dataset_stride' : 1,
        'target_name': 'forces',
        
        'derivative': True,
        'distributed_backend': 'ddp_spawn',
        'accelerator': 'ddp_spawn',
        'num_nodes': 1,
        'early_stopping_patience': 100,
        'inference_batch_size': 1024,
        'label': None,
        
        'activation': 'tanh',
        'embedding_dimension': 128,
        'cutoff_lower': 0.0,
        'cutoff_upper': 30.0,
        'num_filters': 128,
        'num_interactions': 2,
        'num_rbf': 300,
        'trainable_rbf': False,
        'rbf_type': 'gauss',
        'neighbor_embedding': True,
        'cfconv_aggr': 'mean',
        'n_layers' : 1,
        'reduction_factor' : 1,
      
        'lr': 1e-4,
        'lr_factor': 0.999,
        'lr_min': 1e-7,
        'lr_patience': 10,
        'lr_warmup_steps': 0,
        
        'ngpus': -1,
        'num_epochs': 50,
        'num_workers': 8,
        'save_interval': 10,
        'seed': 18574,
        'test_interval': 1,
        'test_ratio': 0.1,
        'val_ratio': 0.1,
        'weight_decay': 0.0,
        'precision': 32,
        
        'data' : None,
        'coords' : None,
        'forces' : None,
        'embed' : None,
        'splits' : None,
    })
    
    model_AA = SchNet(
        hidden_channels=args.embedding_dimension,
        num_filters=args.num_filters,
        num_interactions=args.num_interactions,
        num_rbf=args.num_rbf,
        rbf_type=args.rbf_type,
        trainable_rbf=args.trainable_rbf,
        activation=args.activation,
        neighbor_embedding=args.neighbor_embedding,
        cutoff_lower=args.cutoff_lower,
        cutoff_upper=args.cutoff_upper,
        derivative=args.derivative,
        cfconv_aggr=args.cfconv_aggr,
    )
    
    model_AA.load_state_dict(torch.load('/home/schreibef98/projects/torchmd-net/notebooks/chignolin_logs/test_01/last_model_AA.pt'))
    model_AA.to(device)
    
    baseline_model = chignolin_dataset.get_baseline_model(n_beads=10)
    chignolin_net = CGnet(model_AA, baseline_model).eval()
    chignolin_net.to(device=device)
    
    
    # # Set up simulation
    T = np.array([300, 350, 450, 600])
    R = 8.314462
    e = (R*T)/4184
    betas = 1/e
    
    n_sims = 1000
    n_timesteps = 10000000
    save_interval = 1000
    
    
    ids = np.arange(0, len(chignolin_dataset),len(chignolin_dataset)//n_sims).tolist()
    init = chignolin_dataset[ids]
    initial_coords = torch.cat([init[i].pos.reshape((1,-1,3)) for i in range(len(init))], dim=0).to(device=device)
    initial_coords.requires_grad_()
    
    sim_embeddings = torch.cat([init[i].z.reshape((1,-1)) for i in range(len(init))], dim=0).to(device=device)
    
    mass_scale = 418.4
    masses = list(12*np.ones(10)/mass_scale)
    
    sim = PTSimulation(chignolin_net, initial_coords, sim_embeddings, length=n_timesteps,
                     save_interval=save_interval, betas=betas,
                     save_potential=True, device=device,
                     log_interval=100, log_type='print', masses=masses, friction=1.0)
    
    traj = sim.simulate()
    
    torch.save(traj, '/home/schreibef98/projects/torchmd-net/datasets/trajectories/traj_AA_nsims_1000_n_timessteps_10mio.pt')
    


if __name__ == "__main__":
    main()
