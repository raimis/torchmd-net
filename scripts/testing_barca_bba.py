#!/usr/bin/env python
# coding: utf-8
#!/bin/bash 
#SBATCH -J testing_barca_03 
#SBATCH -D /data/scratch/schreibef98/projects
#SBATCH -o testing_barca_03.%j.out 
#SBATCH --partition=gpu 
#SBATCH --nodes=1
#SBATCH --mem=8000M 
#SBATCH --time=19:00:00 
#SBATCH --mail-type=end 
#SBATCH --mail-user= franz.josef.schreiber@fu-berlin.de
# In[1]:


import numpy as np 
import torch 
import argparse
import sys


# In[2]:


sys.path.insert(0, '../')
from torchmdnet2.simulation_torchmdnetModels import Simulation_, PTSimulation_
from torchmdnet2.dataset.bba_inMemoryDataset import BBADataset
from torchmdnet2.simulation_utils import PT_temps

device = torch.device('cuda')


# In[3]:

def main():
    #path = '/home/mi/schreibef98/projects/torchmd-net/datasets/bba/'
    path = '/home/schreibef98/projects/torchmd-net/datasets/bba/'
    args = argparse.Namespace()
    args.coordinates=path+'/bba_zero_traj_no_box_n_replicas_20.xtc'
    args.cutoff=None
    #args.device='cuda:0'
    args.device='cuda'
    args.extended_system=None
    args.external = {'module': 'torchmdnet.calculators', 'embeddings': [6, 10, 4, 13, 1, 20, 4, 20, 2 ,19, 13, 3, 19, 9, 6, 20, 6, 21, 19, 5, 3, 22, 6, 20, 3, 20, 2, 19], 'file': '/home/schreibef98/projects/torchmd-net/models/prot_spec_bba_epoch=63-val_loss=736.2164-test_loss=21.5246.ckpt'}
    args.forcefield=path+'/ca_priors-dihedrals_general_2xweaker.yaml'
    args.forceterms=['Bonds', 'RepulsionCG', 'Dihedrals']
    args.exclusions = ('bonds')
    args.langevin_gamma=1
    args.langevin_temperature=350
    #args.log_dir='/home/mi/schreibef98/projects/torchmd-net/datasets/trajectories/test_barca_bba/'
    args.log_dir='/home/schreibef98/projects/torchmd-net/datasets/trajectories/test_barca_bba/'
    args.minimize=None
    args.output='output'
    args.output_period=1000
    args.precision='double'
    args.replicas=20
    args.rfa=False
    args.save_period=1000
    args.seed=1
    args.steps=10000000
    args.structure=None
    args.switch_dist=None
    args.temperature = 350
    args.timestep=3
    args.topology= path+'processed/bba.psf'
    
    
    # In[4]:
    
    
    #bba_dataset = BBADataset('/home/mi/schreibef98/projects/torchmd-net/datasets/bba/')
    bba_dataset = BBADataset('/home/schreibef98/projects/torchmd-net/datasets/bba/')
    
    # In[5]:
    
    T = PT_temps(350, 550, 4)
    #T = 350
    R = 8.314462
    e = (R*T)/4184
    betas = 1/e
    
    
    # In[6]:
    
    
    n_sims = 5-1
    n_timesteps = 3000000
    save_interval = 20
    export_interval = 2000
    exchange_interval = 2000
    
    
    # In[7]:
    
    
    ids = np.arange(0, len(bba_dataset),len(bba_dataset)//n_sims).tolist()
    init = bba_dataset[ids]
    initial_coords = torch.cat([init[i].pos.reshape((1,-1,3)) for i in range(len(init))], dim=0).to(device=device)
    initial_coords.requires_grad_()
    
    sim_embeddings = torch.cat([init[i].z.reshape((1,-1)) for i in range(len(init))], dim=0).to(device=device)
    
    # I did not load these correctly
    """
    # overwrite initial coords for this simulation
    initial_coords = np.load(path + '/initial_coords_cgl.npy')[:8]
    initial_coords = torch.from_numpy(initial_coords).to(device)
    initial_coords.requires_grad_()
    
    initial_coords = initial_coords.double()
    sim_embeddings = sim_embeddings.double()
    """
    # In[8]:
    # mass_scale = 418.4
    masses = list(12*np.ones(28))
    dt = 0.02045482949774598 * 5
    friction = 0.04888821
    print(sim_embeddings.dtype)
    sim = PTSimulation_(args, initial_coords, sim_embeddings, length=n_timesteps,
                     save_interval=save_interval, betas=betas,
                     save_potential=True, device=device, dt=dt, exchange_interval=exchange_interval,
                     log_interval=10000, log_type='write', filename='/home/schreibef98/projects/torchmd-net/datasets/trajectories/test_barca_bba/bba_03',
                     masses=masses, friction=friction, export_interval=export_interval)
    """
    sim = Simulation_(args, initial_coords, sim_embeddings, length=n_timesteps,
                     save_interval=save_interval, beta=betas,
                     save_potential=True, device=device, dt=dt,
                     log_interval=10000, log_type='write', filename='/home/mi/schreibef98/projects/torchmd-net/datasets/trajectories/test_barca_bba/bba_03',
                     masses=masses, friction=friction)
    """
    # In[ ]:
    
    
    traj = sim.simulate()
    
    
    # In[ ]:
    
    
    torch.save(traj, '/home/schreibef98/projects/torchmd-net/datasets/trajectories/test_barca_bba/traj_03.pt')


# In[ ]:


if __name__ == "__main__":
    main()


