#!/usr/bin/env python
# coding: utf-8
#!/bin/bash 
#SBATCH -J simulate_nequip 
#SBATCH -D /data/scratch/schreibef98/projects
#SBATCH -o simulate_nequip.%j.out 
#SBATCH --partition=gpu 
#SBATCH --nodes=1
#SBATCH --mem=8000M 
#SBATCH --time=60:00:00 
#SBATCH --mail-type=end 
#SBATCH --mail-user= franz.josef.schreiber@fu-berlin.de
# In[1]:


import numpy as np 
import torch 
import argparse
import sys
from os.path import join
import yaml

# In[2]:


sys.path.insert(0, '../')
from torchmdnet2.simulation_nequip import Simulation, PTSimulation
from torchmdnet2.dataset.chignolin import ChignolinDataset
from torchmdnet2.simulation_utils import PT_temps
from torchmdnet2.models import CGnet
from torchmdnet2.models.nequip import PLModel
from nequip.models import ForceModel

device = torch.device('cuda')


# In[3]:

def main():

    
    
    # In[4]:
    cgl_dataset = ChignolinDataset('/home/mi/schreibef98/projects/torchmd-net/datasets/chignolin_AA/')
    
    # load model
    workdir = '/home/mi/schreibef98/projects/torchmd-net/models/nequip_cgl_06/'
    model_file = 'epoch=269-val_loss=0.0000-test_loss=0.0000.ckpt'
    
    checkpoint_fn = join(workdir, model_file)
    config_fn = join(workdir,'config.yaml')
    with open(config_fn, 'r') as f:
        config = yaml.safe_load(f)
    plmodel = PLModel(ForceModel(**config['model']), **config['optimizer'])
    model = plmodel.load_from_checkpoint(
            checkpoint_path=checkpoint_fn).model
    model.to(device)
    baseline_model = cgl_dataset.get_baseline_model(n_beads=10)
    chignolin_net = CGnet(model, baseline_model).eval()
    chignolin_net.to(device)
    #T = PT_temps(350, 500, 4)
    T = 350
    R = 8.314462
    e = (R*T)/4184
    betas = 1/e
    
    
    # In[6]:
    
    
    n_sims = 20-1
    n_timesteps = 1000000
    save_interval = 100
    export_interval = 20000
    exchange_interval = 10000
    
    
    # In[7]:
    # generate initial coordinates
    
    ids = np.arange(0, len(cgl_dataset),len(cgl_dataset)//n_sims).tolist()
    init = cgl_dataset[ids]
    initial_coords = torch.cat([init[i].pos.reshape((1,-1,3)) for i in range(len(init))], dim=0).to(device=device)
    initial_coords.requires_grad_()
    
    sim_embeddings = torch.cat([init[i].z.reshape((1,-1)) for i in range(len(init))], dim=0).to(device=device)

    # In[8]:
    # mass_scale = 418.4
    masses = list(12*np.ones(10))
    dt = 0.02045482949774598 * 5
    friction = 0.04888821
    """
    sim = PTSimulation(model, initial_coords, sim_embeddings, length=n_timesteps,
                     save_interval=save_interval, betas=betas,
                     save_potential=True, device=device, dt=dt, exchange_interval=exchange_interval,
                     log_interval=10000, log_type='write', filename='/home/mi/schreibef98/projects/torchmd-net/datasets/trajectories/test_barca_bba/bba_01_continued_01',
                     masses=masses, friction=friction, export_interval=export_interval)
    """
    sim = Simulation(chignolin_net, initial_coords, sim_embeddings, length=n_timesteps,
                     save_interval=save_interval, beta=betas,
                     save_potential=True, device=device, dt=dt,
                     log_interval=10000, log_type='write', filename='/home/mi/schreibef98/projects/torchmd-net/datasets/trajectories/nequip_06/nequip_06_01',
                     masses=masses, friction=friction, export_interval=export_interval, batch_size=n_sims+1)
    
    # In[ ]:
    
    
    traj = sim.simulate()
    
    
    # In[ ]:
    
    
    torch.save(traj, '/home/mi/schreibef98/projects/torchmd-net/datasets/trajectories/nequip_06/traj_nequip_06.pt')


# In[ ]:


if __name__ == "__main__":
    main()


