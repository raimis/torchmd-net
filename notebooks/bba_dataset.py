#!/usr/bin/env python
# coding: utf-8
#SBATCH -J bba_dataset
#SBATCH -D /data/scratch/schreibef98/projects
#SBATCH -o bba_dataset.%j.out 
#SBATCH --partition=gpu 
#SBATCH --nodes=1 
#SBATCH --gres=gpu:1 
#SBATCH --mem=4000M 
#SBATCH --time=20:00:00 
#SBATCH --mail-type=end 
#SBATCH --mail-user= franz.josef.schreiber@fu-berlin.de
# In[1]:


import sys
sys.path.insert(0, '../')
from torchmdnet2.dataset.bba_inMemoryDataset import BBADataset


# In[ ]:


dataset = BBADataset("/home/schreibef98/projects/torchmd-net2/datasets/bba/")


# In[ ]:




