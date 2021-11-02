#!/usr/bin/env python
# coding: utf-8
#!/bin/bash 
#SBATCH -J nequip_cgl_06 
#SBATCH -D /data/scratch/schreibef98/projects
#SBATCH -o nequip_cgl_06.%j.out 
#SBATCH --partition=gpu 
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --mem=32000M
#SBATCH --time=200:00:00
#SBATCH --mail-type=end 
#SBATCH --mail-user=franz.josef.schreiber@fu-berlin.de

import sys, os
import numpy as np
import torch
from tqdm import tqdm
from os.path import join
import yaml
from copy import deepcopy

sys.path.insert(0,'../')
# from torchmdnet2.dataset.data_module import  DataModule
from torchmdnet2.models import LNNP, SchNet, CGnet
from torchmdnet2.utils import make_splits
from torchmdnet2.simulation import Simulation
from torchmdnet2.dataset.chignolin import ChignolinDataset

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.accelerators import GPUAccelerator
from pytorch_lightning.plugins import NativeMixedPrecisionPlugin, DDPPlugin, SingleDevicePlugin



from torch.nn import Embedding, Sequential, Linear, ModuleList

#sys.path.insert(0,'/Users/iMac/git/e3nn/')
import e3nn
import e3nn.util.jit

#sys.path.insert(0,'/Users/iMac/git/nequip/')
from nequip.models import ForceModel
from nequip.utils import Config, dataset_from_config
from nequip.data import AtomicDataDict, AtomicData, DataLoader
from nequip.nn import RescaleOutput
from nequip.utils.test import assert_AtomicData_equivariant, set_irreps_debug

from torchmdnet2.models.nequip import PLModel, DataModule

from tqdm import tqdm

workdir = '/home/schreibef98/projects/torchmd-net/models/nequip_cgl_06'

config_fn = join(workdir,'config.yaml')

# checkpoint_fn = join(workdir,'epoch=49-val_loss=0.0000-test_loss=0.0000.ckpt')
checkpoint_fn = None

if __name__ == '__main__':
    cgl_dataset = ChignolinDataset('/home/schreibef98/projects/torchmd-net/datasets/chignolin_AA/')

    with open(config_fn, 'r') as f:
        config = yaml.safe_load(f)


    if checkpoint_fn is not None:
        plmodel = PLModel.load_from_checkpoint(
            checkpoint_path=checkpoint_fn,
            hparams_file=join(workdir,'hparams.yaml')
        )
    else:
        model = ForceModel(**config['model'])
        plmodel = PLModel(model, **config['optimizer'])
    
    data_list = []
    # stride 10 because of memory issues
    for i in tqdm(range(0,len(cgl_dataset),10)):
        ff = cgl_dataset[i]
        data_list.append(AtomicData.from_points(ff['pos'], config['model']['r_max'], **{'atomic_numbers' : ff['z'], 'forces' : ff['forces']}))
    data_list = tuple(data_list)

    dm = DataModule(data_list, **config['datamodule'])


    pl.seed_everything(config['seed'])

    checkpoint_callback = ModelCheckpoint(
        dirpath=config['trainer']['default_root_dir'],
        monitor='validation_loss',
        save_top_k=20, # -1 to save all
        every_n_epochs=config['save_interval'],
        filename="{epoch}-{val_loss:.4f}-{test_loss:.4f}"
    )
    early_stopping = EarlyStopping('validation_loss', patience=config['optimizer']['lr_patience'])

    tb_logger = pl.loggers.TensorBoardLogger(config['trainer']['default_root_dir'], name='tensorbord', version='')
    csv_logger = pl.loggers.CSVLogger(config['trainer']['default_root_dir'], name='', version='')

    accelerator = GPUAccelerator(
        precision_plugin=NativeMixedPrecisionPlugin(),
        training_type_plugin=SingleDevicePlugin(device=torch.device('cuda:0')),
    )

    trainer = pl.Trainer(
        accelerator='ddp',
        plugins=DDPPlugin(find_unused_parameters=False),
        auto_lr_find=False,
        checkpoint_callback=True,
        callbacks=[early_stopping, checkpoint_callback],
        logger=[tb_logger, csv_logger],
        log_every_n_steps=1,
        reload_dataloaders_every_epoch=False,
        **config['trainer'],
    )

    trainer.fit(plmodel, dm)
