import torch

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

import pytorch_lightning as pl

import sys, os

#sys.path.insert(0,'/Users/iMac/git/nequip/')
from nequip.data import AtomicData, DataLoader
sys.path.insert(0,'../')
from torchmdnet2.utils import make_splits

def dump(fn, obj):
    import pickle
    with open(fn, 'wb') as f:
        pickle.dump(obj, f)
def load(fn):
    import pickle
    with open(fn, 'rb') as f:
        data = pickle.load(f)
    return data

def dump_datalist(fn, data_list):
    torch.save(tuple(data_list), fn)

def load_datalist(fn):
    data_list = torch.load(fn)
    return data_list

class PLModel(pl.LightningModule):
    def __init__(self, model,
    lr:float =1e-4, weight_decay:float=0, lr_factor:float=0.8,
    lr_patience:int=10, lr_min:float=1e-7, target_name='forces',
    lr_warmup_steps:int=0,
    test_interval:int=1,
    derivative:bool = True,
):

        super(PLModel, self).__init__()
        
        self.save_hyperparameters()
        self.model = model
        self.losses = None
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_factor = lr_factor
        self.lr_patience = lr_patience
        self.lr_min = lr_min
        self.target_name = target_name
        self.lr_warmup_steps = lr_warmup_steps
        self.test_interval = test_interval
        self.derivative = derivative

    def configure_optimizers(self):
        optimizer = AdamW(self.model.parameters(),
        lr=self.lr, weight_decay=self.weight_decay)
        scheduler = ReduceLROnPlateau(
            optimizer,
            'min',
            factor=self.lr_factor,
            patience=self.lr_patience,
            min_lr=self.lr_min
        )
        lr_scheduler = {'scheduler': scheduler,
                        'monitor': 'validation_loss',
                        'interval': 'epoch',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

    def training_step(self, data, batch_idx):
        loss = self.step(data, 'training')
        return loss


    def validation_step(self, data, batch_idx):
        loss = self.step(data, 'validation')
        return loss

    def test_step(self, data, batch_idx):
        loss = self.step(data, 'test')
        return loss


    def step(self, data, stage):
        with torch.set_grad_enabled(stage == 'train' or self.derivative):
            pred = self.model(AtomicData.to_AtomicDataDict(data))

        loss = 0
        facs = {'forces': 1.}
        for k,fac in facs.items():
            loss += fac * (pred[k] - data[k]).pow(2).mean()

        # Add sync_dist=True to sync logging across all GPU workers
        self.log(f'{stage}_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)

        return loss

class DataModule(pl.LightningDataModule):
    def __init__(self, dataset,
                 log_dir: str,
                val_ratio: float = 0.1,
                test_ratio: float = 0.1,
                splits: str = None, batch_size: int = 512,
                inference_batch_size: int = 64, num_workers: int = 1,
                train_stride: int = 1) -> None:
        super(DataModule, self).__init__()
        self.dataset_root = os.path.join(log_dir,'dataset.pt')
        dump_datalist(self.dataset_root, dataset)

        self.batch_size = batch_size
        self.inference_batch_size = inference_batch_size
        self.num_workers = num_workers
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.log_dir = log_dir
        self.splits = splits
        self.train_stride = train_stride
        self.splits_fn = os.path.join(self.log_dir, 'splits.npz')

    def prepare_data(self):
        # make sure the dataset is downloaded
        dataset = load_datalist(self.dataset_root)
        # setup the train/test/val split
        idx_train, idx_val, idx_test = make_splits(
            len(dataset),
            self.val_ratio,
            self.test_ratio,
            filename=self.splits_fn,
            splits=self.splits,
        )

    def setup(self, stage = None):
        dataset = load_datalist(self.dataset_root)
        self.idx_train, self.idx_val, self.idx_test = make_splits(
            len(dataset),
            self.val_ratio,
            self.test_ratio,
            splits=self.splits_fn,
        )

        self.train_dataset = tuple([dataset[ii] for ii in self.idx_train[::self.train_stride]])
        self.val_dataset = tuple([dataset[ii] for ii in self.idx_val])
        self.test_dataset = tuple([dataset[ii] for ii in self.idx_test])

    def train_dataloader(self):
        return self._get_dataloader(self.train_dataset, 'train')

    def val_dataloader(self):
        return self._get_dataloader(self.val_dataset, 'val')

    def test_dataloader(self):
        return self._get_dataloader(self.test_dataset, 'test')

    def _get_dataloader(self, dataset, stage):
        if stage == 'train':
            batch_size = self.batch_size
            shuffle = True
        elif stage in ['val', 'test']:
            batch_size = self.inference_batch_size
            shuffle = False

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=True
        )
