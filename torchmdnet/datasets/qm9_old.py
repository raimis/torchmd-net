from torchmdnet_old.nnp.atoms import AtomsData
import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Data


class QM9_old(InMemoryDataset):
    def __init__(self, root, transform=None, dataset_arg=None):
        super(QM9_old, self).__init__()
        self.dset_arg = dataset_arg
        self.data = AtomsData(root, available_properties=None, load_only=[dataset_arg])

    def get_atomref(self, max_z=100):
        atomref = torch.from_numpy(self.data.get_atomref(self.dset_arg)[self.dset_arg])
        if atomref is None:
            return None
        if atomref.size(0) != max_z:
            tmp = torch.zeros(max_z).unsqueeze(1)
            idx = min(max_z, atomref.size(0))
            tmp[:idx] = atomref[:idx]
            return tmp
        return atomref

    def get(self, idx):
        x = self.data[idx]
        return Data(
            z=x["_atomic_numbers"], pos=x["_positions"], y=x[self.dset_arg][None]
        )

    def len(self):
        return len(self.data)
