
import numpy
import torch

from torch.utils.data import Dataset, DataLoader

class NumpyDataset(Dataset):
    def __init__(self, data, targets, validation):
        super(NumpyDataset, self).__init__()

        self.data = data
        self.targets = targets
        self.validation = validation # import is transforms are used

    def __getitem__(self, index):
        x = self.data[index]
        x = (x - x.mean()) / x.std()
        y = self.targets[index]

        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)

        return x, y

    def __len__(self):
        return len(self.data)

def get_loader(data, targets, batch_size=16, validation=False):
    """
    Creates a torch.utils.data.DataLoader with the given data and targets

    :param data: A `numpy.ndarray` of the input data
    :param targets: A `numpy.ndarray` of the target data
    """
    dset = NumpyDataset(data, targets, validation=validation)
    return DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=2,
                        drop_last=True)
