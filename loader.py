
import numpy
import torch
import os
import random

from torch.utils.data import Dataset, DataLoader

class NumpyDataset(Dataset):
    def __init__(self, data, targets, validation, probability=0.5):
        super(NumpyDataset, self).__init__()

        self.data = data
        self.targets = targets
        self.validation = validation # import if transforms are used
        self.probability = probability

    def __getitem__(self, index):
        x = self.data[index].astype(numpy.float32)
        x = (x - x.mean()) / x.std()
        y = self.targets[index]

        # Apply small data augmentation
        if not self.validation:
            # intensity scale
            if random.random() < self.probability:
                scale = numpy.random.normal(0.75, 0.25, size=1)
                scale = numpy.clip(scale, 0.25, 1.25)
                x = x * scale

            # left right flip
            if random.random() < self.probability:
                x = numpy.fliplr(x).copy()
                y = numpy.fliplr(y).copy()

            # 90degree rotation
            if random.random() < self.probability:
                k = random.randint(1, 3)
                x = numpy.rot90(x, k=k).copy()
                y = numpy.rot90(y, k=k).copy()

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

    :returns : A `DataLoader`
    """
    dset = NumpyDataset(data, targets, validation=validation)
    return DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=2,
                        drop_last=False)

def get_idx(data, train_ratio=0.7, force_new_idx=False):
    """
    Gets the training, validation and testing indices. By default validation and testing
    are 50/50 split from 1 - train_ratio

    :param data: A `numpy.ndarray` of the input data
    :param train_ratio: The training ratio
    :param force_new_idx: Wheter to force the generation of new indices

    :returns : A `numpy.ndarray` of the training indices
    :returns : A `numpy.ndarray` of the validation indices
    :returns : A `numpy.ndarray` of the testing indices
    """
    if os.path.isfile("./raw_data/idx.npz") and (not force_new_idx):
        idx = numpy.load("./raw_data/idx.npz")
        return idx["train_idx"], idx["valid_idx"], idx["test_idx"]

    train_idx = numpy.random.choice(len(data), size=int(train_ratio * len(data)), replace=False)
    other_idx = numpy.setdiff1d(numpy.arange(len(data)), train_idx)
    numpy.random.shuffle(other_idx)
    valid_idx, test_idx = other_idx[:len(other_idx)//2],  other_idx[len(other_idx)//2:]

    # Save the new generated indices
    # numpy.savez("./raw_data/idx.npz",
    #             train_idx=train_idx, valid_idx=valid_idx, test_idx=test_idx)

    return train_idx, valid_idx, test_idx
