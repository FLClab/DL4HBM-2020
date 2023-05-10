
import numpy
import torch
import os
import random
import tifffile

from torch.utils.data import Dataset, DataLoader

ACTIN = 0

class NormalizationLayer:
    """

    """
    def __init__(self, mode, stats=None):
        self.mode = mode
        self.stats = stats

        if not self.mode:
            self.mode = "default"

    def __call__(self, *args, **kwargs):
        """
        Implements the __call__ method of the `NormalizationLayer`
        """
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """
        Apply the normalization method to an input image
        """
        func = getattr(self, f"_{self.mode}_forward")
        return func(*args, **kwargs)

    def _default_forward(self, image, *args, **kwargs):
        """
        Does not apply any normalization to the image

        :param image: A `numpy.ndarray` of the image

        :returns : A `numpy.ndarray` of the image
        """
        return image

    def _minmax_forward(self, image, *args, **kwargs):
        """
        Apply a min-max normalization to the image using the provided image
        minimum and maximum

        :param image: A `numpy.ndarray` of the image
        :param image_min: A `numpy.ndarray` or `float` of the image minimum
        :param image_min: A `numpy.ndarray` or `float` of the image maximum

        :returns : A `numpy.ndarray` of the normalized image
        """
        image -= self.stats["image_min"]
        image /= (0.8 * (self.stats["image_max"] - self.stats["image_min"]))
        image = numpy.clip(image, 0, 1)
        return image

    def _quantile_forward(self, image, *args, **kwargs):
        """
        Apply a min-max normalization to the image using the {0.01, 0.99} quantiles
        of the image.

        :param image: A `numpy.ndarray` of the image

        :returns : A `numpy.ndarray` of the normalized image
        """
        m, M = numpy.quantile(image[ACTIN], [0.01, 0.99])
        image[ACTIN] = (image[ACTIN] - m) / (M - m)
        return image

class NumpyDataset(Dataset):
    def __init__(self, data, targets, validation, probability=0.5,
                    minmax=(0, 255)):
        """
        Instantiates the `NumpyDataset` class
        
        :param data: A `numpy.ndarray` of data with shape [N, H, W]
        :param targets: A `numpy.ndarray` of targets with shape [N, H, W]
        :param validation: A `bool` wheter the dataset is in validation mode 
        :param probability: A `float` of the probability to apply data augmentation 
        :param minmax: A `tuple` of normalization value
        """
        super(NumpyDataset, self).__init__()

        self.data = data
        self.targets = targets
        self.validation = validation
        self.probability = probability
        self.minmax = minmax

    def __getitem__(self, index):
        x = self.data[index].astype(numpy.float32)
        # x = (x - x.mean()) / x.std()
        x = (x - self.minmax[0]) / (self.minmax[1] - self.minmax[0])
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

        return x, y, index

    def __len__(self):
        return len(self.data)
        
class ImageDataset(Dataset):
    def __init__(self, data, targets, validation, probability=0.5,
                    normalization=None):
        """
        Instantiates the `NumpyDataset` class
        
        :param data: A `numpy.ndarray` of data with shape [N, H, W]
        :param targets: A `numpy.ndarray` of targets with shape [N, H, W]
        :param validation: A `bool` wheter the dataset is in validation mode 
        :param probability: A `float` of the probability to apply data augmentation 
        :param minmax: A `tuple` of normalization value
        """
        super(ImageDataset, self).__init__()

        self.data = data
        self.targets = targets
        self.validation = validation
        self.probability = probability
        
        self.normalization = normalization
        if isinstance(normalization, type(None)):
            self.normalization = NormalizationLayer(mode="default")

    def __getitem__(self, index):
        x = tifffile.imread(self.data[index])
        if x.min() == 2 ** 15:
            x = x - 2 ** 15

        if x.ndim == 2:
            x = x[numpy.newaxis]

        x = x.astype(numpy.float32)
        x = self.normalization(x)

        y = tifffile.imread(self.targets[index])
        if y.ndim == 2:
            y = y[numpy.newaxis]                
        y = y.astype(numpy.float32)

        x = torch.tensor(x, dtype=torch.float)
        y = torch.tensor(y, dtype=torch.long)

        return x, y, index

    def __len__(self):
        return len(self.data)
    
class CropDataset:
    def __init__(self, data, target, size=256, step=0.5, *args, **kwargs):

        self.data = data[[ACTIN]]
        self.target = target

        self.size = size
        self.step = step

        self.samples = []
        for j in range(0, data.shape[-2] - size, int(self.size * self.step)):
            for i in range(0, data.shape[-1] - size, int(self.size * self.step)):
                self.samples.append((j + size // 2, i + size // 2))        

    def __getitem__(self, index):

        j, i = self.samples[index]

        x = self.data[:, j - self.size // 2 : j + self.size // 2,
                         i - self.size // 2 : i + self.size // 2]
        
        y = self.target[:, j - self.size // 2 : j + self.size // 2,
                           i - self.size // 2 : i + self.size // 2]
        
        return x, y, (j, i)
    
    def __len__(self):
        return len(self.samples)

def get_loader(data, targets, batch_size=16, validation=False, minmax=(0, 255)):
    """
    Creates a torch.utils.data.DataLoader with the given data and targets

    :param data: A `numpy.ndarray` of the input data
    :param targets: A `numpy.ndarray` of the target data

    :returns : A `DataLoader`
    """
    dset = NumpyDataset(data, targets, validation=validation, minmax=minmax)
    return DataLoader(dset, batch_size=batch_size, shuffle=True, num_workers=0,
                        drop_last=False)

def get_image_loader(data, targets, batch_size=16, validation=False, normalization=None):
    """
    Creates a torch.utils.data.DataLoader with the given data and targets

    :param data: A `numpy.ndarray` of the input data
    :param targets: A `numpy.ndarray` of the target data

    :returns : A `DataLoader`
    """

    dset = ImageDataset(data, targets, validation=validation, normalization=normalization)
    return dset

def get_crop_loader(data, target, batch_size=16, *args, **kwargs):
    """
    Creates a torch.utils.data.DataLoader with the given data and targets

    :returns : A `DataLoader`
    """
    dset = CropDataset(data, target, *args, **kwargs)
    return DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=0,
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
