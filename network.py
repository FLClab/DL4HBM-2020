
import numpy
import glob
import os
import torch
import pickle

from torch import nn
from collections import defaultdict
from tqdm.auto import tqdm, trange

import loader

class DoubleConvolver(nn.Module):
    """
    Class for the double convolution in the contracting path. The kernel size is
    set to 3x3 and a padding of 1 is enforced to avoid lost of pixels. The convolution
    is followed by a batch normalization and relu.

    :param in_channels: Number of channels in the input tensor
    :param out_channels: Number of channels produced by the convolution
    """
    def __init__(self, in_channels, out_channels):
        super(DoubleConvolver, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Contracter(nn.Module):
    """
    Class for the contraction path. Max pooling of the input tensor is
    followed by the double convolution.

    :param in_channels: Number of channels in the input tensor
    :param out_channels: Number of channels produced by the convolution
    """
    def __init__(self, in_channels, out_channels):
        super(Contracter, self).__init__()
        self.conv = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConvolver(in_channels=in_channels, out_channels=out_channels)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class Expander(nn.Module):
    """
    Class for the expansion path. Upsampling with a kernel size of 2 and stride 2
    is performed and followed by a double convolution following the concatenation
    of the skipping link information from higher layers.

    :param in_channels: Number of channels in the input tensor
    :param out_channels: Number of channels produced by the convolution
    """
    def __init__(self, in_channels, out_channels):
        super(Expander, self).__init__()
        self.expand = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConvolver(in_channels=in_channels, out_channels=out_channels)

    def center_crop(self, links, target_size):
        _, _, links_height, links_width = links.size()
        diff_x = (links_height - target_size[0]) // 2
        diff_y = (links_width - target_size[1]) // 2
        return links[:, :, diff_y:(diff_y + target_size[0]), diff_x:(diff_x + target_size[1])]

    def forward(self, x, link):
        x = self.expand(x)
        crop = self.center_crop(link, x.size()[2 : ])
        concat = torch.cat([x, crop], 1)
        x = self.conv(concat)
        return x


class UNet(nn.Module):
    """
    Class for creating the UNet architecture. A first double convolution is performed
    on the input tensor then the contracting path is created with a given depth and
    a preset number of filters. The number of filter is doubled at every step.

    :param in_channels: Number of channels in the input tensor
    :param out_channels: Number of output channels (i.e. number of classes)
    :param number_filter: Number of filters in the first layer (2 ** number_filter)
    :param depth: Depth of the network
    :param size: The size of the crops that are fed to the network
    """
    def __init__(self, in_channels, out_channels, number_filter=4, depth=4, size=244):
        super(UNet, self).__init__()

        self.input_conv = DoubleConvolver(in_channels=in_channels, out_channels=2**number_filter)
        self.contracting_path = nn.ModuleList()
        for i in range(depth - 1):
            self.contracting_path.append(
                Contracter(in_channels=2**(number_filter + i), out_channels=2**(number_filter + i + 1))
            )
        self.expanding_path = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.expanding_path.append(
                Expander(in_channels=2**(number_filter + i + 1), out_channels=2**(number_filter + i))
            )
        self.output_conv = nn.Conv2d(in_channels=2**number_filter, out_channels=out_channels, kernel_size=1)

    def forward(self, x):
        links = [] # keeps track of the links
        x = self.input_conv(x)
        links.append(x)

        # Contracting path
        for i, contracting in enumerate(self.contracting_path):
            x = contracting(x)
            if i != len(self.contracting_path) - 1:
                links.append(x)

        # Expanding path
        for i, expanding in enumerate(self.expanding_path):
            x = expanding(x, links[- i - 1])
        x = self.output_conv(x)

        return x

    def train_model(self, data, targets, train_idx, valid_idx, epochs=100, cuda=False,
                    lr=1e-4, batch_size=64, save_folder="./chkpt"):
        """
        Implements a train method for the UNet architecture

        :param data: A `numpy.ndarray` of data
        :param tragets: A `numpy.ndarray` of target data
        :param train_idx: A list of index to use for training
        :param valid_idx: A list of index to use for validation
        """
        # Creation of save_folder
        self.save_folder = save_folder
        if not os.path.isdir(self.save_folder):
            os.makedirs(self.save_folder, exist_ok=False)

        # Send model on GPU
        if cuda:
            self = self.cuda()

        # Creates the loaders
        train_loader = loader.get_loader(data[train_idx], targets[train_idx], batch_size=batch_size)
        valid_loader = loader.get_loader(data[valid_idx], targets[valid_idx], batch_size=batch_size, validation=True)

        # Creation of the optimizer
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # To keep track of the statistics
        self.stats = defaultdict(list)

        # Creation of the criterion
        criterion = nn.CrossEntropyLoss()
        if cuda:
            criterion = criterion.cuda()

        # To keep track of the network generalizing the most
        min_valid_loss = numpy.inf

        for epoch in range(epochs):

            print("[----] Starting epoch {}/{}".format(epoch + 1, epochs))

            # Keep track of the loss of train and test
            statLossTrain, statLossTest = [], []

            # training
            self.train()
            for X, y in tqdm(train_loader, leave=False, desc="Training"):

                # Verifies the shape of the data
                if X.ndim == 3:
                    X = X.unsqueeze(1)

                # Send model on GPU
                if cuda:
                    X = X.cuda()
                    y = y.cuda()

                # New batch we reset the optimizer
                optimizer.zero_grad()

                # Prediction and loss computation
                pred = self.forward(X)
                loss = criterion(pred, y)

                # Keeping track of statistics
                statLossTrain.append(loss.cpu().data.numpy())

                # Back-propagation and optimizer step
                loss.backward()
                optimizer.step()

                # To avoid memory leak
                del X, y, pred, loss

            # validation
            self.eval()
            for X, y in tqdm(valid_loader, leave=False, desc="Validation"):

                # Verifies the shape of the data
                if X.ndim == 3:
                    X = X.unsqueeze(1)

                # Send on GPU
                if cuda:
                    X = X.cuda()
                    y = y.cuda()

                # Prediction and loss computation
                pred = self.forward(X)
                loss = criterion(pred, y)

                # Keeping track of statistics
                statLossTest.append(loss.cpu().data.numpy())

                # To avoid memory leak
                del X, y, pred, loss

            # Aggregate stats
            for key, func in zip(("trainMean", "trainMed", "trainMin", "trainStd"),
                                 (numpy.mean, numpy.median, numpy.min, numpy.std)):
                self.stats[key].append(func(statLossTrain))
            for key, func in zip(("testMean", "testMed", "testMin", "testStd"),
                                 (numpy.mean, numpy.median, numpy.min, numpy.std)):
                self.stats[key].append(func(statLossTest))

            if self.stats["testMean"][-1] < min_valid_loss:
                print("[!!!!]   New best model.")
                self.save_model(optimizer, epoch=epoch, best_model=True)
                min_valid_loss = self.stats["testMean"][-1]

            if epoch % 10 == 0:
                self.save_model(optimizer, epoch=epoch, best_model=False)

    def save_model(self, optimizer, epoch, best_model):
        """
        Saves the current epoch model

        :param optimizer: The current state of the optimizer
        :param epoch: The current epoch
        :param best_model: Wheter the model is currently the best model
        """
        print("[----]   Saving current network state")
        epoch = "" if best_model else f"_{epoch}"
        torch.save(self.state_dict(), os.path.join(self.save_folder, f"params{epoch}.net"))
        torch.save(optimizer.state_dict(), os.path.join(self.save_folder, f"optimizer{epoch}.data"))
        pickle.dump(self.stats, open(os.path.join(self.save_folder, f"statsCkpt{epoch}.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)

    def load_model(self, save_folder="./chkpt", cuda=False, epoch=None):
        """
        Loads a previous model

        :param save_folder: The path to the model
        """
        epoch = "" if isinstance(epoch, type(None)) else f"_{epoch}"
        net_params = torch.load(os.path.join(save_folder, f"params{epoch}.net"))
        # optim_params = torch.load(os.path.join(save_folder, f"optimizer{epoch}.net"))

        # Loads state dict of the model and puts it in evaluation mode
        self.load_state_dict(net_params)
        self.eval()

        # Send on GPU
        if cuda:
            self = self.cuda()

    def predict(self, data, targets, idx=None, batch_size=64, cuda=False):
        """
        Infers from the given data

        :param data: A `numpy.ndarray` of data
        :param tragets: A `numpy.ndarray` of target data
        :param idx: A list of index to use for training
        """
        if isinstance(idx, type(None)):
            idx = numpy.arange(len(data))
        predict_loader = loader.get_loader(data[idx], targets[idx], batch_size=batch_size, validation=True)

        self.eval()
        for X, y in tqdm(predict_loader, leave=False, desc="Prediction"):

            # Verifies the shape of the data
            if X.ndim == 3:
                X = X.unsqueeze(1)

            # Send on GPU
            if cuda:
                X = X.cuda()

            # Prediction and loss computation
            pred = self.forward(X)

            if cuda:
                X = X.cpu().data.numpy()
                pred = pred.cpu().data.numpy()
            else:
                X = X.data.numpy()
                pred = pred.data.numpy()
            y = y.data.numpy()

            yield X, y, pred

            # To avoid memory leak
            del X, y, pred


if __name__ == "__main__":

    epochs = 250
    cuda = True

    # Training polygonal bounding boxes
    data = numpy.load("raw_data/data.npz")
    images, targets = data["images"], data["labels"]
    train_idx, valid_idx, test_idx = loader.get_idx(images)
    model = UNet(in_channels=1, out_channels=2)
    model.train_model(images, targets, train_idx, valid_idx, epochs=epochs, cuda=cuda,
                        save_folder="/home-local/DL4HBM-2020/polygonal_bbox")

    # Training bounding boxes
    data = numpy.load("raw_data/data_bbox.npz")
    images, targets = data["images"], data["labels"]
    train_idx, valid_idx, test_idx = loader.get_idx(images)
    model = UNet(in_channels=1, out_channels=2)
    model.train_model(images, targets, train_idx, valid_idx, epochs=epochs, cuda=cuda,
                        save_folder="/home-local/DL4HBM-2020/bbox")

    # Training center-circle25
    data = numpy.load("raw_data/data_center-circle25.npz")
    images, targets = data["images"], data["labels"]
    train_idx, valid_idx, test_idx = loader.get_idx(images)
    model = UNet(in_channels=1, out_channels=2)
    model.train_model(images, targets, train_idx, valid_idx, epochs=epochs, cuda=cuda,
                        save_folder="/home-local/DL4HBM-2020/center-circle25")

    # Training dilation of polygonal bounding boxes
    data = numpy.load("raw_data/data_dilation5.npz")
    images, targets = data["images"], data["labels"]
    train_idx, valid_idx, test_idx = loader.get_idx(images)
    model = UNet(in_channels=1, out_channels=2)
    model.train_model(images, targets, train_idx, valid_idx, epochs=epochs, cuda=cuda,
                        save_folder="/home-local/DL4HBM-2020/dilation5")

    # model.load_model(cuda=True)
    #
    # for pred in model.predict(images, targets, valid_idx, cuda=True):
    #     print(pred.shape)
