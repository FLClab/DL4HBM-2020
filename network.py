
import numpy
import glob
import os
import torch
import pickle
import h5py

from torch import nn
from collections import defaultdict
from tqdm.auto import tqdm, trange

import loader

class PredictionBuilder:
    """
    This class is used to create the final prediction from the predictions
    that are infered by the network. This class stores the predictions in an output
    array to avoid memory overflow with the method `add_predictions` and then
    computes the mean prediction of the overlap with the `return_prediction` method.

    :param shape: The shape of the image
    :param size: The size of the crops
    """
    def __init__(self, shape, size, num_classes=2):
        # Assign member variables
        self.shape = shape
        self.size = size

        # Creates the output arrays
        self.pred = numpy.zeros((num_classes, self.shape[0] + self.size, self.shape[1] + self.size), dtype=numpy.float32)
        self.pixels = numpy.zeros((self.shape[0] + self.size, self.shape[1] + self.size), dtype=numpy.float32)

    def add_predictions(self, predictions, positions):
        """
        Method to store the predictions in the output arrays. We suppose positions
        to be central on crops

        :param predictions: A `numpy.ndarray` of predictions with size (batch_size, features, H, W)
        :param positions: A `numpy.ndarray` of positions of crops with size (batch_size, 2)
        """
        # Verifies the shape of predictions
        if predictions.ndim != 4:
            # The feature channel has a high probabilty of missing
            predictions = predictions[:, numpy.newaxis, ...]
        for pred, (j, i) in zip(predictions, positions):

            # Stores the predictions in output arrays
            self.pred[:, j - self.size // 2 : j + self.size // 2, i - self.size // 2 : i + self.size // 2] += pred
            self.pixels[j - self.size // 2 : j + self.size // 2, i - self.size // 2 : i + self.size // 2] += 1

    def add_predictions_ji(self, prediction, j, i):
        """
        Method to store the predictions in the output array at the corresponding
        position. We suppose a central postion of the crop

        :param predictions: A `numpy.ndarray` of prediction with size (features, H, W)
        :param j: An `int` of the row position
        :param i: An `int` of the column position
        """
        # Verifies the shape of prediction
        if prediction.ndim != 3:
            prediction = prediction[numpy.newaxis, ...]

        # Crops image if necessary
        slc = (
            slice(None, None),
            slice(
                0 if j - self.size // 2 >= 0 else -1 * (j - self.size // 2),
                prediction.shape[-2] if j + self.size // 2 < self.pred.shape[-2] else self.pred.shape[-2] - (j + self.size // 2)
            ),
            slice(
                0 if i - self.size // 2 >= 0 else -1 * (i - self.size // 2),
                prediction.shape[-1] if i + self.size // 2 < self.pred.shape[-1] else self.pred.shape[-1] - (i + self.size // 2)
            )
        )
        pred = prediction[slc]

        # Stores prediction in output arrays
        self.pred[:, max(0, j - self.size // 2) : j + self.size // 2,
                     max(0, i - self.size // 2) : i + self.size // 2] += pred
        self.pixels[max(0, j - self.size // 2) : j + self.size // 2,
                    max(0, i - self.size // 2) : i + self.size // 2] += 1

    def return_prediction(self):
        """
        Method to return the final prediction.

        :returns : The average prediction map from the overlapping predictions
        """
        self.pixels[self.pixels == 0] += 1 # Avoids division by 0
        return (self.pred / self.pixels)[:, :self.shape[0], :self.shape[1]]

def load_ckpt(output_folder, network_name=None, model="UNet", filename="checkpoints.hdf5",
                verbose=True, epoch="best"):
    """
    Saves the current network state to a hdf5 file. The architecture of the hdf5
    file is
    hdf5file
        MICRANet
            network

    :param output_folder: A `str` to the output folder
    :param networks: A `dict` of network models
    :param filename: (optional) A `str` of the filename. Defaults to "checkpoints.hdf5"
    :param verbose: (optional) Wheter the function in verbose
    """
    if verbose:
        print("[----]     Loading network state")
    with h5py.File(os.path.join(output_folder, filename), "r") as file:
        if model in file:
            model_group = file[model]
            if not isinstance(network_name, str):
                network_name = list(model_group.keys())[0]
            state_dict = {k : torch.tensor(v[()]) for k, v in model_group[network_name].items()}
        else:
            if epoch == "max":
                epoch = str(sorted([int(key) for key in file.keys() if key != "best" ])[-1])
            main_group = file[epoch]

            networks = {}
            for key, values in main_group["network"].items():
                networks[key] = {k : torch.tensor(v[()]) for k, v in values.items()}
            state_dict = networks[list(networks.keys())[0]]
    return state_dict

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
    """
    def __init__(self, in_channels, out_channels, number_filter=4, depth=4):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

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

    def forward(self, x, use_sigmoid=False):
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

        if use_sigmoid:
            x = torch.sigmoid(x)

        return x

    def train_model(self, data, targets, train_idx, valid_idx, epochs=100, cuda=False,
                    lr=1e-4, batch_size=64, save_folder="./chkpt"):
        """
        Implements a train method for the UNet architecture

        :param data: A `numpy.ndarray` of data
        :param tragets: A `numpy.ndarray` of target data
        :param train_idx: A `list` of index to use for training
        :param valid_idx: A `list` of index to use for validation
        :param epochs: An `int` of the number of epochs to train the network 
        :param cuda: A `bool` wheter to send the model on GPU 
        :param lr: A `float` of the learning rate 
        :param batch_size: An `int` of the batch size 
        :param save_folder: A `str` path where to save the model 
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
            for X, y, index in tqdm(train_loader, leave=False, desc="Training"):

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
            for X, y, index in tqdm(valid_loader, leave=False, desc="Validation"):

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

        :param save_folder: A `str` of the path of the model 
        :param cuda: A `bool` wheter to load the model on the GPU
        :param epoch: An `int` of the epoch number to load. None results in best model
        """
        tmp = "" if isinstance(epoch, type(None)) else f"_{epoch}"
        # net_params = torch.load(os.path.join(save_folder, f"params{epoch}.net"), map_location=torch.device("cpu"))

        print(f"[%%%%] Loading pretrained model from: {save_folder}")
        model_path = os.path.join(save_folder, f"params{tmp}.net")
        if os.path.isfile(model_path):
            net_params = torch.load(model_path,
                                    map_location=torch.device("cpu"))
        else:
            net_params = load_ckpt(save_folder)
            # This is required since other models were trained with slightly different varaible name
            tmp = {}
            for key, values in net_params.items():
                key = key.replace("firstConvolution", "input_conv")
                key = key.replace("contractingPath", "contracting_path")
                key = key.replace("expandingPath", "expanding_path")
                key = key.replace("lastConv", "output_conv")
                tmp[key] = values
            net_params = tmp 

        # Loads state dict of the model and puts it in evaluation mode
        self.load_state_dict(net_params)
        self.eval()

        # Send on GPU
        if cuda:
            self = self.cuda()

    def predict(self, data, targets, idx=None, batch_size=64, cuda=False, minmax=(0, 255)):
        """
        Infers from the given data

        :param data: A `numpy.ndarray` of data
        :param tragets: A `numpy.ndarray` of target data
        :param idx: A `list` of index to use for prediction
        :param batch_size: An `int` of the batch size 
        :param cuda: A `bool` wheter to load the model on the GPU
        :param minmax: A `tuple` of normalization 
        
        :returns : A `torch.tensor` of the images
        :returns : A `torch.tensor` of the targets
        :returns : A `torch.tensor` of the predictions
        :returns : A `torch.tensor` of the indices 
        """
        if isinstance(idx, type(None)):
            idx = numpy.arange(len(data))
        predict_loader = loader.get_loader(data[idx], targets[idx], batch_size=batch_size, validation=True,
                                            minmax=minmax)

        self.eval()
        for X, y, index in tqdm(predict_loader, leave=False, desc="Prediction"):

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

            yield X, y, pred, index

            # To avoid memory leak
            del X, y, pred

    def predict_complete_image(self, data, targets, batch_size=64, cuda=False, normalization=None, size=256, step=0.5):
        """
        Infers from the given data

        :param data: A `numpy.ndarray` of data
        :param tragets: A `numpy.ndarray` of target data
        :param idx: A `list` of index to use for prediction
        :param batch_size: An `int` of the batch size 
        :param cuda: A `bool` wheter to load the model on the GPU
        :param minmax: A `tuple` of normalization 
        
        :returns : A `torch.tensor` of the images
        :returns : A `torch.tensor` of the targets
        :returns : A `torch.tensor` of the predictions
        :returns : A `torch.tensor` of the indices 
        """
        image_loader = loader.get_image_loader(data, targets, normalization=normalization)

        self.eval()
        for image, target, index in tqdm(image_loader, leave=False, desc="Images"):

            image = numpy.pad(image, ((0, 0), (size, size), (size, size)), mode="symmetric")
            target = numpy.pad(target, ((0, 0), (size, size), (size, size)), mode="symmetric")

            predict_loader = loader.get_crop_loader(image, target, size=size, step=step) 
            pb = PredictionBuilder(
                image.shape[-2:], size, self.out_channels
            )
            for X, y, positions in tqdm(predict_loader, leave=False, desc="Prediction"):

                # Verifies the shape of the data
                if X.ndim == 3:
                    X = X.unsqueeze(1)

                # Send on GPU
                if cuda:
                    X = X.cuda()

                # Prediction and loss computation
                pred = self.forward(X, use_sigmoid=True)

                if cuda:
                    X = X.cpu().data.numpy()
                    pred = pred.cpu().data.numpy()
                else:
                    X = X.data.numpy()
                    pred = pred.data.numpy()
                y = y.data.numpy()

                positions = numpy.array([
                    item.data.numpy() for item in positions
                ])
                pb.add_predictions(pred, positions.T)

                # To avoid memory leak
                del X, y, pred
            
            prediction = pb.return_prediction()

            image = image[:, size : -size, size:-size]
            target = target[:, size : -size, size:-size]
            prediction = prediction[:, size : -size, size:-size]

            yield image_loader.data[index], image, target, prediction


if __name__ == "__main__":

    epochs = 500
    cuda = True

    # Training polygonal bounding boxes
    data = numpy.load("raw_data/data_polygonal_bbox.npz")
    images, targets = data["images"], data["labels"]
    train_idx, valid_idx, test_idx = loader.get_idx(images)

    minimas = images[train_idx].min(axis=(1, 2))
    maximas = images[train_idx].max(axis=(1, 2))
    minmax = (numpy.mean(minimas), numpy.mean(maximas) + 3 * numpy.std(maximas))
    numpy.save("raw_data/minmax.npy", minmax)

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
    # data = numpy.load("raw_data/data_center-circle25.npz")
    # images, targets = data["images"], data["labels"]
    # train_idx, valid_idx, test_idx = loader.get_idx(images)
    # model = UNet(in_channels=1, out_channels=2)
    # model.train_model(images, targets, train_idx, valid_idx, epochs=epochs, cuda=cuda,
    #                     save_folder="/home-local/DL4HBM-2020/center-circle25")

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
