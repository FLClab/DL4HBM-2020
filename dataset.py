
import numpy
import os
import warnings

from skimage import draw, io, morphology, measure
from matplotlib import pyplot

def load_poly(path):
    """
    Loads the poly.txt file

    :param path: The path of the file to open

    :returns : A list of every line in the file
    """
    with open(path, "r") as f:
        data = f.readlines()
    return [x.strip() for x in data]

def read_poly(path, shape):
    """
    Reads and creates the labels from the poly.txt files

    :param path: The path to the poly.txt file
    :param shape: The shape of the image
    :returns : A numpy array of the labels

    NOTE. 0 is uncertain rings
          1 is clear rings
    """
    label = numpy.zeros(shape)
    data = load_poly(path)
    for row in data:
        l = int(row[0:1])
        if l in [0, 1]:
            coordinates = eval(row[2:])
            r, c = [], []
            for coord in coordinates:
                r.append(int(coord[1]))
                c.append(int(coord[0]))
            rr, cc = draw.polygon(r, c, shape=shape)
            label[rr, cc] = 1
    return label

def extract_rawdata(path):
    """
    Extracts the raw data from the folder and save it to a numpy array

    :param path: A `str` of to the folder path
    """
    images, labels = [], []
    for dirpath, dirnames, filenames in os.walk(path):
        if os.path.basename(dirpath) == "STED":
            image_names = list(filter(lambda name : name.endswith(".tiff"), filenames))
            polys_names = ["{}.polys.txt".format(image_name.split(".")[0]) for image_name in image_names]

            for image_name, polys_name in zip(image_names, polys_names):
                image = io.imread(os.path.join(dirpath, image_name))
                label = read_poly(os.path.join(dirpath, polys_name), shape=image.shape)

                images.append(image)
                labels.append(label)
    images, labels = map(numpy.array, (images, labels))
    # numpy.savez_compressed(os.path.join(path, "data.npz"),
    #                         images=images, labels=labels)

class LabelModifier():
    def __init__(self, method=None):
        # Verifies if attribute exists, else defaults in hungarian
        if isinstance(method, type(None)):
            self.apply_method = lambda l : l
        else:
            try:
                self.apply_method = getattr(self, f"_apply_{method}")
            except AttributeError:
                warnings.warn(f"The chosen method `{method}` does not exist. Defaults in `None`.", category=UserWarning)
                self.apply_method = lambda l : l

    def apply(self, label, **kwargs):
        return self.apply_method(label, **kwargs)

    def _apply_dilation(self, label, **kwargs):
        selem = morphology.square(**kwargs)
        return morphology.binary_dilation(label, selem=selem)

    def _apply_bbox(self, label, **kwargs):
        labeled_img = measure.label(label)
        regionprops = measure.regionprops(labeled_img)
        output_mask = numpy.zeros_like(label)
        for regionprop in regionprops:
            output_mask[regionprop.slice] = 1
        return output_mask

    def _apply_center_circle(self, label, **kwargs):
        labeled_img = measure.label(label)
        regionprops = measure.regionprops(labeled_img)
        output_mask = numpy.zeros_like(label)
        for regionprop in regionprops:
            rr, cc = draw.disk(regionprop.centroid, **kwargs, shape=label.shape)
            output_mask[rr, cc] = 1
        return output_mask


if __name__ == "__main__":

    path = "./raw_data"
    # extract_rawdata(path)
    data = numpy.load("./raw_data/data.npz")
    images, labels = data["images"], data["labels"]

    modified_labels = []
    dilation = 5
    for image, label in zip(images, labels):
        labelMod = LabelModifier(method="dilation")
        label_mod = labelMod.apply(label, width=dilation)

        modified_labels.append(label_mod)
    numpy.savez_compressed("./raw_data/data_dilation{}.npz".format(dilation),
        images=images, labels=numpy.array(labels)
    )

    modified_labels = []
    for image, label in zip(images, labels):
        labelMod = LabelModifier(method="bbox")
        label_mod = labelMod.apply(label)

        modified_labels.append(label_mod)
    numpy.savez_compressed("./raw_data/data_bbox.npz",
        images=images, labels=numpy.array(labels)
    )

    modified_labels = []
    radius = 25
    for image, label in zip(images, labels):
        labelMod = LabelModifier(method="center_circle")
        label_mod = labelMod.apply(label, radius=radius)

        modified_labels.append(label_mod)
    numpy.savez_compressed("./raw_data/data_center-circle{}.npz".format(radius),
        images=images, labels=numpy.array(labels)
    )
