import os
import glob
import numpy as np
from sklearn.utils import shuffle
import torch
from torch.utils.data import Dataset, DataLoader


def load_train(train_path, classes):
    samples = []
    labels = []
    sample_names = []
    cls = []

    for fields in classes:
        index = classes.index(fields)
        path = os.path.join(train_path, fields, '*.npy')
        files = glob.glob(path)
        for fl in files:
            iq_samples = np.load(fl)

            real = np.real(iq_samples)
            imag = np.imag(iq_samples)

            iq_samples = []
            for i in range(0, np.ma.count(real) - 212):
                iq_samples.append(real[i])
                iq_samples.append(imag[i])
            iq_samples = np.reshape(iq_samples, (-1, 128, 2))

            samples.append(iq_samples)

            label = np.zeros(len(classes))
            label[index] = 1.0
            labels.append(label)
            flbase = os.path.basename(fl)
            sample_names.append(flbase)
            cls.append(fields)

    samples = np.array(samples)
    labels = np.array(labels)
    sample_names = np.array(sample_names)
    cls = np.array(cls)

    return samples, labels, sample_names, cls


class DataSet(Dataset):
    def __init__(self, images, labels, img_names, cls):
        self.images = images
        self.labels = labels
        self.img_names = img_names
        self.cls = cls

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index], self.img_names[index], self.cls[index]


def get_data_loaders(train_path, classes, validation_size, train_batch_size, valid_batch_size):
    class DataLoaders(object):
        pass

    data_loaders = DataLoaders()

    images, labels, img_names, cls = load_train(train_path, classes)
    images, labels, img_names, cls = shuffle(images, labels, img_names, cls)

    if isinstance(validation_size, float):
        validation_size = int(validation_size * images.shape[0])

    validation_images = images[:validation_size]
    validation_labels = labels[:validation_size]
    validation_img_names = img_names[:validation_size]
    validation_cls = cls[:validation_size]

    train_images = images[validation_size:]
    train_labels = labels[validation_size:]
    train_img_names = img_names[validation_size:]
    train_cls = cls[validation_size:]

    train_dataset = DataSet(train_images, train_labels, train_img_names, train_cls)
    valid_dataset = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)

    data_loaders.train = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True)
    data_loaders.valid = DataLoader(valid_dataset, batch_size=valid_batch_size, shuffle=False)

    return data_loaders
