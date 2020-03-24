from glob import glob
from pathlib import Path
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset
from flags import FLAGS
import os
import random


def class_encoding(training_path):
    """
    takes the training csv files path and creates an int-encoding for all the class names
    """
    words = [Path(x).stem for x in glob(os.path.join(training_path, "*.csv"))]
    nums = np.arange(len(words))
    encoding = dict(zip(words, nums))

    return encoding


class QuickData(TensorDataset):
    """
    This class creates the training and validation sets
    """
    def __init__(self, data, train_data_path=None, num_data_per_class=None, transform=None):
        super(QuickData, self).__init__()
        self.data = pd.read_csv(data)
        self.train_data_path = train_data_path
        if num_data_per_class is not None:
            self.data = self.data.iloc[0: num_data_per_class]
        self.transform = transform

    def __getitem__(self, idx):
        # get encoding for numerical labeling
        encoding = class_encoding(self.train_data_path)

        # get samples based on mode: The first #val_samples_per_class instances go into
        # the validation set. The rest go to the training set
        sample = {'image': self.data['drawing'][idx], 'label': encoding[self.data['word'][idx]]}
        # apply pre-processing transforms
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample

    def __len__(self):
        return len(self.data)


class QuickTestData(TensorDataset):
    """
    This class creates the test set
    """
    def __init__(self, data, transform=None):
        super(QuickTestData, self).__init__()
        self.data = pd.read_csv(data)
        self.transform = transform

    def __getitem__(self, idx):
        sample = {'image': self.data['drawing'][idx], 'key_id': self.data['key_id'][idx]}
        # apply pre-processing transforms
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample

    def __len__(self):
        return len(self.data)

