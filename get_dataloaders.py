import os
from tqdm import tqdm
import torch
from multiprocessing import cpu_count
from torch.utils.data import DataLoader, ConcatDataset, random_split
from torchvision.transforms import Compose, ToTensor, Normalize

from flags import FLAGS
from datasets import QuickData, QuickTestData
from draw_functions import Draw


def get_dataloaders(train_csv_paths, test, num_data_per_class=None):
    """
    :param train_csv_paths: list of train_csv_paths csv file paths
    :param test: the test csv file path
    :return: the train and test data loaders
    """

    kwargs = {'num_workers': cpu_count(), 'pin_memory': True} if torch.cuda.is_available() else {}
    # define your transform
    transform = Compose([Draw(), ToTensor(), Normalize([0], [255])])

    # initialize empty lists for the individual csv data sets
    train_data_sets = [None] * FLAGS.num_classes
    # populate the lists and weights
    for i in tqdm(range(FLAGS.num_classes)):
        train_data_sets[i] = QuickData(
            train_csv_paths[i], os.path.dirname(train_csv_paths[0]), num_data_per_class, transform)
    # get the training, and test datasets
    train_data = ConcatDataset(train_data_sets)
    test_data = QuickTestData(test, transform)

    # get training, validation, and test data loaders
    train_loader = DataLoader(train_data, batch_size=FLAGS.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_data, batch_size=FLAGS.batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader, test_data.__len__()


