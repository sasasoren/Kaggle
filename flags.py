import torch
import argparse
from glob import glob
import os
from pathlib import Path
from sys import argv

# Argument parser for CLI interaction
parser = argparse.ArgumentParser(description="Quick Draw")
#################################################################
# Data set options
parser.add_argument('--train_data_path', type=str, default=None)
parser.add_argument('--test_data_path', type=str, default=None)
parser.add_argument('--num_data_per_class', type=int, default=600, help="2e14")
parser.add_argument('--num_classes', type=int, default=100)
parser.add_argument('--save_path', type=str, default=None)
parser.add_argument('--input_channels', type=int, default=3)
parser.add_argument('--image_shape', type=int, default=(32, 32))
# parser.add_argument('--num_classes', type=int,
#                            default=len(list(glob(parser.parse_known_args()[0].training_data + '*.csv'))))
parser.add_argument('--kernels_path_7', type=str, default=None)
parser.add_argument('--kernels_path_3', type=str, default=None)
parser.add_argument('--num_kernels_7', type=int, default=49)
parser.add_argument('--num_kernels_3', type=int, default=9)
# parser.add_argument('--resnet_arch', type=str, default="resnet18", help="use resnet18 or resnet34 or resnet50")
# parser.add_argument('--conv_model', type=str, default="Conv2d", help="use Conv2d or Conv2dRF")
#################################################################
# model training options
batch_size = 128 if torch.cuda.device_count() == 0 else torch.cuda.device_count() * 128
parser.add_argument('--batch_size', type=int, default=batch_size)
parser.add_argument('--num_epochs', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=.001)
# how ofter do you want to print training and validation accuracies/losses
parser.add_argument('--log_interval', type=int, default=1000)
# parser.add_argument('--run_id', type=str, default="1")
parser.add_argument('--num_runs', type=int, default=10)
#################################################################
FLAGS, _ = parser.parse_known_args()

