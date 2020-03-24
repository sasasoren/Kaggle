import os
from glob import glob
from tqdm import tqdm
# import h5py
import torch
import torch.nn as nn
import numpy as np
import resnet_mod
from flags import FLAGS
# from get_dataloaders import get_dataloaders
from training_functions import train_function, test_function, val_full
# from scoring_functions import get_batch_top3score
# import gc
from utils import get_training_dataloader, get_test_dataloader
from conf import settings


def main():
    # release gpu memory
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()
    # torch.cuda.ipc_collect()
    # get the list of csv training files
    # training_csv_files = glob(os.path.join(FLAGS.train_data_path, "*.csv"))

    # build training, validation, and test data loaders
    print(' Preparing the data!')
    # Fix the random seed for identical experiments
    # train_loader, test_loader, test_length = \
    #     get_dataloaders(training_csv_files[0: FLAGS.num_classes], FLAGS.test_data_path, FLAGS.num_data_per_class)

    # build test and train for cifar100
    train_loader = get_training_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        batch_size=FLAGS.batch_size,
    )

    test_loader = get_test_dataloader(
        settings.CIFAR100_TRAIN_MEAN,
        settings.CIFAR100_TRAIN_STD,
        batch_size=FLAGS.batch_size,
    )

    test_length = len(test_loader.dataset)

    kwargs = {
        "kernels_path_7": FLAGS.kernels_path_7,
        "kernels_path_3": FLAGS.kernels_path_3,
        "num_kernels_7": FLAGS.num_kernels_7,
        "num_kernels_3": FLAGS.num_kernels_3,
        "num_classes": FLAGS.num_classes
    }
    loss_function = nn.CrossEntropyLoss()

    for conv_model in ["Conv2d", "Conv2dRF"]:
        for resnet_arch in ["resnet18", "resnet34", "resnet50"]:
            name = resnet_arch + '_' + conv_model

            kwargs["conv_model"] = conv_model
            model = getattr(resnet_mod, resnet_arch)(**kwargs)
            optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.learning_rate)

            if torch.cuda.device_count() > 1:
                model = nn.DataParallel(model)
            print(' Started training!')

            for run_id in range(FLAGS.num_runs):
                torch.cuda.empty_cache()
                # model.weights_init()
                if torch.cuda.device_count() > 1:
                    model.module.weights_init()
                else:
                    model.weights_init()
                model.to(device)

                for epoch in range(FLAGS.num_epochs):
                    model.train()
                    for batch_idx, (images_, labels_) in tqdm(enumerate(train_loader)):
                        # print("batch_{}: {}".format(batch_idx, torch.cuda.memory_cached()/1e6))
                        train_function(model, optimizer, loss_function, device, images_, labels_)
                        if (batch_idx + 1) % FLAGS.log_interval == 0 or (batch_idx + 1) == len(train_loader):

                            # test_function(
                            #     model, test_loader, device, test_length, FLAGS.batch_size,
                            #     FLAGS.train_data_path,
                            #     os.path.join(FLAGS.save_path, "submissions",
                            #                  name+'_r={}_e={}_idx={}.csv'.format(run_id, epoch+1, batch_idx+1)))
                            model.train()  # reset back to train mode

                # Table for output of validation
                val_output = np.zeros((FLAGS.num_runs, 102))
                val_oa, val_aa, val_pca = val_full(model, device, test_loader, 100)

                val_output[run_id, 0] = val_oa
                val_output[run_id, 1] = val_aa
                val_output[run_id, 2:] = val_pca

                # saving the model
                torch.save({'model_state_dict': model.state_dict()},
                           os.path.join(
                               FLAGS.save_path,
                               "models",
                               "{}.pt".format(name+'_r={}'.format(run_id+1))))

            np.save(os.path.join(FLAGS.save_path, "validation_{}.npy".format(name)), val_output)


if __name__ == '__main__':
    main()
