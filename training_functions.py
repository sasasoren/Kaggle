import torch
import numpy as np
from tqdm import tqdm
import pandas as pd
from datasets import class_encoding
from flags import FLAGS
from scoring_functions import get_batch_top3score
import os


def train_function(model, optimizer, loss_fn, device, images_, labels_):
    data = images_.to(device)
    target = labels_.to(device)
    optimizer.zero_grad()
    # Forward pass
    output = model(data)
    loss = loss_fn(output, target)
    # Backprop and optimize
    loss.backward()
    optimizer.step()


def validation_function(model, val_loader, loss_fn, device):
    model.eval()
    score = 0
    total = 0
    with torch.no_grad():
        for i, sample in enumerate(val_loader):
            data = sample['image'].to(device)
            target = sample['label'].to(device)
            output = model(data)
            val_loss = loss_fn(output, target).item()
            # calculate mean average precision @3
            score += get_batch_top3score(target, output)
            total += target.size(0)
    return score * 100 / total, val_loss


def validation_summary(model, device, val_loader, num_classes):
    # Test the model
    model.eval()
    confusion_matrix = torch.zeros(num_classes, num_classes)
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (images_, labels_) in enumerate(val_loader):
            data = images_.to(device)
            target = labels_.to(device)
            output = model(data)
            _, predictions = torch.max(output, dim=1)
            total += target.size(0)
            correct += (predictions == target).sum().item()
            for t, p in zip(target.view(-1), predictions.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
        val_overall_accuracy = 100 * correct / total
        val_average_accuracy = np.mean((confusion_matrix.diag() / confusion_matrix.sum(1)).numpy() * 100)
        confusion_matrix_np = confusion_matrix.numpy()
        print('val oa: %f, val aa: %f' % (val_overall_accuracy, val_average_accuracy))
        print('confusion matrix:')
        print(confusion_matrix_np)

    return val_overall_accuracy, val_average_accuracy, confusion_matrix_np


def val_full(model, device, val_loader, nclasses):
    # Test the model
    model.eval()  # eval mode (batchnorm uses moving mean/variance instead of mini-batch mean/variance)
    confusion_matrix = torch.zeros(nclasses, nclasses)
    with torch.no_grad():
        correct = 0
        total = 0
        for i, (images_, labels_) in enumerate(val_loader):
            images = images_.to(device)
            labels = labels_.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
        val_oa = 100 * correct / total
        val_pca = (confusion_matrix.diag() / confusion_matrix.sum(1)).numpy() * 100
        val_aa = np.mean(val_pca)

        # for i, item in enumerate(label_values):
        #     print(item, val_pca[i])
        # print('val oa: {}%\tval aa: {}%'.format(val_oa, val_aa))
        # print('\n')
    return val_oa, val_aa, val_pca


def test_function(model, test_loader, device, length_test, batch_size, train_data_path, save_path):
    # Test the model
    model.eval()
    predictions = np.zeros((length_test, 4), dtype=object)
    # get the class encoding for submission purposes
    current_enc = class_encoding(train_data_path)
    enc_stripped = [key.replace(" ", "_") for key in list(current_enc.keys())]
    enc2word = np.vectorize(lambda x: enc_stripped[x])

    with torch.no_grad():
        for i, (images_, labels_) in tqdm(enumerate(test_loader)):
            data = images_.to(device)
            key_ids = labels_.reshape(data.size(0), 1)
            key_ids = key_ids.cpu().data.numpy().astype(object)
            output = model(data)

            # preds will be a tensor array of integer class predictions
            _, preds = torch.topk(output, k=3, dim=1)
            # enc2word takes the predictions and gives a numpy array of top3 word predictions
            preds = enc2word(preds.cpu().data.numpy()).astype(object)
            predictions[i * batch_size: i * batch_size + data.size(0)] = np.concatenate([key_ids, preds], axis=1)

    # merge predictions to create submission file
    guesses = (predictions[:, 1] + ' ' + predictions[:, 2] + ' ' + predictions[:, 3]).reshape(length_test, 1)
    submission = np.concatenate([predictions[:, 0].reshape(length_test, 1), guesses], axis=1)
    submission = pd.DataFrame(submission, columns=['key_id', 'word'])
    submission.to_csv(save_path, index=False)


