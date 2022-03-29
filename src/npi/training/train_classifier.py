#           Pre-train content classifer to          #
#                 use in NPI training               #
#                                                   #
#          Fulda, Brown, Wingate, Robinson          #
#                       DRAGN                       #
#                    NPI Project                    #
#                       2020                        #

"""
Neural Programming Interfaces (NPI)

Overview:
    Classifier Code:
        - Includes functionality for loading pretrained classifiers
        - Supported neural models:
            - GPT2
"""

import argparse
import gc
import random as rand
import pickle as pkl
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from npi.dataset.npi_dataset import NPIDataSet

from npi.modeling_neural_program_interfaces import *
from npi.models.classifiers import StyleClassifier  # from transformers import *

def load_training_data(file_path, args, split_ratio=.25):  # with test-train split
    with open(file_path, 'rb') as datafile:
        dataset = pkl.load(datafile)
    rand.shuffle(dataset)
    max_train = len(dataset) - int(split_ratio * len(dataset))
    # This commented-out bit for if you want the validation data to be set aside as you train
    #   (recommended to just set it aside beforehand by using fewer than the total number of pkl's)
    # max_test = len(dataset[max_train:]) - int(split_ratio*len(dataset[max_train:]))
    return NPIDataSet(dataset[:max_train], pred_inds=args.pred_inds), NPIDataSet(dataset[max_train:],
                                                                 pred_inds=args.pred_inds)  # ClassDataSet(dataset[max_train:max_train+max_test]), ClassDataSet(dataset[max_train+max_test:])

def train_classifier(args):
    """
    DATA: [
        [(200x768x1), (2x1), [], 'gpt2', {}, ...]
        ...
    ]

    ( possible shapes if we are using small GPT-2 )
    """
    # initialize function vars
    # raise NotImplementedError("Interpretation protocol has not been written")
    save_file_path = args.save_file_path
    train_file_path_base = args.train_file_path_base
    train_file_path = train_file_path_base + ".pkl_0"

    device = args.device
    num_epochs = args.num_epochs
    total_epochs = args.continue_epoch + args.num_epochs if args.continue_epoch else num_epochs
    batch_size = args.batch_size
    test_freq = args.test_freq
    save_freq = args.save_freq

    try:
        # READ IN DATASET
        print(F"Loading Data from {train_file_path}")
        train_data, test_data = load_training_data(train_file_path, args)
        # Random selection of sentence-lists in each list (train_data, test_data, val_data)

        # Save the train_data, test_data, val_data to pkl files

        # CREATE TRAIN / TEST LOADERS
        train_loader = DataLoader(
            train_data, batch_size=batch_size, pin_memory=True)
        test_loader = DataLoader(
            test_data, batch_size=batch_size, pin_memory=True)

        # MODEL INITIALIZATION
        print("Creating Classifier Model", flush=True)
        classifier_model = StyleClassifier(train_data.n, train_data.m, 1).float()
        if args.continue_epoch:
            classifier_model.load_state_dict(torch.load(F"{save_file_path}Classifier_classification_network_epoch{args.continue_epoch}.bin",
                                                    map_location=torch.device('cpu')))
        classifier_model.cuda(device=device)

        print("Initializing class loss", flush=True)
        class_objective = torch.nn.BCELoss()
        class_optimizer = optim.Adam(
            classifier_model.parameters(), lr=args.class_lr)

        # PERFORM MODEL TRAINING

        epoch_losses = []
        batch_losses = []
        tests = []
        class_epoch_losses = []
        class_batch_losses = []
        class_tests = []
        false_class_tests = []

        pkl_num = 0
        total_pkls = args.num_pkls

        print("Training", flush=True)
        loop = tqdm(total=len(train_loader) * total_pkls * num_epochs,
                    mininterval=10, miniters=1000, position=0, leave=False)
        epochs = range(args.continue_epoch, total_epochs) if args.continue_epoch else range(num_epochs)
        for epoch in epochs:
            gc.collect()
            class_batch_losses = []

            # Get data from each pickle for each epoch
            for pkl_num in range(total_pkls):
                train_file_path = train_file_path_base + ".pkl_" + str(pkl_num)
                train_data, test_data = load_training_data(
                    train_file_path, args)
                train_loader = DataLoader(
                    train_data, batch_size=batch_size, pin_memory=True)
                test_loader = DataLoader(
                    test_data, batch_size=batch_size, pin_memory=True)
                for batch, (orig_activ, real_label, _target_label) in enumerate(train_loader):
                    # Catch any infs!!!!!
                    if -np.inf in orig_activ:
                        raise ValueError("Found inf in array")
                    # prepare the batch for model processing
                    orig_activ, real_label = orig_activ.cuda(device=device).float(), real_label.cuda(device=device).float()

                    # UPDATE CLASSIFIER WEIGHTS

                    for p in classifier_model.parameters():
                        p.requires_grad = True

                    # Find labels and loss
                    class_optimizer.zero_grad()
                    class_loss = None
                    labels = classifier_model(orig_activ)
                    class_loss = class_objective(labels.squeeze(),
                                                 real_label[:, 1].squeeze()) * 1e8  # gradient boosting constant
                    # backprop
                    class_loss.backward()
                    class_batch_losses.append(
                        class_loss.item())  # append to losses
                    class_optimizer.step()

                    if batch % test_freq == 0:
                        # perform npi_model testing
                        class_test_losses = []
                        # run testing loop
                        for test_batch, (test_x, test_truth, test_y) in enumerate(test_loader):
                            test_x, test_truth, test_y = test_x.cuda(device=device).float(
                            ), test_truth.cuda(device=device).float(), test_y.cuda(device=device).float()

                            # Test values
                            # lhat like labels above
                            lhat = classifier_model(test_x)
                            class_loss = class_objective(
                                lhat.squeeze(), test_truth[:, 1].squeeze()) * 1e8
                            class_test_losses.append(class_loss.item())

                        test_loss = sum(class_test_losses) / float(len(class_test_losses))
                        class_tests.append((epoch, test_loss))
                        training_loss = sum(class_batch_losses) / float(len(class_batch_losses))
                        # report current state to terminal
                        loop.set_description(
                            F'epoch: {epoch} loss: train={training_loss:.2f} test={test_loss:.2f}')
                        loop.update(test_freq)

            # record average loss for epoch
            if len(class_batch_losses) > 0:
                class_epoch_losses.append(
                    (sum(class_batch_losses) / float(len(class_batch_losses))))
            else:
                class_epoch_losses.append(np.nan)

            if epoch % save_freq == 0:
                # save the current version of the npi_mode, along with optimizer and loss record

                # save model
                save_classifier(save_file_path, classifier_model, class_optimizer, epoch)
                # save training info
                out_path = save_file_path + "N8_classification_loss_summaries.pkl"
                with open(out_path, 'wb') as outfile:
                    pkl.dump(
                        {"epoch_losses": class_epoch_losses,
                            "batch_losses": class_batch_losses, "tests": class_tests},
                        outfile)

        print("SAVING AFTER EPOCH ITERATIONS")
        # Save training info once again after training
        out_path = save_file_path + "N8_classification_loss_summaries.pkl"
        with open(out_path, 'wb') as outfile:
            pkl.dump({"epoch_losses": class_epoch_losses, "batch_losses": class_batch_losses, "tests": class_tests},
                     outfile)

        # save model after training
        save_classifier(save_file_path, classifier_model, class_optimizer, total_epochs)

        loop.close()
        print("Epoch train loss history == ", class_epoch_losses)
        gc.collect()

        return classifier_model

    except Exception as e:
        print(e)
        torch.cuda.empty_cache()
        raise

def save_classifier(save_file_path, classifier_model, class_optimizer, epoch):
    out_path = save_file_path + \
                    "{}_classification_network_epoch{}.bin".format(
                        'Classifier', epoch)
    torch.save(classifier_model.state_dict(), out_path)
                # save optimizer
    out_path = save_file_path + \
                    "{}_classification_optimizer_epoch{}.bin".format(
                        'Classifier', epoch)
    torch.save(class_optimizer, out_path)


if __name__ == "__main__":
    # main function: train_classifier()
    parser = argparse.ArgumentParser()
    parser.add_argument("--save-file-path",
                        default="classifiers/",
                        help="directory to save classifiers to")
    parser.add_argument("--train-file-path-base",
                        default="data/sentence_arrays",
                        help="path to data (standard file name witout pkl suffix, full or relative file path)")
    parser.add_argument("--num-epochs",
                        type=int,
                        default=70,
                        help="number of epochs to train for")
    parser.add_argument("--batch-size",
                        type=int,
                        default=5,
                        help="number of language model generated sequences to put into each training batch")
    parser.add_argument("--test-freq",
                        type=int,
                        default=5,
                        help="test every test_freq batches")
    parser.add_argument("--save-freq",
                        type=int,
                        default=5,
                        help="save the model during training every save_freq epochs")
    parser.add_argument("--num-pkls",
                        type=int,
                        # IMPORTANT NOTE: We only use 53 of the 57 available pickles (in default example)
                        default=53,
                        # The remaining 4 (~6.25%) are for testing
                        help="how many pickle of data we got?")
    parser.add_argument("--gpu-num",
                        type=int,
                        default=0,
                        help="which GPU to use")
    parser.add_argument("--perturbation-indices",
                        type=str,
                        default="5,11",
                        help="indices for layers to extract from language model activations: string of numbers separated by commas")
    parser.add_argument("--class-lr",
                        type=float,
                        default=1e-3,
                        help="model optimizer learning rate")

    args = parser.parse_args()

    args.device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")

    if torch.cuda.is_available():
        args.device = torch.device("cuda:{}".format(args.gpu_num))

    if not os.path.exists(args.save_file_path):
        os.mkdir(args.save_file_path)

    if args.save_file_path[-1] != '/':
        args.save_file_path = args.save_file_path + '/'

    # perturbation/prediction indices
    pis = [int(pi) for pi in args.perturbation_indices.split(',')]
    # construct file directory suffix
    dir_suffix = ""
    for pi in pis:
        dir_suffix = dir_suffix + "_" + str(pi)
    args.save_file_path = args.save_file_path + "layers" + dir_suffix + "/"
    if not os.path.exists(args.save_file_path):
        os.mkdir(args.save_file_path)
    # save pis as args
    args.pred_inds = pis

    # TRAIN
    mod = train_classifier(args=args)
