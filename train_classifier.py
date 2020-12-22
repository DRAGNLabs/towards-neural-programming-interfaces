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

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from .modeling_neural_program_interfaces import *  # from transformers import *

torch.manual_seed(1)

model_to_total_layers = {'gpt2-medium': 25, 'gpt2': 13}
model_to_m = {'gpt2-medium': 1024, 'gpt2': 768}
model_to_use = 'gpt2'  # This can be changed to gpt2-medium to use the medium GPT-2
num_total_layers = model_to_total_layers[model_to_use]
seq_len = 10  # This is a hyperparam that can be changed
num_iters = 10  # This is a hyperparam that can be changed
CONST_M = model_to_m[model_to_use]  # 768


# NPI Code Block ################################################################################################

def extract_needed_layers(array, pis):
    """
    Accepts array of size (1300, 768, 1) (as an example)
        (really the size is ((num_total_layers*seq_len*num_iters), m, 1)
        and prediction indices
    Returns array of size (n, m, 1)

    * accepts and returns a numpy array
    """

    # Calculate CONST_N from pis
    CONST_N = len(pis) * seq_len * num_iters  # could be 200

    # First note that the pis may have some negative numbers in them - we fix this
    for i in range(len(pis)):
        if pis[i] < 0:
            pis[i] = num_total_layers + pis[i]  # Should be good now

    original_length = array.shape[0]
    assert original_length == num_total_layers * seq_len * num_iters  # Should contain all layers originally (recommended)
    all_layers_len_for_one_iter = original_length / num_iters
    # We construct a mask for the large array
    mask = np.array([False for _ in range(original_length)])
    for i in range(original_length):
        position = i % all_layers_len_for_one_iter
        corresponding_pi = position // seq_len
        if corresponding_pi in pis:
            mask[i] = True

    # Now we should have it
    extracted_layer_array = array[mask]
    assert extracted_layer_array.shape == (CONST_N, CONST_M, 1)
    return extracted_layer_array


def load_training_data(file_path, args, split_ratio=.25):  # with test-train split
    with open(file_path, 'rb') as datafile:
        dataset = pkl.load(datafile)
    rand.shuffle(dataset)
    max_train = len(dataset) - int(split_ratio * len(dataset))
    # This commented-out bit for if you want the validation data to be set aside as you train
    #   (recommended to just set it aside beforehand by using fewer than the total number of pkl's)
    # max_test = len(dataset[max_train:]) - int(split_ratio*len(dataset[max_train:]))
    return ClassDataSet(dataset[:max_train], args), ClassDataSet(dataset[max_train:],
                                                                 args)  # ClassDataSet(dataset[max_train:max_train+max_test]), ClassDataSet(dataset[max_train+max_test:])


class ClassDataSet(Dataset):
    def __init__(self, dataset, args):
        """
        Assumes input dataset is of the form:
            [[language_model_activations, 
              activations_classification, 
              target_classification (no longer used), 
              language_model_type, 
              meta_data,
              ...
            ],  
            ...]
        With objects of the following types:
            language_model_activations : nxmx1 ndarray representing flattened activation sequences (required)
            activations_classification : small ndarray representing the sentiment/content classification of the original activations (required)
            target_classification : (not required)
            language_model_type : str naming the language model being controlled (optional - assumed None)
            meta_data : dict recording desired metadata (required for NPI training later)
        """
        self.ORIG_ACTIV_INDEX = 0
        self.ORIG_LABEL_INDEX = 1
        self.TARG_LABEL_INDEX = 2
        self.LANG_MODEL_INDEX = 3
        self.META_DATA_INDEX = 4

        self.dataset = dataset
        # NPI DATA SET LOOP
        for i in range(len(self.dataset)):
            self.dataset[i][self.ORIG_ACTIV_INDEX] = torch.from_numpy(
                extract_needed_layers(self.dataset[i][self.ORIG_ACTIV_INDEX], pis=args.pred_inds))
            self.dataset[i][self.ORIG_LABEL_INDEX] = torch.from_numpy(
                self.dataset[i][self.ORIG_LABEL_INDEX])  # .type(torch.FloatTensor)
            self.dataset[i][self.TARG_LABEL_INDEX] = torch.tensor([])  # .type(torch.FloatTensor)

    def __getitem__(self, i):
        acts = self.dataset[i][self.ORIG_ACTIV_INDEX]  # .type(torch.FloatTensor)
        truth = self.dataset[i][self.ORIG_LABEL_INDEX]  # .type(torch.FloatTensor)
        targ = self.dataset[i][self.TARG_LABEL_INDEX]  # .type(torch.FloatTensor)
        return acts, truth, targ

    def __len__(self):
        return len(self.dataset)


# NPI Classifier code ------------------------------------------------------------------------------------------------------


class Classifier(nn.Module):  # classifies NPI outputs
    def __init__(self, input_activs_shape, input_label_shape):
        """
        input_activs_shape: tuple of (b, n, m, 1)
            b is the number of batches
            n x m x 1 slices contain the elements of the original activations, flattened into a 2D array
        target_label: tuple of (b, 1, m, 1)
            the desired label for the predicted activations, as passed into the NPI network
        """
        super(Classifier, self).__init__()

        print("Classifier INIT", flush=True)
        self.b = input_activs_shape[0]  # batch size
        self.n = input_activs_shape[1]
        self.m = input_activs_shape[2]
        self.k = input_activs_shape[3]
        self.N = self.n * self.m

        fact1 = 2 ** 4
        fact2 = 2 ** 5
        fact3 = 2 ** 6

        print("Defining classifier model", flush=True)

        self.model = nn.Sequential(
            nn.Linear(self.n * self.m * self.k, self.n // fact1),
            nn.ReLU(),
            nn.Linear(self.n // fact1, self.n // fact2),
            nn.ReLU(),
            nn.Linear(self.n // fact2, self.n // fact3),
            nn.ReLU(),
            nn.Linear(self.n // fact3, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x.view(-1, self.n * self.m * self.k))
        pass


def train_classifier(args):
    """
    DATA: [
        [(200x768x1), (2x1), [], 'gpt2', {}, ...]
        ...
    ]

    ( possible shapes if we are using small GPT-2 )
    """

    # compute CONST_N
    CONST_N = len(args.pred_inds) * seq_len * num_iters  # could be 200
    if -1 in args.pred_inds or num_total_layers - 1 in args.pred_inds:
        CONST_N = CONST_N - num_iters

    # initialize function vars
    # raise NotImplementedError("Interpretation protocol has not been written")
    save_file_path = args.save_file_path
    train_file_path_base = args.train_file_path_base
    train_file_path = train_file_path_base + ".pkl_0"

    device = args.device
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    test_freq = args.test_freq
    save_freq = args.save_freq

    try:
        # READ IN DATASET
        print("Loading Data")
        train_data, test_data = load_training_data(train_file_path, args)
        # Random selection of sentence-lists in each list (train_data, test_data, val_data)

        # Save the train_data, test_data, val_data to pkl files

        # CREATE TRAIN / TEST LOADERS
        train_loader = DataLoader(train_data, batch_size=batch_size, pin_memory=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, pin_memory=True)

        # MODEL INITIALIZATION
        print("Creating Classifier Model", flush=True)
        input_activs_shape = (batch_size, CONST_N, CONST_M, 1)
        input_truth_shape = (batch_size, 4)
        print("Act shapes ==", input_activs_shape, flush=True)
        print("label shapes ==", input_truth_shape, flush=True)
        classifier_model = None
        classifier_model = Classifier(input_activs_shape, input_truth_shape).float()
        classifier_model.cuda()

        print("Initializing class loss", flush=True)
        class_objective = torch.nn.BCELoss()
        class_optimizer = optim.Adam(classifier_model.parameters(), lr=args.class_lr)

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

        print("Defining loop report", flush=True)
        loop = tqdm(total=len(train_loader) * total_pkls, position=0, leave=False)

        print("Training", flush=True)

        for epoch in range(num_epochs):
            gc.collect()
            class_batch_losses = []

            # Get data from each pickle for each epoch 
            for pkl_num in range(total_pkls):
                train_file_path = train_file_path_base + ".pkl_" + str(pkl_num)
                train_data, test_data = load_training_data(train_file_path, args)
                train_loader = DataLoader(train_data, batch_size=batch_size, pin_memory=True)
                test_loader = DataLoader(test_data, batch_size=batch_size, pin_memory=True)

                for batch, (orig_activ, real_label, _target_label) in enumerate(train_loader):
                    # Catch any infs!!!!!
                    if -np.inf in orig_activ:
                        raise ValueError("Found inf in array")
                    # prepare the batch for model processing
                    orig_activ, real_label = orig_activ.cuda().float(), real_label.cuda().float()

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
                    class_batch_losses.append(class_loss.item())  # append to losses
                    class_optimizer.step()

                    # report current state to terminal
                    loop.set_description('epoch:{}'.format(epoch))
                    loop.update(1)

                    if batch % test_freq == 0:
                        # perform npi_model testing
                        class_test_losses = []
                        # run testing loop
                        for test_batch, (test_x, test_truth, test_y) in enumerate(test_loader):
                            test_x, test_truth, test_y = test_x.cuda().float(), test_truth.cuda().float(), test_y.cuda().float()

                            # Test values
                            lhat = classifier_model(test_x)  # lhat like labels above
                            class_loss = class_objective(lhat.squeeze(), test_truth[:, 1].squeeze()) * 1e8
                            class_test_losses.append(class_loss.item())

                        class_tests.append((epoch, (sum(class_test_losses) / float(len(class_test_losses)))))

            # record average loss for epoch
            if len(class_batch_losses) > 0:
                class_epoch_losses.append((sum(class_batch_losses) / float(len(class_batch_losses))))
            else:
                class_epoch_losses.append(np.nan)

            if epoch % save_freq == 0:
                # save the current version of the npi_mode, along with optimizer and loss record

                # save model
                out_path = save_file_path + "{}_classification_network_epoch{}.bin".format('Classifier', epoch)
                torch.save(classifier_model, out_path)
                # save optimizer
                out_path = save_file_path + "{}_classification_optimizer_epoch{}.bin".format('Classifier', epoch)
                torch.save(class_optimizer, out_path)
                # save training info
                out_path = save_file_path + "N8_classification_loss_summaries.pkl"
                with open(out_path, 'wb') as outfile:
                    pkl.dump(
                        {"epoch_losses": class_epoch_losses, "batch_losses": class_batch_losses, "tests": class_tests},
                        outfile)

        print("SAVING AFTER EPOCH ITERATIONS")
        # Save training info once again after training
        out_path = save_file_path + "N8_classification_loss_summaries.pkl"
        with open(out_path, 'wb') as outfile:
            pkl.dump({"epoch_losses": class_epoch_losses, "batch_losses": class_batch_losses, "tests": class_tests},
                     outfile)

        # save model after training
        out_path = save_file_path + "{}_classification_network_epoch{}.bin".format('Classifier', num_epochs)
        torch.save(classifier_model, out_path)
        # save optimizer after training
        out_path = save_file_path + "{}_classification_optimizer_epoch{}.bin".format('Classifier', num_epochs)
        torch.save(class_optimizer, out_path)

        loop.close()
        print("Epoch loss history == ", epoch_losses)
        gc.collect()

        return classifier_model

    except Exception as e:
        print(e)
        torch.cuda.empty_cache()
        raise


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
                        default=53,  # IMPORTANT NOTE: We only use 53 of the 57 available pickles (in default example)
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

    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
