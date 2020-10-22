from __future__ import absolute_import, division, print_function, unicode_literals
"""
Neural Program Interfaces (NPI's) Draft 1

Overview:
    Classifier Code:
        - Includes functionality for either training in-tandem with NPI or not
        - Includes functionality for loading pretrained classifiers
    NPI Code:
        - Includes functionality for:
            - Convolutional NPI
            - Inputting desired class label vector
    Style Transfer Inspired Adversarial Loss
    Functionality for controlling various network activations:
        - Supported neural models:
            - GPT2
    Functionality for interpretting NPI outputs:
        - Not part of the NPI class, allows for reshaping generated 'controlled' 
          activations and running them through a given neural model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)

from modeling_neural_program_interfaces import * # from transformers import *

from torch.utils.data import Dataset, DataLoader
# try:
#   import cPickle as pkl
# except:
import pickle as pkl
from torch.autograd import Variable

from tqdm import trange
from tqdm import tqdm
import gc
import random as rand

import argparse
import numpy as np
import os
import copy as cp
import pdb

#HAS_INFS = False
#if HAS_INFS:
#    BAD_ROWS = [29, 59, 89, 119, 149, 179, 209, 239, 269, 299, 329, 359, 389, 419, 449]
#    # Construct MASK 
#    MASK = np.array([True for _ in range(450)])
#    for br in BAD_ROWS:
#        MASK[br] = False
#else:
#    MASK = np.array([True for _ in range(435)])

#PRED_INDS = [7,10,11]
model_to_total_layers = {'gpt2-medium':25,'gpt2':13}
model_to_m = {'gpt2-medium':1024,'gpt2':768}
model_to_use = 'gpt2'
num_total_layers = model_to_total_layers[model_to_use]
seq_len = 10
num_iters = 10
#CONST_N = len(PRED_INDS) * seq_len * num_iters # 200
CONST_M = model_to_m[model_to_use] # 768
#if -1 in PRED_INDS or num_total_layers-1 in PRED_INDS:
#    CONST_N = CONST_N - num_iters

# NPI Code Block ################################################################################################
"""
NPI Network Draft 4
"""
def extract_needed_layers(array,pis):
    """
    Accepts array of size (1290, 768, 1) (as an examples)
        (really the size is ((num_total_layers*seq_len*num_iters - num_iters), m, 1)
        and prediction indices
    Returns array of size (n, m, 1)

    * accepts and returns a numpy array
    """
    
    # Calculate CONST_N from pis
    CONST_N = len(pis) * seq_len * num_iters # 200
    if -1 in pis or num_total_layers-1 in pis:
        CONST_N = CONST_N - num_iters

    # First note that the pis may have some negative numbers in them - we fix this
    for i in range(len(pis)):
        if pis[i] < 0:
            pis[i] = num_total_layers + pis[i] # Should be good now
    
    original_length = array.shape[0]
    assert original_length == num_total_layers*seq_len*num_iters #- num_iters
    all_layers_len_for_one_iter = original_length/num_iters
    # We construct a mask for the large array
    mask = np.array([False for _ in range(original_length)])
    for i in range(original_length):
        position = i % all_layers_len_for_one_iter
        corresponding_pi = position // seq_len
        if corresponding_pi in pis:
            mask[i] = True
    
    # Now we should have it
    extracted_layer_array = array[mask]
    #pdb.set_trace()
    assert extracted_layer_array.shape == (CONST_N, CONST_M, 1)
    return extracted_layer_array

def load_training_data(file_path, args, split_ratio=.25): # with test-train split
    with open(file_path, 'rb') as datafile:
        dataset = pkl.load(datafile)
    rand.shuffle(dataset)
    max_train = len(dataset) - int(split_ratio*len(dataset))
    #max_test = len(dataset[max_train:]) - int(split_ratio*len(dataset[max_train:]))
    # print("calling dataset constructors")
    return NPIDataSet(dataset[:max_train],args), NPIDataSet(dataset[max_train:],args) #NPIDataSet(dataset[max_train:max_train+max_test]), NPIDataSet(dataset[max_train+max_test:])


class NPIDataSet(Dataset):
    def __init__(self, dataset, args):
        """
        Assumes input dataset is of the form:
            [[language_model_activations, 
              activations_classification, 
              target_classification, 
              language_model_type, 
              meta_data, 
            ],  
            ...]
        With objects of the following types:
            language_model_activations : nxmx1 ndarray representing flattened activation sequences (required)
            activations_classification : 1xmx1 ndarray representing the sentiment/content classification of the original activations (optional - assumed None)
            target_classification : 1xmx1 ndarray representing the desired sentiment/content classification of generated activations (required)
            language_model_type : str naming the language model being controlled (optional - assumed None)
            meta_data : dict recording desired metadata (optional - assumed None)
        """
        self.ORIG_ACTIV_INDEX = 0
        self.ORIG_LABEL_INDEX = 1
        self.TARG_LABEL_INDEX = 2
        self.LANG_MODEL_INDEX = 3
        self.META_DATA_INDEX = 4

        self.dataset = dataset
        # print("NPI DATASET LOOP")
        #print("made it?",flush=True)
        for i in range(len(self.dataset)):
            # print("self.dataset[i][self.ORIG_ACTIV_INDEX].shape == ", self.dataset[i][self.ORIG_ACTIV_INDEX].shape)
            # print("self.dataset[i][self.TARG_LABEL_INDEX].shape == ", self.dataset[i][self.TARG_LABEL_INDEX].shape)
            self.dataset[i][self.ORIG_ACTIV_INDEX] = torch.from_numpy(extract_needed_layers(self.dataset[i][self.ORIG_ACTIV_INDEX], pis=args.pred_inds))
            self.dataset[i][self.ORIG_LABEL_INDEX] = torch.from_numpy(self.dataset[i][self.ORIG_LABEL_INDEX])#.type(torch.FloatTensor)
            self.dataset[i][self.TARG_LABEL_INDEX] = torch.from_numpy(np.array(self.dataset[i][self.TARG_LABEL_INDEX]))#.type(torch.FloatTensor)
            # print("self.dataset[i][self.ORIG_ACTIV_INDEX].size == ", self.dataset[i][self.ORIG_ACTIV_INDEX].size())
            # print("self.dataset[i][self.TARG_LABEL_INDEX].size == ", self.dataset[i][self.TARG_LABEL_INDEX].size())
        #print("made it.:",flush=True)
        pass

    def __getitem__(self, i):
        acts = self.dataset[i][self.ORIG_ACTIV_INDEX]# = self.dataset[i][self.ORIG_ACTIV_INDEX].type(torch.FloatTensor)
        truth = self.dataset[i][self.ORIG_LABEL_INDEX]# = self.dataset[i][self.ORIG_LABEL_INDEX].type(torch.FloatTensor)
        targ = self.dataset[i][self.TARG_LABEL_INDEX]# = self.dataset[i][self.TARG_LABEL_INDEX].type(torch.FloatTensor)
        return acts, truth, targ
 
    def __len__(self):
        return len(self.dataset)

# NPI Neural Model Code -------------------------------------------------------------------------------
class UpConvLayer(nn.Module):
    def __init__(self, a, b):
        super(UpConvLayer, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(a, b, (3, 3), padding=(1, 1)), #, init_mode=init_mode),#250
                                 nn.BatchNorm2d(b),
                                 nn.ReLU(),
                                 nn.Conv2d(b, b, (3, 3), padding=(1, 1)), #, init_mode=init_mode),#90
                                 nn.BatchNorm2d(b),
                                 nn.ReLU(), 
                                 nn.ConvTranspose2d(b, b//2, (2, 2), stride=2, padding=(0, 0)), #, outputpadding=(1, 1))
                                 #nn.BatchNorm2d(b//2),
                                )
 
    def forward(self, input):
        return self.net(input)#.squeeze(2).squeeze(2)
    
class DownConvLayer(nn.Module):
    def __init__(self, a, b):
        super(DownConvLayer, self).__init__()
        self.net = nn.Sequential(nn.MaxPool2d((2, 2), stride=2),
                                 nn.Conv2d(a, b, (3, 3), padding=(1, 1)), #, init_mode=init_mode),#250
                                 nn.BatchNorm2d(b),
                                 nn.ReLU(),
                                 nn.Conv2d(b, b, (3, 3), padding=(1, 1)), #, init_mode=init_mode),#90
                                 nn.BatchNorm2d(b),
                                 nn.ReLU(),
                                )
 
    def forward(self, input):
        return self.net(input)#.squeeze(2).squeeze(2)

class ConvLayer(nn.Module):
    def __init__(self, a, b):
        super(ConvLayer, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(a, b, (3, 3), padding=(1, 1)), #, init_mode=init_mode),#250
                                 nn.BatchNorm2d(b),
                                 nn.ReLU(),
                                 nn.Conv2d(b, b, (3, 3), padding=(1, 1)), #, init_mode=init_mode),#90
                                 nn.BatchNorm2d(b),
                                 nn.ReLU()
                                )
 
    def forward(self, input):
        return self.net(input)#.squeeze(2).squeeze(2)

class ConvNPI(nn.Module): # Convolutional Version
    def __init__(self, input_activs_shape, input_targ_shape):
        """
        input_activs_shape: tuple of (b, n, m, 1)
            b is the number of batches
            n x m x 1 slices contain the elements of the original activations, flattened into a 2D array
        target_label: tuple of (b, 1, m, 1)
            the desired label for the predicted activations, as passed into the NPI network
        """
        super(ConvNPI, self).__init__()
        print("NPI INITIALIZATION")
        self.b = input_activs_shape[0]
        self.n = input_activs_shape[1]
        self.m = input_activs_shape[2]
        self.k = input_activs_shape[3]
        self.n = self.n*self.m

        print("Setting Scaling Factors")
        fact1 = 2**4
        fact2 = 2**5
        fact3 = 2**6

        print("Defining first npi layer")
        self.first_linear = nn.Sequential(nn.Linear((self.n+1)*self.m*self.k, self.n//fact1),  
                                          nn.ReLU(), 
                                          nn.Linear(self.n//fact1, self.n//fact2), 
                                          nn.ReLU(), 
                                          nn.Linear(self.n//fact2, self.n//fact3), 
                                          nn.ReLU(), 
                                          nn.Linear(self.n//fact3, self.n//fact2), 
                                          nn.ReLU(), 
                                          nn.Linear(self.n//fact2, self.n//fact1), 
                                          nn.ReLU(), 
                                          nn.Linear(self.n//fact1, self.n*self.m*self.k), 
                                          )
        # print("Defining down npi layer")
        # self.conv1 = ConvLayer(self.k, self.k*fact1)
        # self.down_conv1 = DownConvLayer(self.k*fact1, self.k*fact2)
        # self.down_conv2 = DownConvLayer(self.k*fact2, self.k*fact3)
        # print("Defining bottom npi layer")
        # # DownConvLayer(self.k, self.k*(2**4)), DownConvLayer(self.k, self.k*(2**5)), 
        # self.first_conv_trans = nn.ConvTranspose2d(self.k*fact3, self.k*fact2, (2, 2), stride=2, padding=(0, 0))
        # print("Defining up npi layer")
        # self.up_conv1 = UpConvLayer(self.k*fact3, self.k*fact2)
        # self.conv2 = ConvLayer(self.k*fact2, self.k*fact1)
        # self.final_conv = nn.Sequential(nn.Conv2d(self.k*fact1, self.k, kernel_size=3, stride=1, padding=1), 
        #                                 nn.ReLU(), 
        #                                 )
        # print("Defining final npi layer")
        # self.final_linear = nn.Linear(self.k, self.k)
        pass

    def forward(self, orig_activs, target_label):
        # print("NPI forward: Combining targ {} and orig activs {}".format(target_label.size(), orig_activs.size()))
        combined = torch.cat((target_label, orig_activs), dim=1)
        # print("NPI forward: combined shape == ", combined.size())
        # print("NPI forward: passing through 1st linear")
        out_linear = self.first_linear(combined.view(-1, (self.n+1)*self.m*self.k))
        # out1 = self.conv1(out_linear)
        # out2 = self.down_conv1(out1)
        # out3 = self.down_conv2(out2)
        # out3 = self.first_conv_trans(out3)
        # out4 = self.up_conv1(torch.cat((out2, out3), dim=2))
        # out5 = self.conv2(torch.cat((out1, out4), dim=2))
        # out6 = self.final_conv(out5)
        # return self.final_linear(out6)
        return out_linear.view(-1, self.n, self.m, self.k)
#------------------------------------------------------------------------------------------------------


class Classifier(nn.Module): # classifies NPI outputs
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
        self.b = input_activs_shape[0] # batch size
        self.n = input_activs_shape[1]
        self.m = input_activs_shape[2]
        self.k = input_activs_shape[3]
        self.N = self.n*self.m

        fact1 = 2**4#2**10#4 #NOTE HACK
        fact2 = 2**5#2**11#5
        fact3 = 2**6#2**12#6

        print("Defining classifier model", flush=True)
        # print("self.n == ", self.n)
        # print("self.n//fact1 == ", self.n//fact1)
        # print("self.n//fact2 == ", self.n//fact2)
        # print("self.n//fact3 == ", self.n//fact3)
        self.model = nn.Sequential(# nn.Linear(self.n, self.n), 
                                    # nn.ReLU(), 
                                    nn.Linear(self.n*self.m*self.k, self.n//fact1), # HACK: NOTE: changed self.n to self.N
                                    nn.ReLU(), 
                                    nn.Linear(self.n//fact1, self.n//fact2), 
                                    nn.ReLU(), 
                                    nn.Linear(self.n//fact2, self.n//fact3), 
                                    nn.ReLU(), 
                                    nn.Linear(self.n//fact3, 1),#*self.m*self.k), 
                                    nn.Sigmoid(),
                                    )

    def forward(self, x):
        return self.model(x.view(-1, self.n*self.m*self.k))#.view(-1, 1, self.m, self.k)
        pass


class ConvClassifier(nn.Module):
    # initializers
    def __init__(self, d=128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d*4, d*8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d*8)
        self.conv5 = nn.Conv2d(d*8, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.conv4_bn(self.conv4(x)), 0.2)
        x = torch.sigmoid(self.conv5(x))

        return x




def train_classifier(args): # UNDER CONSTRUCTIONNNNNNNNNNNNNNNNNNN

    """
    DATA: [
        [(435x1024x1), (1x1024x1), [], 'gpt2', {}]
        ...
    ]
    """

    # compute CONST_N
    CONST_N = len(args.pred_inds) * seq_len * num_iters # 200
    if -1 in args.pred_inds or num_total_layers-1 in args.pred_inds:
        CONST_N = CONST_N - num_iters

    # initialize function vars
    # raise NotImplementedError("Interpretation protocol has not been written")
    save_file_path = args.save_file_path
    train_file_path_base = args.train_file_path_base
    train_file_path = train_file_path_base + ".pkl_0"#"_0.pkl"
    #NORM_LOSS_WT = args.norm_loss_wt # NOTE: norm loss weight

    device = args.device
    # scaling_coeff = args.scaling_coeff
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    test_freq = args.test_freq
    save_freq = args.save_freq
    
    try:
        # READ IN DATASET
        print("Loading Data")
        train_data, test_data = load_training_data(train_file_path,args) 
                                # Random selection of sentence-lists in each list (train_data, test_data, val_data)
        
        # Save the train_data, test_data, val_data to pkl files

        # NOTE: commented out to expedite testing
        # out_path = save_file_path+"train_data.pkl"
        # with open(out_path, 'wb') as outfile:
        #     pkl.dump(train_data, outfile, protocol=pkl.HIGHEST_PROTOCOL)
        # out_path = save_file_path+"test_data.pkl"
        # with open(out_path, 'wb') as outfile:
        #     pkl.dump(test_data, outfile, protocol=pkl.HIGHEST_PROTOCOL)
        # out_path = save_file_path+"validation_data.pkl"
        # with open(out_path, 'wb') as outfile:
        #     pkl.dump(val_data, outfile, protocol=pkl.HIGHEST_PROTOCOL)

        # CREATE TRAIN / TEST LOADERS
        train_loader = DataLoader(train_data, batch_size=batch_size, pin_memory=True)
        test_loader = DataLoader(test_data, batch_size=batch_size, pin_memory=True)

        # MODEL INITIALIZATION
        print("Creating Classifier Model", flush=True)
        try:
            # act0, truth0, targ0 = next(iter(train_loader)) HACK's
            input_activs_shape = (batch_size, CONST_N, CONST_M, 1) # act0.size() HACK HACK
            input_truth_shape = (batch_size, 4) # truth0.size() HACK HACK
        except:
            pdb.set_trace()
        print("Act shapes ==", input_activs_shape, flush=True)
        print("label shapes ==", input_truth_shape, flush=True)
        classifier_model = None
        classifier_model = Classifier(input_activs_shape, input_truth_shape).float()
        classifier_model.cuda()#.cuda()

        print("Initializing class loss", flush=True)
        class_objective = torch.nn.BCELoss() #torch.nn.MSELoss() # See NOTE below
        #def class_objective(pred_labels,true_label): # NOTE: HACK to get rid of .5's in the classification
        #    loss_part_one = torch.nn.MSELoss()
        #    helper_loss = lambda x: torch.norm(NORM_LOSS_WT * torch.exp(-(x-.5)**2 / .007),p=1)
        #    return loss_part_one(pred_labels,true_label) + helper_loss(pred_labels[:,:,0,:])
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
        # num_epochs = (num_epochs//total_pkls) * total_pkls 
        
        print("Defining loop report", flush=True)
        loop = tqdm(total=len(train_loader)*total_pkls, position=0, leave=False)

        print("Training", flush=True)

        for epoch in range(num_epochs):
            gc.collect()
            # class_batch_accuracies = []
            class_batch_losses = []

            # Get data from next pickle if necessary 
            for pkl_num in range(total_pkls):
                train_file_path = train_file_path_base + ".pkl_" + str(pkl_num) #"_" + str(pkl_num) + ".pkl"
                train_data, test_data = load_training_data(train_file_path,args) # komya
                train_loader = DataLoader(train_data, batch_size=batch_size, pin_memory=True)
                test_loader = DataLoader(test_data, batch_size=batch_size, pin_memory=True)

                for batch, (orig_activ, real_label, _target_label) in enumerate(train_loader):
                    # Catch any infs!!!!!
                    if -np.inf in orig_activ:
                        raise ValueError("Found inf in array")
                    # prepare the batch for model processing
                    orig_activ, real_label = orig_activ.cuda().float(), real_label.cuda().float()
                    # print("<<< BATCH {} >>>".format(batch))

                    # UPDATE CLASSIFIER WEIGHTS
                    # print("Updating Classifier")
                    for p in classifier_model.parameters():
                        p.requires_grad = True
                
                    # Find labels and loss
                    class_optimizer.zero_grad()
                    class_loss = None
                    labels = classifier_model(orig_activ)
                    #pdb.set_trace()
                    #class_loss = class_objective(labels, real_label) * 1e12 # calculate lossssssss
                    class_loss = class_objective(labels.squeeze(), real_label[:,1].squeeze()) * 1e8 # NOTE: HACK here
                    # backprop
                    class_loss.backward()
                    class_batch_losses.append(class_loss.item()) # append to losses
                    class_optimizer.step()

                    # report current state to terminal
                    loop.set_description('epoch:{}'.format(epoch))
                    loop.update(1)

                    if batch % test_freq == 0:
                        # perform npi_model testing
                        class_test_losses = []
                        # print("TESTING")
                        for test_batch, (test_x, test_truth, test_y) in enumerate(test_loader):
                            test_x, test_truth, test_y = test_x.cuda().float(), test_truth.cuda().float(), test_y.cuda().float()

                            # print("Testing Classifier")
                            lhat = classifier_model(test_x)
                            # class_loss = class_objective(lhat, test_truth) * 1e12
                            class_loss = class_objective(lhat.squeeze(), test_truth[:,1].squeeze()) * 1e8
                            class_test_losses.append(class_loss.item())

                        class_tests.append((epoch, (sum(class_test_losses)/float(len(class_test_losses)))))
                        pass
                    pass

            # record average loss for epoch
            if len(class_batch_losses) > 0:
                class_epoch_losses.append((sum(class_batch_losses) / float(len(class_batch_losses))))
            else:
                class_epoch_losses.append(np.nan)

            if epoch % save_freq == 0:
                # print("SAVING DURING EPOCH ITERATION")
                # save the current version of the npi_mode

                out_path = save_file_path+"{}_classification_network_epoch{}.bin".format('Classifier', epoch)
                torch.save(classifier_model, out_path)
                out_path = save_file_path+"{}_classification_optimizer_epoch{}.bin".format('Classifier', epoch)
                torch.save(class_optimizer, out_path)
                out_path = save_file_path + "N8_classification_loss_summaries.pkl"
                with open(out_path, 'wb') as outfile:
                    pkl.dump({"epoch_losses":class_epoch_losses, "batch_losses":class_batch_losses, "tests":class_tests}, outfile)

        print("SAVING AFTER EPOCH ITERATION")    
        out_path = save_file_path + "N8_classification_loss_summaries.pkl"
        with open(out_path, 'wb') as outfile:
            pkl.dump({"epoch_losses":class_epoch_losses, "batch_losses":class_batch_losses, "tests":class_tests}, outfile)

        loop.close()
        print("Epoch loss history == ", epoch_losses)
        gc.collect()

        return classifier_model

    except Exception as e:
        print(e)
        torch.cuda.empty_cache()
        raise
    pass


if __name__ == "__main__":
    # train_classifier()
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_file_path", 
                        default=None, 
                        help="npi_models/")
    parser.add_argument("--train_file_path_base", 
                        default=None, 
                        help="data/")
    parser.add_argument("--num_epochs", 
                        type=int, 
                        default=70, 
                        help="number of epochs to train for")
    parser.add_argument("--batch_size", 
                        type=int, 
                        default=5, 
                        help="number of language model generated sequences to put into each training batch")
    parser.add_argument("--test_freq", 
                        type=int, 
                        default=5, 
                        help="test every test_freq batches")
    parser.add_argument("--save_freq", 
                        type=int, 
                        default=5, 
                        help="save the model during training every save_freq epochs")
    #parser.add_argument("--norm_loss_wt", # NOTE: added by nate
    #                    type=float,
    """
    python3 train_classifier.py --num_epochs 10 --batch_size 5 --test_freq 2 --save_freq 2 --num_pkls 1 --first_perturbation_index 5 --second_perturbation_index 11 --save_file_path ./classifiers/ --train_file_path_base ./data/
    """
    #                    default=.5,
    #                    help="how badly do we want the classifier not to choose .5 ?")
    parser.add_argument("--num_pkls",
                        type=int,
                        default=70,
                        help="how many pickle of data we got?")
    parser.add_argument("--gpu_num",
                        type=int,
                        default=0,
                        help="which GPU to use")
    parser.add_argument("--first_perturbation_index",
                        type=int,
                        default=0,
                        help="which first layer to perturb")
    parser.add_argument("--second_perturbation_index",
                        type=int,
                        default=1,
                        help="which second layer to perturb")
    parser.add_argument("--class_lr",
                        type=float,
                        default=1e-5,
                        help="model optimizer learning rate")


    """
    Nate's example for how to run this dang thing:

python3 train_class.py --save_file_path class_attempt0/ --train_file_path_base transformers/gan_pkls/sentence_arrays --num_epochs 1000 --batch_size 5 --test_freq 20 --save_freq 200 --num_pkls 70

OR

python3 train_class.py --save_file_path class_attempt0/ --train_file_path_base transformers/gan_pkls/REAL_OFFENSE/REAL_OFFENSE_sentence_arrays --num_epochs 350 --batch_size 5 --test_freq 40 --save_freq 50 --num_pkls 29
    """
    # parser.add_argument("--length", type=int, default=20)
    # parser.add_argument("--temperature", type=float, default=1.0)
    # parser.add_argument("--top_k", type=int, default=0)
    # parser.add_argument("--top_p", type=float, default=0.9)
    # parser.add_argument("--no_cuda", action='store_true',
    #                     help="Avoid using CUDA when available")

    args = parser.parse_args()

    args.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if torch.cuda.is_available():
        if args.gpu_num == 0:
            args.device = torch.device("cuda:0")
        elif args.gpu_num == 1:
            args.device = torch.device("cuda:1")

    # args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")

    if not os.path.exists(args.save_file_path):
        os.mkdir(args.save_file_path)

    if args.save_file_path[-1] != '/':
        args.save_file_path = args.save_file_path + '/'

    pi0, pi1 = args.first_perturbation_index, args.second_perturbation_index
    pis = [pi0, pi1]
    args.save_file_path = args.save_file_path + "layers_{}_{}/".format(pi0,pi1)
    if not os.path.exists(args.save_file_path):
        os.mkdir(args.save_file_path)
    args.pred_inds = pis
    mod = train_classifier(args=args)

    # args.n_gpu = torch.cuda.device_count()

    # torch.cuda.empty_cache()
    # npi_model, classifier_model = train_NPI(args)
    
    # out_path = args.save_file_path+"{}_npi_vfinal.bin".format(args.npi_type)
    # torch.save(npi_model, out_path)
    # out_path = args.save_file_path+"{}_classifier_vfinal.bin".format(args.classifier_type)
    # torch.save(classifier_model, out_path)
    # torch.cuda.empty_cache()
    pass

# THIS IS A COMMENT that I inserted from my windows machine because it's cool

"""
Train command 1: 02/01/2020
python3 train_neural_program_interfaces.py --save_file_path ~/Documents// --train_file_path_base ~/Documents/data.pkl --style_coeff 1.5 --similarity_coeff 2.05 --npi_type 'ConvNPI' --classifier_type adversarial --num_epochs 50 --batch_size 2 --test_freq 2 --save_freq 3

"""
