#         Train Neural Programming Interfaces       #
#                                                   #
#     Adversarially in tandem with discriminator    #
#            (or 'Generation Classifier')           #
#                                                   #
#          Fulda, Brown, Wingate, Robinson          #
#                       DRAGN                       #
#                    NPI Project                    #
#                       2020                        #

"""
Overview:
    Classifiers:
        - Includes functionality for either training in-tandem with NPI or not
        - Includes functionality for loading pretrained classifiers
    Style Transfer Inspired Adversarial Loss
    Functionality for controlling various network activations:
        - Supported neural models:
            - GPT2
    Functionality for interpretting NPI outputs:
        - Not part of the NPI class, allows for reshaping generated 'controlled' 
          activations and running them through a given neural model
"""

import argparse
import gc
import time
import pickle as pkl
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from matplotlib import pyplot as plt
from npi.dataset.npi_dataset import NPIDataLoader, NPIDataSet
from npi.models.classifiers import StyleClassifier, Discriminator
from npi.models.npi import GPT2LMWithNPI, NPINetwork

from npi.transformers import GPT2Tokenizer

import faulthandler # Debugging seg faults

torch.manual_seed(1)

# NPI Code Block ################################################################################################

LOSS_BOOSTING_COEFF = 10000.

# first helper fcns
def my_accuracy(x, y):
    """
    Accepts vector of ground truth labels and vector of generated labels
    Order not important, as long as dims are equal
        x, y are both 1-dim torch.Tensor objects or np.ndarray
    """
    x, y = x.squeeze().data.cpu().numpy(), y.squeeze().data.cpu().numpy()
    x = np.array([round(xi) for xi in x])
    y = np.array([round(yi) for yi in y])
    if len(x) != 0:
        return len(x[x == y]) / len(x)
    else:
        return 0.


def load_training_data(file_path, pred_inds, split_ratio=.25):  # with test-train split
    with open(file_path, 'rb') as datafile:
        dataset = pkl.load(datafile)

    # rand.shuffle(dataset) # NOTE: WE ASSUME DATA HAS ALREADY BEEN SHUFFLED
    max_train = len(dataset) - int(split_ratio * len(dataset))
    # This commented-out bit for if you want the validation data to be set aside as you train
    #   (recommended to just set it aside beforehand by using fewer than the total number of pkl's)
    # max_test = len(dataset[max_train:]) - int(split_ratio*len(dataset[max_train:])) 

    return NPIDataSet(dataset[:max_train], pred_inds, return_row=True), NPIDataSet(dataset[max_train:], pred_inds, return_row=True)
    # , NPIDataSet(dataset[max_train:max_train+max_test]), NPIDataSet(dataset[max_train+max_test:]), None


class NPILoss(nn.Module):
    def __init__(self, discrim_coeff, style_coeff, similarity_coeff, style_class_model=None,
                 discrim_model=None):
        super(NPILoss, self).__init__()
        self.gamma = discrim_coeff
        self.alpha = style_coeff
        self.beta = similarity_coeff
        self.mse = torch.nn.MSELoss()
        self.bce = torch.nn.BCELoss()

        if discrim_model is not None:
            self.discrim_model = discrim_model
        if style_class_model is not None:
            self.style_class_model = style_class_model
        pass

    def forward(self, predicted_activs, true_activs, target_label,
                style_class_model=None, discrim_model=None, return_loss_data=False):
        """
        predicted_activs: torch tensor of shape (n, m, 1, b)
            b is the number of batches
            n x m x 1 slices contain the elements of the predicted activations, flattened into a 2D array

        true_activs: torch tensor of shape (n, m, 1, b)
            b is the number of batches
            n x m x 1 slices contain the elements of the original activations, flattened into a 2D array

        target_label: torch tensor of shape (1, m, 1, b)
            the desired label for the predicted activations, as passed into the NPI network

        classifier_model: an updated classifier model (optional: use for adversarial training)
        """
        generation_classifier_labels, _ = self.discrim_model(predicted_activs)
        content_classifier_labels = self.style_class_model(predicted_activs).unsqueeze(1).unsqueeze(3)
        aggregate_size = torch.cat((generation_classifier_labels, content_classifier_labels), dim=2).size()
        classifier_labels = torch.zeros(aggregate_size, dtype=torch.float64).cuda()
        classifier_labels[:, :, 0, :] = generation_classifier_labels[:, :, 0, :]
        classifier_labels[:, :, 1, :] = content_classifier_labels[:, :, 0, :]  # 1: to 1 and to 0

        new_discrim_score = self.gamma * self.bce(classifier_labels[:, :, 0, :], target_label[:, :, 0, :].double())
        new_style_score = self.alpha * self.bce(classifier_labels[:, :, 1, :],
                                                target_label[:, :, 1, :].double())  # 1: to 1
        old_content_score = self.beta * self.mse(predicted_activs, true_activs)

        if return_loss_data:
            return LOSS_BOOSTING_COEFF * (new_discrim_score + new_style_score + old_content_score), \
                   {"gen_class_loss": new_discrim_score.item(), "content_class_loss": new_style_score.item(),
                    "similarity_loss": old_content_score.item()}
        return LOSS_BOOSTING_COEFF * (new_discrim_score + new_style_score + old_content_score)

    # ------------------------------------------------------------------------------------------------------


def load_models(args, input_activs_shape, input_targ_shape):
    npi_type = args.npi_type
    content_class_type = args.content_classifier_type
    generate_class_type = args.generation_classifier_type

    # Creating NPI Model
    npi_model = None
    if npi_type == "adversarial":
        npi_model = NPINetwork(input_activs_shape, input_targ_shape).float()
        if args.npi_model_path is not None:
            npi_model.load_state_dict(torch.load(args.npi_model_path, map_location="cpu"))
            npi_model.eval()
    else:
        raise NotImplementedError("NPI should be trained adversarially")
    npi_model.cuda()

    # Creating ContentClassifier Model
    style_class_model = None
    if content_class_type == 'adversarial':
        raise NotImplementedError("Content classifier should be pre-trained")
        print("INITIALIZING NEW CONTENT CLASSIFIER NETWORK")
        style_class_model = ContentClassifier(input_activs_shape, input_targ_shape).float()
    elif content_class_type == 'pretrained' and args.style_classifier_path is not None:
        print("LOADING PRE-TRAINED CONTENT CLASSIFIER NETWORK")
        style_class_model: StyleClassifier = StyleClassifier()
        style_class_model.load_state_dict(torch.load(args.style_classifier_path, map_location=torch.device('cpu')))
        style_class_model.eval()
    else:
        raise NotImplementedError("Requested model {} has not been implemented.".format(content_class_type))
    style_class_model.cuda()

    # Creating GenerationClassifier Model
    discrim_model = None
    if generate_class_type == 'adversarial':
        discrim_model = Discriminator(input_activs_shape, input_targ_shape).float()
    elif generate_class_type == 'pretrained' and args.discrim_model_path is not None:
        raise NotImplementedError("Generation classifier should be trained adversarially in tandem with NPI")
        discrim_model = torch.load(args.discrim_model_path)
        discrim_model.eval()
    else:
        raise NotImplementedError("Requested model {} has not been implemented.".format(generate_class_type))
    
    if args.discrim_model_path is not None:
        discrim_model.load_state_dict(torch.load(args.discrim_model_path, map_location="cpu"))
        discrim_model.eval()

    discrim_model.cuda()

    return npi_model, style_class_model, discrim_model


def make_classifier_plots(classifier_label, epoch, save_file_path, epoch_losses, false_test_losses, true_test_losses,
                          train_accuracies, false_test_accuracies, true_test_accuracies):
    """
    Plot training progress for classifier network
    """
    test_epochs = []
    for i, elem in enumerate(true_test_losses):
        if elem[0] not in test_epochs:
            test_epochs.append(elem[0])

    avg_epoch_test_losses = []
    avg_epoch_test_accuracies = []
    avg_epoch_false_test_losses = []
    avg_epoch_false_test_accuracies = []
    avg_epoch_train_accuracies = []
    num_files = 0
    # make_classifier_plots : constructing test / accuracy avgs
    for i, ep in enumerate(test_epochs):
        curr_ep_losses = [x[1] for x in true_test_losses if x[0] == ep]
        curr_ep_accuracies = [x[1] for x in true_test_accuracies if x[0] == ep]
        curr_ep_false_losses = [x[1] for x in false_test_losses if x[0] == ep]
        curr_ep_false_accuracies = [x[1] for x in false_test_accuracies if x[0] == ep]

        # condense everything into lists of averages
        if curr_ep_losses:
            avg_epoch_test_losses.append(sum(curr_ep_losses) / len(curr_ep_losses))
        else:
            avg_epoch_test_losses.append(0)
        if curr_ep_accuracies:
            avg_epoch_test_accuracies.append(sum(curr_ep_accuracies) / len(curr_ep_accuracies))
        else:
            avg_epoch_test_accuracies.append(0)
        if curr_ep_false_losses:
            avg_epoch_false_test_losses.append(sum(curr_ep_false_losses) / len(curr_ep_false_losses))
        else:
            avg_epoch_false_test_losses.append(0)
        if curr_ep_false_accuracies:
            avg_epoch_false_test_accuracies.append(sum(curr_ep_false_accuracies) / len(curr_ep_false_accuracies))
        else:
            avg_epoch_false_test_accuracies.append(0)

        if train_accuracies is not None:
            curr_ep_accuracies = [x[1] for x in train_accuracies if
                                  x[0] == ep]  # train_accuracies[i*num_files:(i+1)*num_files]
            if curr_ep_accuracies:
                avg_epoch_train_accuracies.append(sum(curr_ep_accuracies) / len(curr_ep_accuracies))
            else:
                avg_epoch_train_accuracies.append(0)

        if i == 0:
            num_files = len(curr_ep_losses)

    avg_epoch_train_losses = []
    if epoch_losses is not None:
        # make_classifier_plots : averaging epoch losses
        for i in range(epoch):
            curr_ep_losses = epoch_losses[i * num_files:(i + 1) * num_files]
            if curr_ep_losses:
                avg_epoch_train_losses.append(sum(curr_ep_losses) / len(curr_ep_losses))
            else:
                avg_epoch_train_losses.append(0)

    # make_classifier_plots
    fig1, ax1 = plt.subplots()
    if epoch_losses is not None:
        ax1.plot(avg_epoch_train_losses, label='average train')
    ax1.plot(test_epochs, avg_epoch_test_losses, label='average test')
    ax1.plot(test_epochs, avg_epoch_false_test_losses, label='generated test')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Average Loss")
    ax1.set_title("{} Average Losses Per Epoch".format(classifier_label))
    plt.legend()
    plt.draw()
    fig1.savefig(save_file_path + "visualization_epoch{}_{}_train_vs_test_losses.png".format(epoch, classifier_label))

    # make_classifier_plots : making plot 2
    fig2, ax2 = plt.subplots()
    if train_accuracies is not None:
        ax2.plot(test_epochs, avg_epoch_train_accuracies, label='average train')
    ax2.plot(test_epochs, avg_epoch_test_accuracies, label='average test')
    ax2.plot(test_epochs, avg_epoch_false_test_accuracies, label='generated test')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Average Accuracy")
    ax2.set_title("{} Average Accuracies Per Epoch".format(classifier_label))
    plt.legend()
    plt.draw()
    fig2.savefig(
        save_file_path + "visualization_epoch{}_{}_train_vs_test_accuracies.png".format(epoch, classifier_label))

    return avg_epoch_train_losses, avg_epoch_test_losses, avg_epoch_false_test_losses, avg_epoch_train_accuracies, \
           avg_epoch_test_accuracies, avg_epoch_false_test_accuracies, test_epochs
    pass


def make_npi_plots(epoch, save_file_path, epoch_losses, test_losses):
    """
    Make plots for training progress of NPI network
    """
    test_epochs = []
    # make_npi_plots: test_losses[0] == test_losses[0]

    for i, elem in enumerate(test_losses):
        if elem[0] not in test_epochs:
            test_epochs.append(elem[0])

    avg_epoch_test_losses = []
    num_files = 0
    # make_npi_plots: obtaining avg test losses
    for i, ep in enumerate(test_epochs):
        curr_ep_losses = [x[1] for x in test_losses if x[0] == ep]
        if i == 0:
            num_files = len(curr_ep_losses)
        if curr_ep_losses:
            avg_epoch_test_losses.append(sum(curr_ep_losses) / len(curr_ep_losses))
        else:
            avg_epoch_test_losses.append(0)

    # make_npi_plots: obtaining avg train losses
    avg_epoch_train_losses = []
    for i in range(epoch):
        curr_ep_losses = epoch_losses[i * num_files:(i + 1) * num_files]
        if curr_ep_losses:
            avg_epoch_train_losses.append(sum(curr_ep_losses) / len(curr_ep_losses))
        else:
            avg_epoch_train_losses.append(0)

    # make_npi_plots: plotting
    fig1, ax1 = plt.subplots()
    ax1.plot(avg_epoch_train_losses, label='training')
    ax1.plot(test_epochs, avg_epoch_test_losses, label='testing')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Average Loss")
    ax1.set_title("NPI Average Losses Per Epoch")
    plt.legend()
    plt.draw()
    fig1.savefig(save_file_path + "visualization_epoch{}_NPI_train_vs_test_losses.png".format(epoch))

    return avg_epoch_train_losses, avg_epoch_test_losses, test_epochs


def train_adversarial_NPI(args):  # train NPI and Classifiers in-tandem
    """
    ***Main function***
    """

    start_time = time.time()

    LANG_MODEL_ACTS_IND = 0
    ACTS_CLASSIF_IND = 1
    TARG_CLASSIF_IND = 2
    LANG_MODEL_TYPE_IND = 3
    META_DATA_IND = 4
    ORIG_TEXT_IND = 5
    PRED_TEXT_IND = 6
    TARG_TEXT_INDEX = 7
    GPT2_TEXT_INDEX = 8  # the text of what the gpt2 actually produced

    HEAD_START_NUM = args.head_start_num

    print("############################################################")
    print("<<<        USING THE FOLLOWING INPUT ARGUMENTS!!!        >>>")
    print(args)
    print("############################################################")

    # initialize function vars
    save_file_path = args.save_file_path
    train_file_path = args.train_file_path
    if not "pkl" in train_file_path:  # train file path should have specific format
        train_file_path = train_file_path + ".pkl_"
    num_pkls = args.num_pkls
    train_file_names = [str(pn) for pn in range(num_pkls)]  # os.listdir(train_file_path)
    # train_file_names.sort()
    print("############################################################")
    print("<<<  NOTE :  ONLY FIRST {} FILES IN DATA SET BEING USED  >>>".format(num_pkls))
    print("############################################################")

    device = torch.device('cuda:{}'.format(args.gpu_num))
    discrim_coeff = args.discrim_coeff
    style_coeff = args.style_coeff
    similarity_coeff = args.similarity_coeff
    npi_type = args.npi_type
    content_class_type = args.content_classifier_type
    generate_class_type = args.generation_classifier_type
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    test_freq = args.test_freq
    save_freq = args.save_freq

    try:
        torch.cuda.empty_cache()
        # READ IN DATASET
        print("Loading Data")
        print("<<<<<<<<< NOT FILTERING DATA -- ASSUMING RELATIVE CLASS BALANCE >>>>>>>>>>")
        train_data, _ = load_training_data(train_file_path + train_file_names[0], args.perturbation_indices,
                                           split_ratio=.25)  # _, _, _ and .25

        # CREATE TRAIN LOADER
        train_loader = NPIDataLoader(train_data, batch_size=batch_size, pin_memory=True)

        # MODEL INITIALIZATION
        print("Creating ", npi_type, " npi")
        act0, _, targ0, data_rows = train_data[0]
        print("act0", act0.shape)
        input_activs_shape = act0.size()
        input_targ_shape = (1, 1)  # targ0.size() <-- None

        npi_model, style_class_model, discrim_model = load_models(args, input_activs_shape, input_targ_shape)

        print("Initializing GPT2WithNPI model with tokenizer -- not being placed on GPU until npi loss evaluation")
        gpt2_with_npi = GPT2LMWithNPI.from_pretrained(
            args.language_model_type)  # lang model type may be 'gpt2' or 'gpt2-medium'
        gpt2_with_npi = gpt2_with_npi.cuda()
        gpt2_with_npi.initialize_npi(args.perturbation_indices, lang_model_type=args.language_model_type)
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained(args.language_model_type)

        # CREATE LOSS FUNCTION
        print("Initializing npi loss func")
        npi_objective = NPILoss(discrim_coeff, style_coeff, similarity_coeff, style_class_model, discrim_model)
        npi_optimizer = optim.Adam(npi_model.parameters(), lr=args.npi_lr)

        print("Initializing classifier losses")
        # content_class_objective = torch.nn.BCELoss()
        # content_class_optimizer = optim.Adam(style_class_model.parameters(), lr=args.disc_lr)
        discrim_objective = torch.nn.BCELoss()
        discrim_model_optimizer = optim.Adam(discrim_model.parameters(), lr=args.disc_lr)

        mse_loss = torch.nn.MSELoss()
        bce_loss = torch.nn.BCELoss()

        print("Setting Content Classifier and GPT-2 Parameters to requires_grad=False")
        for p in style_class_model.parameters():
            p.requires_grad = False
        for p in gpt2_with_npi.parameters():
            p.requires_grad = False

        # PERFORM MODEL TRAINING
        npi_epoch_losses = []
        npi_batch_losses = []
        npi_test_losses = []

        content_class_tests = []
        content_false_class_tests = []

        generate_class_epoch_losses = []
        generate_class_batch_losses = []
        generate_class_tests = []
        generate_false_class_tests = []

        generate_class_train_accuracies = []
        content_class_test_accuracies = []
        generate_class_test_accuracies = []
        content_class_false_test_accuracies = []
        generate_class_false_test_accuracies = []

        class_sample_meta_data = {"training data": {}, "testing data": {}}
        npi_sample_meta_data = {"training data": {}, "testing data": {}}

        print("Training")

        for epoch in range(num_epochs):
            gc.collect()
            print("############ Epoch == ", epoch, " ############")

            # ITERATE THROUGH ALL AVAILABLE TRAIING DATA (different pkl files)
            loop = tqdm(total=len(train_file_names), position=0, leave=False)
            for file_num, train_file_name in enumerate(train_file_names):
                gc.collect()
                train_data, test_data = load_training_data(train_file_path + train_file_name, args.perturbation_indices,
                                                           split_ratio=.25)  # val_data, _ and .25

                # CREATE TRAIN / TEST LOADERS
                train_loader = NPIDataLoader(train_data, batch_size=batch_size, pin_memory=True)
                test_loader = NPIDataLoader(test_data, batch_size=batch_size, pin_memory=True)

                npi_batch_losses = []
                generate_class_batch_losses = []
                generate_class_train_batch_accuracies = []

                # Looping through training batches
                for batch, (orig_activ, real_label, target_label, data_idx) in enumerate(train_loader):

                    functional_batch_size = orig_activ.shape[0]

                    # prepare the batch for model processing
                    input_activs_shape = orig_activ.size()
                    input_targ_shape = (1, 1)  # target_label.size() <-- this is None
                    orig_activ, real_label = orig_activ.cuda(non_blocking=True).float(), real_label.cuda(
                        non_blocking=True).float()

                    # ~~~~ TRAINING SEGMENT open ~~~~

                    curr_rows = train_loader.get_row_data(data_idx)
                    for i in range(len(curr_rows)):
                        curr_rows[i] = [None] * 4 + curr_rows[i][4:]

                        # Get perturbed activations that we'll need throughout training iteration
                    pred_activs = npi_model(orig_activ)
                    gpt2_with_npi = gpt2_with_npi.cuda()
                    pred_gpt2_outs, training_text = gpt2_with_npi.obtain_perturbed_GPT2WithNPI_outputs(
                        pred_activs,
                        args.perturbation_indices,
                        curr_rows,
                        tokenizer=gpt2_tokenizer,
                        max_seq_len=args.max_seq_len,
                        num_seq_iters=args.num_seq_iters,
                        device=args.device)

                    g_class_loss_item = None

                    if epoch >= HEAD_START_NUM:  # NPI gets a headstart on the generation classifier in adversarial training

                        # UPDATE CLASSIFIER WEIGHTS

                        discrim_model.train()

                        for p in npi_model.parameters():
                            p.requires_grad = False
                        for p in discrim_model.parameters():
                            p.requires_grad = True

                        discrim_model.zero_grad()  # discrim_model_optimizer.zero_grad()

                        # labels 
                        y_real_GPT2 = torch.zeros(functional_batch_size).float().cuda()  # 0 = real GPT2
                        y_fake_GPT2 = torch.ones(functional_batch_size).float().cuda()  # 1 = fake GPT2
                        # y_real_GPT2, y_fake_GPT2 = Variable(y_real_GPT2), Variable(y_fake_GPT2)

                        # Now predict and get loss
                        real_gen_pred = discrim_model(orig_activ)
                        fake_gen_pred = discrim_model(pred_gpt2_outs.detach())
                        # loss
                        real_loss = discrim_objective(real_gen_pred.squeeze(), y_real_GPT2.squeeze())
                        fake_loss = discrim_objective(fake_gen_pred.squeeze(), y_fake_GPT2.squeeze())
                        g_class_loss = LOSS_BOOSTING_COEFF * (real_loss + fake_loss)
                        # record and .backward()
                        g_class_loss_item = g_class_loss.item()
                        generate_class_batch_losses.append(g_class_loss_item)
                        g_class_loss.backward()
                        discrim_model_optimizer.step()

                        # UPDATE NPI WEIGHTS

                    npi_model.train()

                    for p in npi_model.parameters():
                        p.requires_grad = True
                    for p in discrim_model.parameters():
                        p.requires_grad = False

                    npi_model.zero_grad()  # npi_optimizer.zero_grad()

                    npi_objective.discrim_model = discrim_model
                    gpt2_with_npi.npi_model = npi_model

                    # labels 
                    y_word = torch.ones(
                        functional_batch_size).float().cuda()  # ones here corresponds to having NO sexist slurs
                    y_real_GPT2 = torch.zeros(functional_batch_size).float().cuda()

                    # pred activations already calculated
                    resulting_gpt2_outs = pred_gpt2_outs

                    # get classifications and loss 
                    style_classification = style_class_model(resulting_gpt2_outs)
                    discrim_classification = discrim_model(resulting_gpt2_outs)
                    # loss
                    discrim_loss = bce_loss(discrim_classification.squeeze(), y_real_GPT2.squeeze())
                    style_loss = bce_loss(style_classification.squeeze(), y_word.squeeze())
                    similarity_loss = mse_loss(resulting_gpt2_outs, orig_activ)
                    npi_loss = LOSS_BOOSTING_COEFF * (
                            discrim_coeff * discrim_loss + style_coeff * style_loss + similarity_coeff * similarity_loss)
                    # now record and report state to terminal and then .backward()
                    npi_batch_losses.append(npi_loss.item())
                    if g_class_loss_item is not None:  # will be None if we are still in the headstart
                        loop.set_description(
                            'epoch:{}, gen_class_loss:{:.2f}, npi_loss:{:.2f}, time_elapsed:{:.1f}'.format(epoch,
                                                                                                           g_class_loss_item,
                                                                                                           npi_loss.item(),
                                                                                                           (
                                                                                                                   time.time() - start_time)))
                    else:
                        loop.set_description(
                            'epoch:{}, gen_class_loss:N/A, npi_loss:{:.2f}, time_elapsed:{:.1f}'.format(epoch,
                                                                                                        npi_loss.item(),
                                                                                                        (
                                                                                                                time.time() - start_time)))
                    # loop.update(1)
                    npi_loss.backward()
                    npi_optimizer.step()

                    # save meta data
                    if (epoch % save_freq == 0) and file_num == len(
                            train_file_names) - 1 and batch == 0 and epoch >= HEAD_START_NUM:
                        class_sample_meta_data["training data"]["epoch {}".format(epoch)] = \
                            {"real array classifications": real_gen_pred.squeeze().data.cpu().numpy(),
                             "NPI-produced array classifications": fake_gen_pred.squeeze().data.cpu().numpy(),
                             "training loss": g_class_loss.cpu().item()
                             }
                        npi_sample_meta_data["training data"]["epoch {}".format(epoch)] = \
                            {"style loss": style_loss.cpu().item(),
                             "similarity loss": similarity_loss.cpu().item(),
                             "discrim loss": discrim_loss.cpu().item(),
                             "content classifier classifications": style_classification.squeeze().data.cpu().numpy(),
                             "test_samples": training_text
                             }

                        # This marks the end of looping through the training batches! :D

                # collect more averages for meta data
                if npi_batch_losses:
                    npi_epoch_losses.append((sum(npi_batch_losses) / float(len(npi_batch_losses))))

                if generate_class_batch_losses:
                    generate_class_epoch_losses.append(
                        (sum(generate_class_batch_losses) / float(len(generate_class_batch_losses))))

                if epoch % test_freq == 0 and generate_class_train_batch_accuracies and epoch >= HEAD_START_NUM:
                    generate_class_train_accuracies.append((epoch, (sum(generate_class_train_batch_accuracies) / float(
                        len(generate_class_train_batch_accuracies)))))

                # TESTING 

                if epoch % test_freq == 0 and epoch >= HEAD_START_NUM:  # and epoch >= 1: # AFTER TRAINING PERFORM ANY REQUIRED TESTS
                    # print("Testing: START")
                    # perform npi_model testing
                    npi_test_batch_losses = []
                    content_class_test_losses = []
                    content_false_class_test_losses = []
                    generation_class_test_losses = []
                    generation_false_class_test_losses = []

                    content_class_test_batch_accuracies = []
                    generate_class_test_batch_accuracies = []
                    content_class_false_test_batch_accuracies = []
                    generate_class_false_test_batch_accuracies = []

                    for test_batch, (test_x, test_t, test_y, test_inds) in enumerate(test_loader):

                        # For testing we don't even deal with weirdly sized batches because that messes with averages
                        if test_x.shape[0] != batch_size:
                            continue

                        # Now we know functional_batch_size == batch_size
                        y_real_GPT2 = torch.zeros(batch_size).float().cuda()  # 0 = real GPT2
                        y_fake_GPT2 = torch.ones(batch_size).float().cuda()  # 1 = fake GPT2
                        y_word = torch.ones(batch_size).float().cuda()

                        test_x, test_t, test_y = test_x.cuda(non_blocking=True).float(), test_t.cuda(
                            non_blocking=True).float(), test_y.cuda(non_blocking=True).float()

                        curr_rows = test_loader.get_row_data(test_inds)
                        for i in range(len(curr_rows)):
                            curr_rows[i] = [None] * 4 + curr_rows[i][4:]

                        test_deltas = npi_model(test_x)
                        test_gpt2_outs, test_text = gpt2_with_npi.obtain_perturbed_GPT2WithNPI_outputs(test_deltas,
                                                                                                       args.perturbation_indices, \
                                                                                                       curr_rows,
                                                                                                       tokenizer=gpt2_tokenizer,
                                                                                                       max_seq_len=args.max_seq_len, \
                                                                                                       num_seq_iters=args.num_seq_iters,
                                                                                                       device=args.device)

                        discrim_model.eval()
                        test_real_gen_pred = discrim_model(test_x)
                        test_fake_gen_pred = discrim_model(test_gpt2_outs)
                        test_real_gen_loss = discrim_objective(test_real_gen_pred.squeeze(),
                                                                      y_real_GPT2.squeeze())
                        test_fake_gen_loss = discrim_objective(test_fake_gen_pred.squeeze(),
                                                                      y_fake_GPT2.squeeze())
                        test_g_class_loss = LOSS_BOOSTING_COEFF * (test_real_gen_loss + test_fake_gen_loss)
                        # append losses and get accuracy 
                        generation_class_test_losses.append(
                            test_g_class_loss.item())  # note this is the sum of real and fake loss
                        generation_false_class_test_losses.append(test_fake_gen_loss.item())
                        test_real_gen_acc = my_accuracy(test_real_gen_pred.squeeze(), y_real_GPT2.squeeze())
                        test_fake_gen_acc = my_accuracy(test_fake_gen_pred.squeeze(), y_fake_GPT2.squeeze())
                        test_avg_gen_acc = (test_real_gen_acc + test_fake_gen_acc) / 2.
                        generate_class_test_batch_accuracies.append(test_avg_gen_acc)
                        generate_class_false_test_batch_accuracies.append(test_fake_gen_acc)

                        npi_model.eval()
                        test_style_classification = style_class_model(test_gpt2_outs)
                        test_discrim_classification = test_fake_gen_pred
                        test_discrim_loss = bce_loss(test_discrim_classification.squeeze(), y_real_GPT2.squeeze())
                        test_style_loss = bce_loss(test_style_classification.squeeze(), y_word.squeeze())
                        test_similarity_loss = mse_loss(test_gpt2_outs, test_x)
                        test_npi_loss = LOSS_BOOSTING_COEFF * (
                                discrim_coeff * test_discrim_loss + style_coeff * test_style_loss + similarity_coeff * test_similarity_loss)
                        # append losses and get accuracy 
                        npi_test_batch_losses.append(test_npi_loss.item())
                        # Don't forget the accuracy number from the classifier 
                        acc_from_content_class = my_accuracy(test_style_classification.squeeze(), y_word.squeeze())
                        content_class_false_test_batch_accuracies.append(acc_from_content_class)

                        if file_num == len(train_file_names) - 1 and test_batch == 0:
                            class_sample_meta_data["testing data"]["epoch {}".format(epoch)] = \
                                {"real array classifications": test_real_gen_pred.squeeze().data.cpu().numpy(),
                                 "NPI-produced array classifications": test_fake_gen_pred.squeeze().data.cpu().numpy(),
                                 "testing loss": test_g_class_loss.cpu().item(),
                                 "testing accuracy": test_avg_gen_acc
                                 }
                            npi_sample_meta_data["testing data"]["epoch {}".format(epoch)] = \
                                {"style loss": test_style_loss.cpu().item(),
                                 "similarity loss": test_similarity_loss.cpu().item(),
                                 "discrim loss": test_discrim_loss.cpu().item(),
                                 "content classifier classifications": test_style_classification.squeeze().data.cpu().numpy(),
                                 "text samples": test_text
                                 }

                    # Testing: Storing loss avgs
                    if npi_test_batch_losses:
                        npi_test_losses.append(
                            (epoch, (sum(npi_test_batch_losses) / float(len(npi_test_batch_losses)))))
                    if content_class_test_losses:
                        content_class_tests.append(
                            (epoch, (sum(content_class_test_losses) / float(len(content_class_test_losses)))))
                    if content_false_class_test_losses:
                        content_false_class_tests.append((epoch, (sum(content_false_class_test_losses) / float(
                            len(content_false_class_test_losses)))))
                    if generation_class_test_losses:
                        generate_class_tests.append(
                            (epoch, (sum(generation_class_test_losses) / float(len(generation_class_test_losses)))))
                    if generation_false_class_test_losses:
                        generate_false_class_tests.append((epoch, (sum(generation_false_class_test_losses) / float(
                            len(generation_false_class_test_losses)))))

                    # Testing: Storing accuracy avgs
                    if content_class_test_batch_accuracies:
                        content_class_test_accuracies.append((epoch, (sum(content_class_test_batch_accuracies) / float(
                            len(content_class_test_batch_accuracies)))))
                    if generate_class_test_batch_accuracies:
                        generate_class_test_accuracies.append((epoch, (
                                sum(generate_class_test_batch_accuracies) / float(
                            len(generate_class_test_batch_accuracies)))))
                    if content_class_false_test_batch_accuracies:
                        content_class_false_test_accuracies.append((epoch, (
                                sum(content_class_false_test_batch_accuracies) / float(
                            len(content_class_false_test_batch_accuracies)))))
                    if generate_class_false_test_batch_accuracies:
                        generate_class_false_test_accuracies.append((epoch, (
                                sum(generate_class_false_test_batch_accuracies) / float(
                            len(generate_class_false_test_batch_accuracies)))))

                    # Testing: STOP

                # report current state to terminal
                torch.cuda.empty_cache()
                if g_class_loss_item is not None:
                    loop.set_description(
                        'epoch:{}, gen_class_loss:{:.2f}, npi_loss:{:.2f}, time_elapsed:{:.1f}'.format(epoch,
                                                                                                       g_class_loss_item,
                                                                                                       npi_loss.item(),
                                                                                                       (
                                                                                                               time.time() - start_time)))
                else:
                    loop.set_description(
                        'epoch:{}, gen_class_loss:N/A, npi_loss:{:.2f}, time_elapsed:{:.1f}'.format(epoch,
                                                                                                    npi_loss.item(), (
                                                                                                            time.time() - start_time)))
                loop.update(1)

            print("end of regular epoch")

            if epoch % save_freq == 0 and epoch >= HEAD_START_NUM:
                # save the current version of the npi_model
                print("Saving NPI Model")
                out_path = save_file_path + "{}_npi_network_epoch{}.bin".format(npi_type, epoch)
                torch.save(npi_model.state_dict(), out_path)

                print("Saving NPI Loss Summary")
                out_path = save_file_path + "{}_npi_loss_summaries_epoch{}.pkl".format(npi_type, epoch)
                with open(out_path, 'wb') as outfile:
                    pkl.dump({"epoch_losses": npi_epoch_losses,
                              "test_losses": npi_test_losses,
                              "accuracies_from_content_class": content_class_false_test_accuracies,
                              "sample_meta_data": npi_sample_meta_data,
                              }, outfile)

                # print("Saving ContentClassifier Loss Summary")
                # out_path = save_file_path + "{}_loss_summaries_epoch{}.pkl".format("ContentClassifier", epoch)
                # with open(out_path, 'wb') as outfile:
                #    pkl.dump({"false_test_losses": content_false_class_tests, 
                #                "avg_test_losses": content_class_tests, 
                #                "false_test_accuracies": content_class_false_test_accuracies, 
                #                "avg_test_accuracies": content_class_test_accuracies, 
                #             }, outfile)

                print("Saving GenerationClassifier Model")
                out_path = save_file_path + "{}_network_epoch{}.bin".format('GenerationClassifier', epoch)
                torch.save(discrim_model.state_dict(), out_path)

                print("Saving GenerationClassifier Loss Summary")
                out_path = None
                out_path = save_file_path + "{}_loss_summaries_epoch{}.pkl".format("GenerationClassifier", epoch)
                with open(out_path, 'wb') as outfile:
                    pkl.dump({"epoch_losses": generate_class_epoch_losses,
                              "false_tests": generate_false_class_tests,
                              "avg_tests": generate_class_tests,
                              "training_accuracies": generate_class_train_accuracies,
                              "false_test_accuracies": generate_class_false_test_accuracies,
                              "avg_test_accuracies": generate_class_test_accuracies,
                              "sample_meta_data": class_sample_meta_data,
                              }, outfile)

                print("Done saving for current epoch")

                # ~~~~~~NOW for the visualizations~~~~~~~~~~~~~~~~~~~~~~~~~~

                print("Saving Data Visualizations: START")

                print("obtaining NPI visualizations")
                npi_avg_epoch_train_losses, \
                npi_avg_epoch_test_losses, \
                npi_test_epochs = make_npi_plots(epoch, save_file_path, npi_epoch_losses, npi_test_losses)

                with open(save_file_path + "{}_averages_for_visualization_plots.pkl".format('NPI'), 'wb') as outfile:
                    pkl.dump({'avg_epoch_train_losses': npi_avg_epoch_train_losses,
                              'avg_epoch_test_losses': npi_avg_epoch_test_losses,
                              'test_epochs': npi_test_epochs,
                              }, outfile)

                print("obtaining ContentClassifier visualizations")
                content_class_avg_epoch_train_losses, \
                content_class_avg_epoch_test_losses, \
                content_class_avg_epoch_false_test_losses, \
                content_class_avg_epoch_train_accuracies, \
                content_class_avg_epoch_test_accuracies, \
                content_class_avg_epoch_false_test_accuracies, \
                content_test_epochs = make_classifier_plots("ContentClassifier", epoch, save_file_path,
                                                            None,
                                                            content_false_class_tests,
                                                            content_class_tests,
                                                            None,
                                                            content_class_false_test_accuracies,
                                                            content_class_test_accuracies)

                with open(save_file_path + "{}_averages_for_visualization_plots.pkl".format('ContentClassifier'),
                          'wb') as outfile:
                    pkl.dump({'avg_epoch_train_losses': content_class_avg_epoch_train_losses,
                              'avg_epoch_test_losses': content_class_avg_epoch_test_losses,
                              'avg_epoch_false_test_losses': content_class_avg_epoch_false_test_losses,
                              'avg_epoch_train_accuracies': content_class_avg_epoch_train_accuracies,
                              'avg_epoch_test_accuracies': content_class_avg_epoch_test_accuracies,
                              'avg_epoch_false_test_accuracies': content_class_avg_epoch_false_test_accuracies,
                              'test_epochs': content_test_epochs,
                              }, outfile)

                print("obtaining GenerationClassifier visualizations")
                gen_class_avg_epoch_train_losses, \
                gen_class_avg_epoch_test_losses, \
                gen_class_avg_epoch_false_test_losses, \
                gen_class_avg_epoch_train_accuracies, \
                gen_class_avg_epoch_test_accuracies, \
                gen_class_avg_epoch_false_test_accuracies, \
                gen_test_epochs = make_classifier_plots("GenerationClassifier", epoch, save_file_path,
                                                        generate_class_epoch_losses,
                                                        generate_false_class_tests,
                                                        generate_class_tests,
                                                        generate_class_train_accuracies,
                                                        generate_class_false_test_accuracies,
                                                        generate_class_test_accuracies)

                with open(save_file_path + "{}_averages_for_visualization_plots.pkl".format('GenerationClassifier'),
                          'wb') as outfile:
                    pkl.dump({'avg_epoch_train_losses': gen_class_avg_epoch_train_losses,
                              'avg_epoch_test_losses': gen_class_avg_epoch_test_losses,
                              'avg_epoch_false_test_losses': gen_class_avg_epoch_false_test_losses,
                              'avg_epoch_train_accuracies': gen_class_avg_epoch_train_accuracies,
                              'avg_epoch_test_accuracies': gen_class_avg_epoch_test_accuracies,
                              'avg_epoch_false_test_accuracies': gen_class_avg_epoch_false_test_accuracies,
                              'test_epochs': gen_test_epochs,
                              }, outfile)

                print("Saving Data Visualizations: STOP")
            torch.cuda.empty_cache()
            loop.close()

        print("SAVING META-DATA AFTER FULL TRAINING - NPI AND CLASSIFIER RETURNED (TO MAIN), NOT SAVED")
        out_path = save_file_path + "{}_loss_summaries_final.pkl".format(npi_type)
        with open(out_path, 'wb') as outfile:
            pkl.dump({"epoch_losses": npi_epoch_losses,
                      "test_losses": npi_test_losses,
                      "accuracies_from_content_class": content_class_false_test_accuracies,
                      "sample_meta_data": npi_sample_meta_data,
                      }, outfile)

        # print("Saving ContentClassifier Loss Summary")
        # out_path = save_file_path + "{}_loss_summaries_final.pkl".format("ContentClassifier")
        # with open(out_path, 'wb') as outfile:
        #    pkl.dump({"false_tests": content_false_class_tests, 
        #                "true_tests": content_class_tests, 
        #                "false_test_accuracies": content_class_false_test_accuracies, 
        #                "true_test_accuracies": content_class_test_accuracies, 
        #             }, outfile)

        print("Saving GenerationClassifier Loss Summary")
        out_path = save_file_path + "{}_loss_summaries_final.pkl".format("GenerationClassifier")
        with open(out_path, 'wb') as outfile:
            pkl.dump({"epoch_losses": generate_class_epoch_losses,
                      "false_tests": generate_false_class_tests,
                      "avg_tests": generate_class_tests,
                      "training_accuracies": generate_class_train_accuracies,
                      "false_test_accuracies": generate_class_false_test_accuracies,
                      "avg_test_accuracies": generate_class_test_accuracies,
                      "sample_meta_data": class_sample_meta_data,
                      }, outfile)

        print("Saving Data Visualizations: START")

        npi_avg_epoch_train_losses, \
        npi_avg_epoch_test_losses, \
        npi_test_epochs = make_npi_plots(num_epochs, save_file_path, npi_epoch_losses, npi_test_losses)

        with open(save_file_path + "{}_final_averages_for_visualization_plots.pkl".format('NPI'), 'wb') as outfile:
            pkl.dump({'avg_epoch_train_losses': npi_avg_epoch_train_losses,
                      'avg_epoch_test_losses': npi_avg_epoch_test_losses,
                      'test_epochs': npi_test_epochs,
                      }, outfile)

        content_class_avg_epoch_train_losses, \
        content_class_avg_epoch_test_losses, \
        content_class_avg_epoch_false_test_losses, \
        content_class_avg_epoch_train_accuracies, \
        content_class_avg_epoch_test_accuracies, \
        content_class_avg_epoch_false_test_accuracies, \
        content_test_epochs = make_classifier_plots("ContentClassifier", num_epochs, save_file_path,
                                                    None,
                                                    content_false_class_tests,
                                                    content_class_tests,
                                                    None,
                                                    content_class_false_test_accuracies,
                                                    content_class_test_accuracies)

        with open(save_file_path + "{}_final_averages_for_visualization_plots.pkl".format('ContentClassifier'),
                  'wb') as outfile:
            pkl.dump({'avg_epoch_train_losses': content_class_avg_epoch_train_losses,
                      'avg_epoch_test_losses': content_class_avg_epoch_test_losses,
                      'avg_epoch_false_test_losses': content_class_avg_epoch_false_test_losses,
                      'avg_epoch_train_accuracies': content_class_avg_epoch_train_accuracies,
                      'avg_epoch_test_accuracies': content_class_avg_epoch_test_accuracies,
                      'avg_epoch_false_test_accuracies': content_class_avg_epoch_false_test_accuracies,
                      'test_epochs': content_test_epochs,
                      }, outfile)

        gen_class_avg_epoch_train_losses, \
        gen_class_avg_epoch_test_losses, \
        gen_class_avg_epoch_false_test_losses, \
        gen_class_avg_epoch_train_accuracies, \
        gen_class_avg_epoch_test_accuracies, \
        gen_class_avg_epoch_false_test_accuracies, \
        gen_test_epochs = make_classifier_plots("GenerationClassifier", num_epochs, save_file_path,
                                                generate_class_epoch_losses,
                                                generate_false_class_tests,
                                                generate_class_tests,
                                                generate_class_train_accuracies,
                                                generate_class_false_test_accuracies,
                                                generate_class_test_accuracies)

        with open(save_file_path + "{}_final_averages_for_visualization_plots.pkl".format('GenerationClassifier'),
                  'wb') as outfile:
            pkl.dump({'avg_epoch_train_losses': gen_class_avg_epoch_train_losses,
                      'avg_epoch_test_losses': gen_class_avg_epoch_test_losses,
                      'avg_epoch_false_test_losses': gen_class_avg_epoch_false_test_losses,
                      'avg_epoch_train_accuracies': gen_class_avg_epoch_train_accuracies,
                      'avg_epoch_test_accuracies': gen_class_avg_epoch_test_accuracies,
                      'avg_epoch_false_test_accuracies': gen_class_avg_epoch_false_test_accuracies,
                      'test_epochs': gen_test_epochs,
                      }, outfile)

        print("Saving Data Visualizations: STOP")

        print("Epoch loss history == ", npi_epoch_losses)
        gc.collect()
        epoch_losses_cleaned = [x for x in npi_epoch_losses if x is not np.nan]
        return npi_model, style_class_model, discrim_model, np.mean(epoch_losses_cleaned)

    except Exception as e:
        print(e)
        torch.cuda.empty_cache()
        raise
    pass


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--save-file-path",
                        default="npi_models/",
                        help="/path/to/save/model/to/")
    parser.add_argument("--train-file-path",
                        default="data/sentence_arrays",
                        help="/path/to/training/dataset/")
    parser.add_argument("--npi-lr",
                        type=float,
                        default=1e-6,  # -- CHANGED from 1e-4 05/11/2020 at 10:07pm
                        help="npi learning rate")
    parser.add_argument("--disc-lr",
                        type=float,
                        default=1e-6,
                        help="(generation) classifiers' learning rate")
    parser.add_argument("--language-model-type",
                        default='gpt2',
                        help="one of: [gpt2, gpt2-medium]")
    parser.add_argument("--num-epochs",
                        type=int,
                        default=60,
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
                        default=10,
                        help="save the model during training every save_freq epochs")
    parser.add_argument("--npi-type",
                        default='adversarial',
                        help="one of: [pretrained, adversarial]")
    parser.add_argument("--content-classifier-type",
                        default='pretrained',
                        help="one of: [pretrained, adversarial]")
    parser.add_argument("--generation-classifier-type",
                        default='adversarial',
                        help="one of: [pretrained, adversarial]")
    parser.add_argument("--npi-model-path",
                        default=None,
                        help="/path/to/optional_pretrained_npi.bin")
    parser.add_argument("--content-classifier-path",
                        default="classifiers/layers_5_11/Classifier_classification_network_epoch50.bin",
                        # ^ CHANGE this if a different classifier performed better when you ran test_classifier script
                        help="/path/to/optional_content_classifier.bin")
    parser.add_argument("--generation-classifier-path",
                        default=None,
                        help="/path/to/optional_generation_classifier.bin")
    # parser.add_argument("--update-pretrained-npi", 
    #                     type=bool, 
    #                     default=True, 
    #                     help="Whether or not to update npi weights with backprop")
    # parser.add_argument("--update-pretrained-content-classifier", 
    #                     type=bool, 
    #                     default=False, 
    #                     help="Whether or not to update content_classifier weights with backprop")
    # parser.add_argument("--update-pretrained-generation-classifier", 
    #                     type=bool, 
    #                     default=True, 
    #                     help="Whether or not to update generation_classifier weights with backprop")
    parser.add_argument("--num-pkls",
                        type=int,
                        default=53,
                        help="Number of training data files")
    parser.add_argument("--gpu-num",  # NOTE: not implemented fully
                        type=int,
                        default=0,
                        help="Which GPU to use")
    parser.add_argument("--discrim-coeff",  # gamma
                        type=float,
                        default=3.0,
                        help="Discriminator (or 'Generation Classifer') loss coefficient")
    parser.add_argument("--style-coeff",  # alpha
                        type=float,
                        default=10.0,
                        help="Content classifier loss coefficient")
    parser.add_argument("--similarity-coeff",  # beta
                        type=float,
                        default=1.0,
                        help="MSE similarity loss coefficient")
    parser.add_argument("--head-start-num",
                        type=int,
                        default=5,
                        help="Give the NPI this many epochs of head start on the discriminator")
    parser.add_argument("--perturbation-indices",
                        type=str,
                        default="5,11",
                        help="indices for layers to extract from language model activations: string of numbers separated by commas")
    parser.add_argument("--max-seq-len",
                        type=int,
                        default=10,
                        help="Length of tokens list to pass through language model")
    parser.add_argument("--num-seq-iters",
                        type=int,
                        default=10,
                        help="Number of times to run text through language model iteratively")

    parser.add_argument("--no-cuda", action='store_true',
                        help="Avoid using CUDA when available")

    args = parser.parse_args()

    faulthandler.enable()
    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    torch.cuda.empty_cache()

    # discrim_coeffs = [args.discrim_coeff] # for grid search
    # style_coeffs = [args.style_coeff]
    # similarity_coeffs = [args.similarity_coeff]

    # format perturbation indices correctly
    args.perturbation_indices = [int(pi) for pi in args.perturbation_indices.split(',')]
    # construct file directory suffix
    dir_suffix = ""
    for pi in args.perturbation_indices:
        dir_suffix = dir_suffix + "_" + str(pi)

    # best_min_epoch_loss = None # These var's for grid search
    # best_discrim_coeff = None
    # best_style_coeff = None
    # best_similarity_coeff = None

    # Just creating save directory here, if it doesn't exist
    #   First make sure it is formatted correctly
    args.save_file_path = args.save_file_path if args.save_file_path[-1] == '/' else args.save_file_path + '/'
    orig_save_file_path = args.save_file_path
    split_file_path = orig_save_file_path.split('/')
    gradual_path = ""
    for path_elem in split_file_path:
        gradual_path = gradual_path + path_elem + "/"
        if not os.path.exists(gradual_path):
            os.mkdir(gradual_path)

    # language model type should be global
    global LANG_MODEL_TYPE
    LANG_MODEL_TYPE = args.language_model_type

    # grid_search_iter = 0
    if True:  # for coeffs in coeffs_hyperparam_list: # Commented out because we don't want a grid search
        if True:  # for layers in layers_hyperparam_list:

            args.save_file_path = orig_save_file_path + "params_discco{}_styco{}_simco{}_layers{}/".format(
                args.discrim_coeff, args.style_coeff, \
                args.similarity_coeff, dir_suffix)
            # Finish making save directory
            if not os.path.exists(args.save_file_path):
                os.mkdir(args.save_file_path)

            try:
                torch.cuda.empty_cache()
                # Main event:
                npi_model, style_class_model, discrim_model, avg_epoch_loss = train_adversarial_NPI(
                    args)

                out_path = args.save_file_path + "{}_npi_vfinal.bin".format(args.npi_type)
                torch.save(npi_model.state_dict(), out_path)
                out_path = args.save_file_path + "content_classifier_vfinal.bin"
                torch.save(style_class_model.state_dict(), out_path)
                out_path = args.save_file_path + "generation_classifier_vfinal.bin"
                torch.save(discrim_model.state_dict(), out_path)

                print("Avg epoch loss == ", avg_epoch_loss)

                del npi_model
                del style_class_model
                del discrim_model

            except:
                raise

            finally:
                torch.cuda.empty_cache()
    print("DONE!!!")
