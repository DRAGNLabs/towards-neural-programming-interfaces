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

from transformers import GPT2LMHeadModel, GPT2Tokenizer
import run_generation as rg

from modeling_neural_program_interfaces import *
from train_classifier import Classifier, extract_needed_layers

from torch.nn import CrossEntropyLoss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.manual_seed(1)
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable, grad

from tqdm import trange
from tqdm import tqdm
import gc

import numpy as np
from matplotlib import pyplot as plt
import pickle as pkl
import random as rand
import argparse
import os
import time
import pdb

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
        return len(x[x==y])/len(x)
    else:
        return 0.


def load_training_data(file_path, pred_inds, split_ratio=.25, permitted_rows=None): # with test-train split
    with open(file_path, 'rb') as datafile:
        dataset = pkl.load(datafile)

    # rand.shuffle(dataset) # NOTE: WE ASSUME DATA HAS ALREADY BEEN SHUFFLED
    max_train = len(dataset) - int(split_ratio*len(dataset))
    # This commented-out bit for if you want the validation data to be set aside as you train
    #   (recommended to just set it aside beforehand by using fewer than the total number of pkl's)
    # max_test = len(dataset[max_train:]) - int(split_ratio*len(dataset[max_train:])) 

    return NPIDataSet(dataset[:max_train], pred_inds), NPIDataSet(dataset[max_train:], pred_inds) 
        #, NPIDataSet(dataset[max_train:max_train+max_test]), NPIDataSet(dataset[max_train+max_test:]), None

class NPIDataSet(Dataset):
    def __init__(self, dataset, pred_inds, permitted_rows=None, start_index=0):
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

        # self.masking_coeff = 1e12

        if permitted_rows is None:
            self.dataset = dataset
        else:
            self.dataset = []
            for i in range(len(dataset)):
                if start_index+i in permitted_rows:
                    self.dataset.append(dataset[i])

        for i in range(len(self.dataset)):
            # mask the inf values in the activations to simply be VERY VERY LARGE values
            #self.dataset[i][self.ORIG_ACTIV_INDEX][self.dataset[i][self.ORIG_ACTIV_INDEX] == np.inf] = self.masking_coeff 
            #self.dataset[i][self.ORIG_ACTIV_INDEX][self.dataset[i][self.ORIG_ACTIV_INDEX] == -1.*np.inf] = -1.*self.masking_coeff 

            # cast everything as torch tensors, extract needed layers
            self.dataset[i][self.ORIG_ACTIV_INDEX] = torch.from_numpy(extract_needed_layers(self.dataset[i][self.ORIG_ACTIV_INDEX],pis=pred_inds)).double() 
            self.dataset[i][self.ORIG_LABEL_INDEX] = torch.from_numpy(np.array(self.dataset[i][self.ORIG_LABEL_INDEX])).double()
        pass

    def __getitem__(self, i):
        acts = self.dataset[i][self.ORIG_ACTIV_INDEX]
        true_label = self.dataset[i][self.ORIG_LABEL_INDEX]
        targ = self.dataset[i][self.TARG_LABEL_INDEX] # None
        return acts, true_label, targ, i
 
    def __len__(self):
        return len(self.dataset)

    def get_row_data(self, i):
        return self.dataset[i].copy()

class NPIDataLoader(DataLoader):
    def __init__(self, data, batch_size, pin_memory):
        super(NPIDataLoader, self).__init__(data, batch_size=batch_size, pin_memory=pin_memory)
        self.data = data

    def get_row_data(self, dataset_indices):
        dataset_indices = dataset_indices.tolist()
        rows = []
        for index in dataset_indices:
            rows.append(self.data.get_row_data(index))
        return rows

class GPT2WithNPI(GPT2Model):
    r"""
    Modified from GPT2Model class in transformers module (from HuggingFace)

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the last layer of the model.
        **past**:
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            that contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2Model.from_pretrained('gpt2')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """
    def __init__(self, config): # NPI added functionality
        super(GPT2WithNPI, self).__init__(config) # NPI added functionality

        GPT2Model.__init__(self, config) # NPI added functionality
        pass
        

    def initialize_npi(self, prediction_indices):
        self.perturbation_indices = prediction_indices # NPI added functionality
        self.output_hidden_states = True
        pass

    def forward(self, input_ids, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None, 
                activation_perturbations=None):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Attention mask.
        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, input_shape[-1])
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.n_layer

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = ()
        all_attentions = []
        all_hidden_states = ()
        # print("GPT2WithNPI: Total num layers == ", len(self.h))
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(hidden_states,
                            layer_past=layer_past,
                            attention_mask=attention_mask,
                            head_mask=head_mask[i])

            hidden_states, present = outputs[:2]
            for j, index in enumerate(self.perturbation_indices): 
                if i == index:
                    # GPT2 MODEL: perturbing activation layer == i
                    # GPT2 MODEL: hidden_states size == hidden_states.size()
                    # GPT2 MODEL: activation_perturbations[j] size == activation_perturbations[j].size()
                    hidden_states = hidden_states + activation_perturbations[j]
            presents = presents + (present,)

            if self.output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states, presents)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)
        return outputs  # last hidden state, presents, (all hidden_states), (attentions)

class GPT2LMWithNPI(GPT2LMHeadModel):
    r"""
    Modified from GPT2LMHeadModel class in transformers module (from HuggingFace)

        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``lm_labels = input_ids``
            Indices are selected in ``[-1, 0, ..., config.vocab_size]``
            All labels set to ``-1`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Language modeling loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **past**:
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            that contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import torch
        from transformers import GPT2Tokenizer, GPT2LMHeadModel

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]

    """
    def __init__(self, config):#, npi_config):
        super(GPT2LMWithNPI, self).__init__(config)
        
        GPT2LMHeadModel.__init__(self, config) # NPI added functionality
        pass

    def initialize_npi(self, prediction_indices):
        self.perturbation_indices = prediction_indices # NPI added functionality
        # self.output_hidden_states = True
        self.transformer = GPT2WithNPI.from_pretrained(LANG_MODEL_TYPE)#(config, self.npi, self.prediction_indices) # NPI added functionality
        self.transformer.initialize_npi(prediction_indices)
        self.npi_model = None
        pass

    def forward(self, input_ids, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None, activation_perturbations=None):
        """
        target_classification : nx1x1 target classification vector  # NPI added functionality
        """
        transformer_outputs = self.transformer(input_ids,
                                               past=past,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               head_mask=head_mask, 
                                               activation_perturbations=activation_perturbations) # NPI added functionality
        hidden_states = transformer_outputs[0]
        
        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)

    def obtain_perturbed_GPT2WithNPI_outputs(self, npi_batched_perturbations, perturbation_indices, \
                            data_rows, tokenizer=None, max_seq_len=10, num_seq_iters=10, device=None, data_inds=None):
        # obtain perturbed GPT2WithNPI outputs: START
        LANG_MODEL_ACTS_IND = 0 
        ACTS_CLASSIF_IND = 1
        TARG_CLASSIF_IND = 2
        LANG_MODEL_TYPE_IND = 3
        META_DATA_IND = 4
        ORIG_TEXT_IND = 5
        PRED_TEXT_IND = 6
        TARG_TEXT_INDEX = 7
        GPT2_TEXT_INDEX = 8 # the text of what the gpt2 actually produced
        top_k=1
        top_p=.9
        temperature = 1.
        masking_coeff = 1e12

        batched_deltas_shape = npi_batched_perturbations.size()
        b = batched_deltas_shape[0]
        n = batched_deltas_shape[1]
        m = batched_deltas_shape[2]
        k = batched_deltas_shape[3]

        gpt2_perturbed_outs = []
        npi_resulting_text = []
        # iterating over batches
        for j in range(b):
            # create input_ids
            tokens = data_rows[j][META_DATA_IND]['orig_tokens'] 
            tokens = torch.tensor(tokens, dtype=torch.long)#, device=device) 
            tokens = tokens.unsqueeze(0).repeat(1, 1) 
            tokens = tokens.cuda()

            # create list of un-flattened activation_perturbations from current batch elem
            # creating curr_perturbs
            reshaped = npi_batched_perturbations[j,:,:,0].view(1, n, m, 1)
            # chunking with reshaped size == reshaped.size()
            chunked = torch.chunk(reshaped, num_seq_iters*len(self.perturbation_indices), dim=1) 
                # ^ each hidden layer in the hugging face repo has shape (batch, seq_len, hidden_size)
            # casting chunked as list
            curr_perturbs = [x.view(1, max_seq_len, m) for x in chunked]
            
            # initializing big_array
            #   obtain flattened representation of the resulting perturbed forward pass in GPT-2
            big_array = [] # nxmx1
            sent = data_rows[j][ORIG_TEXT_IND]
            generated_sent = ""
            # iteratively producing big_array
            for i in range(num_seq_iters): 

                # Now run the model
                logits, presents, all_hiddens = self.forward(input_ids=tokens[:,-max_seq_len:], \
                                                      activation_perturbations=curr_perturbs[i*len(self.perturbation_indices):(i+1)*len(self.perturbation_indices)]) 
                                                                # all_hiddens is a list of len
                                                                # 25 or 13 with tensors of shape (gpt2 medium of small)
                                                                # (1,sent_len,1024) or (1,sent_len,768)
                # Add to big_array
                for index in self.perturbation_indices:
                    big_array.append(all_hiddens[index])#.data)

                # Now we extract the new token and add it to the list of tokens
                next_token_logits = logits[0,-1,:] / temperature
                filtered_logits = rg.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                next_token_list = next_token.tolist()
                next_word = tokenizer.decode(next_token_list)
                sent = sent + " " + next_word # we just update this so sent remains accurate for dict
                generated_sent = generated_sent + next_word + " "
                
                # ...update list of tokens
                tokens = torch.cat((tokens,next_token.unsqueeze(0)),dim=1).cuda()
            if tokenizer is not None:
                npi_sent_for_data_set = tokenizer.decode([x.item() for x in tokens[:,-max_seq_len:].flatten()])
                npi_resulting_text.append([data_rows[j][ORIG_TEXT_IND], data_rows[j][GPT2_TEXT_INDEX], data_rows[j][TARG_TEXT_INDEX], npi_sent_for_data_set, sent])

            del tokens

            # Now the big_array is a list of length (max_seq_len*2) of tensors with shape (1,max_seq_len,1024) or (1,max_seq_len,768)
            # completing big_array
            big_array = torch.cat(big_array, dim=1)
            big_array = big_array.permute(1,2,0).view(1, n, m, 1)

            # mask the inf values in the activations to simply be VERY VERY LARGE values
            #big_array[big_array == float("Inf")] = masking_coeff 
            #big_array[big_array == -1.*float("Inf")] = -1.*masking_coeffs

            # store for later concatenation
            gpt2_perturbed_outs.append(big_array)
            #   iteration stop
        # create the end-result of npi_perturbations
        # casting output as single torch tensor
        resulting_gpt2_activations = torch.cat(gpt2_perturbed_outs, dim=0)
        # obtain perturbed GPT2WithNPI outputs: STOP
        return resulting_gpt2_activations, npi_resulting_text # this is batched

# NPI Neural Model Code -------------------------------------------------------------------------------
class NPINetwork(nn.Module):
    def __init__(self, input_activs_shape, input_targ_shape):
        """
        input_activs_shape: tuple of (b, n, m, 1)
            b is the number of batches
            n x m x 1 slices contain the elements of the original activations, flattened into a 2D array
        """
        super(NPINetwork, self).__init__()
        print("NPI INITIALIZATION")
        self.b = input_activs_shape[0]
        self.n = input_activs_shape[1]
        self.m = input_activs_shape[2]
        self.k = input_activs_shape[3]

        # Setting Scaling Factors
        fact1 = 2**2
        fact2 = 2**3
        fact3 = 2**3

        # Defining first npi layer
        self.first_linear = nn.Sequential(nn.Linear((self.n)*self.m*self.k, self.n//fact1),  
                                          nn.ReLU(), 
                                          )
        self.second_linear = nn.Sequential(nn.Linear(self.n//fact1, self.n//fact1), 
                                          nn.ReLU(), 
                                          )
        self.third_linear = nn.Sequential(nn.Linear(self.n//fact1, self.n//fact2), 
                                          nn.ReLU(),  
                                          )
        self.fourth_linear = nn.Sequential(nn.Linear(self.n//fact2, self.n//fact2), 
                                           nn.ReLU(),  
                                           )
        self.fourth_linear_residual = nn.Sequential(nn.Linear(self.n//fact2, self.n//fact3), 
                                                    nn.ReLU(),  
                                                    )
        self.fifth_linear = nn.Sequential(nn.Linear(self.n//fact3, self.n//fact2), 
                                          nn.ReLU(),  
                                          )
        self.sixth_linear = nn.Sequential(nn.Linear(self.n//fact2, self.n//fact1), 
                                          nn.ReLU(),  
                                          )
        self.seventh_linear = nn.Sequential(nn.Linear(self.n//fact1, self.n//fact1), 
                                            nn.ReLU(),  
                                            )
        self.last_linear = nn.Sequential(nn.Linear(self.n//fact1, self.n*self.m*self.k), 
                                        )
        
        pass

    def forward(self, orig_activs):
        metadata = {'ordered_hidden_activations':[], 
                    'final_out_preview':None, 
                    'final_out_returned':None, 
                    'concatenated_input':None}
        combined = orig_activs #torch.cat((target_label, orig_activs), dim=1) 
        first_out = self.first_linear(combined.view(-1, (self.n)*self.m*self.k)) 
        second_out = self.second_linear(first_out)
        third_out = self.third_linear(second_out)
        fourth_out = self.fourth_linear(third_out)
        # fourth_out_resid = self.fourth_linear_residual(third_out+fourth_out)
        # fifth_out = self.fifth_linear(fourth_out_resid)
        # sixth_out = self.sixth_linear(third_out+fifth_out)
        # seventh_out = self.seventh_linear(second_out+sixth_out)
        # out_linear = self.last_linear(first_out+seventh_out)
        fourth_out_resid = self.fourth_linear_residual(fourth_out)
        fifth_out = self.fifth_linear(fourth_out_resid)
        sixth_out = self.sixth_linear(fifth_out)
        seventh_out = self.seventh_linear(sixth_out)
        out_linear = self.last_linear(seventh_out)
        final_out = out_linear.view(-1, self.n, self.m, self.k)

        #metadata['ordered_hidden_activations'] = [first_out.detach().data.cpu().numpy(), 
        #                                          second_out.detach().data.cpu().numpy(), 
        #                                          third_out.detach().data.cpu().numpy(), 
        #                                          fourth_out.detach().data.cpu().numpy(), 
        #                                          fourth_out_resid.detach().data.cpu().numpy(), 
        #                                          fifth_out.detach().data.cpu().numpy(), 
        #                                          sixth_out.detach().data.cpu().numpy(), 
        #                                          seventh_out.detach().data.cpu().numpy(), 
        #                                          ]
        #metadata['final_out_preview'] = out_linear.detach().data.cpu().numpy()
        #metadata['final_out_returned'] = final_out.detach().data.cpu().numpy()
        #metadata['concatenated_input'] = combined.detach().data.cpu().numpy()

        return final_out#, metadata 

#------------------------------------------------------------------------------------------------------
class ContentClassifier(nn.Module): # classifies NPI outputs
    def __init__(self, input_activs_shape, input_targ_shape):
        raise NotImplementedError("Content classifier should be pre-trained") 
        """
        input_activs_shape: tuple of (b, n, m, 1)
            b is the number of batches
            n x m x 1 slices contain the elements of the original activations, flattened into a 2D array
        """
        super(ContentClassifier, self).__init__()
        
        print("ContentClassifier INIT")
        self.b = input_activs_shape[0]
        self.n = input_activs_shape[1]
        self.m = input_activs_shape[2]
        self.k = input_activs_shape[3]

        self.l = 1 # input_targ_shape[2]

        fact1 = 2**3
        fact2 = 2**3
        fact3 = 2**3

        print("Defining ContentClassifier model")
        self.linear1 = nn.Sequential(nn.Linear(self.n*self.m*self.k, self.n//fact1), 
                                     nn.ReLU(), 
                                    )
        self.linear1Post = nn.Sequential(nn.Linear(self.n//fact1, self.n//fact1), 
                                     nn.ReLU(), 
                                    )
        self.linear2 = nn.Sequential(nn.Linear(self.n//fact1, self.n//fact1), 
                                     nn.ReLU(), 
                                    )
        self.linear3 = nn.Sequential(nn.Linear(self.n//fact1, self.n//fact2), 
                                     nn.ReLU(), 
                                    )
        self.linear4 = nn.Sequential(nn.Linear(self.n//fact2, self.n//fact2), 
                                     nn.ReLU(), 
                                    )
        self.linear5 = nn.Sequential(nn.Linear(self.n//fact2, self.n//fact3),
                                     nn.ReLU(), 
                                    )
        self.linear6 = nn.Sequential(nn.Linear(self.n//fact3, self.n//fact3),
                                     nn.ReLU(), 
                                    )
        self.linear7Pre = nn.Sequential(nn.Linear(self.n//fact3, self.n//fact3), 
                                     nn.ReLU(), 
                                    )
        self.linear7 = nn.Sequential(nn.Linear(self.n//fact3, 1*self.l*self.k), 
                                     nn.Sigmoid(), 
                                    )

    def forward(self, x):
        metadata = {'ordered_hidden_activations':[], 'final_out_preview':None, 'final_out_returned':None}
        out1 = self.linear1(x.view(-1, self.n*self.m*self.k))
        out1Post = self.linear1Post(out1)
        out2 = self.linear2(out1Post)
        out3 = self.linear3(out2)
        out4 = self.linear4(out3)
        out5 = self.linear5(out4)
        out6 = self.linear6(out5)
        out7Pre = self.linear7Pre(out6)
        final_out = self.linear7(out6)

        metadata['ordered_hidden_activations'] = [out1.detach().data.cpu().numpy(), 
                                                  out1Post.detach().data.cpu().numpy(), 
                                                  out2.detach().data.cpu().numpy(), 
                                                  out3.detach().data.cpu().numpy(), 
                                                  out4.detach().data.cpu().numpy(), 
                                                  out5.detach().data.cpu().numpy(), 
                                                  out6.detach().data.cpu().numpy(), 
                                                  out7Pre.detach().data.cpu().numpy(), 
                                                  ]
        metadata['final_out_preview'] = final_out.detach().data.cpu().numpy()
        metadata['final_out_returned'] = final_out.view(-1, 1, self.l, self.k).detach().data.cpu().numpy()
        return final_out.view(-1, 1, self.l, self.k), metadata


class GenerationClassifier(nn.Module): # classifies NPI outputs
    def __init__(self, input_activs_shape, input_targ_shape):
        """
        input_activs_shape: tuple of (b, n, m, 1)
            b is the number of batches
            n x m x 1 slices contain the elements of the original activations, flattened into a 2D array
        target_label: tuple of (b, 1, m, 1)
            the desired label for the predicted activations, as passed into the NPI network
        """
        super(GenerationClassifier, self).__init__()
        
        print("GenerationClassifier INIT")
        self.b = input_activs_shape[0]
        self.n = input_activs_shape[1]
        self.m = input_activs_shape[2]
        self.k = input_activs_shape[3]

        self.l = 1 

        fact1 = 2**3
        fact2 = 2**4
        fact3 = 2**5

        print("Defining GenerationClassifier model")

        self.layer1 = nn.Sequential(nn.Linear(self.n*self.m*self.k, self.n//fact1), 
                                    nn.ReLU(), 
                                    )
        self.layer2 = nn.Sequential(nn.Linear(self.n//fact1, self.n//fact1), 
                                    nn.ReLU(), 
                                    )
        self.layer3 = nn.Sequential(nn.Linear(self.n//fact1, self.n//fact2), 
                                    nn.ReLU(), 
                                    )
        self.layer4 = nn.Sequential(nn.Linear(self.n//fact2, self.n//fact2), 
                                    nn.ReLU(), 
                                    )
        self.layer5 = nn.Sequential(nn.Linear(self.n//fact2, self.n//fact3), 
                                    nn.ReLU(), 
                                    )
        self.layer6 = nn.Sequential(nn.Linear(self.n//fact3, self.n//fact3),
                                    nn.ReLU(), 
                                    )
        self.layer7 = nn.Sequential(nn.Linear(self.n//fact3, self.l*self.k),
                                    nn.Sigmoid(), 
                                    )

    def forward(self, x):
        metadata = {'ordered_hidden_activations':[], 'final_out_preview':None, 'final_out_returned':None}

        out1 = self.layer1(x.view(-1, self.n*self.m*self.k))
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)
        out6 = self.layer6(out5)
        final_out = self.layer7(out6)

        #metadata['ordered_hidden_activations'] = [out1.detach().data.cpu().numpy(), 
        #                                          out2.detach().data.cpu().numpy(), 
        #                                          out3.detach().data.cpu().numpy(), 
        #                                          out4.detach().data.cpu().numpy(), 
        #                                          out5.detach().data.cpu().numpy(), 
        #                                          out6.detach().data.cpu().numpy(), 
        #                                          ]
        #metadata['final_out_preview'] = final_out.detach().data.cpu().numpy()
        #metadata['final_out_returned'] = final_out.view(-1, 1, self.l, self.k).detach().data.cpu().numpy()
        return final_out.view(-1, 1, self.l, self.k)#, metadata 


class NPILoss(nn.Module): 
    def __init__(self, discrim_coeff, style_coeff, similarity_coeff, content_classifier_model=None, 
                 generation_classifier_model=None):
        super(NPILoss, self).__init__()
        self.gamma = discrim_coeff
        self.alpha = style_coeff
        self.beta = similarity_coeff
        self.mse = torch.nn.MSELoss()
        self.bce = torch.nn.BCELoss() 

        if generation_classifier_model is not None:
            self.generation_classifier_model = generation_classifier_model
        if content_classifier_model is not None:
            self.content_classifier_model = content_classifier_model
        pass

    def forward(self, predicted_activs, true_activs, target_label, 
                content_classifier_model=None, generation_classifier_model=None,return_loss_data=False):
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
        generation_classifier_labels, _ = self.generation_classifier_model(predicted_activs)
        content_classifier_labels = self.content_classifier_model(predicted_activs).unsqueeze(1).unsqueeze(3) 
        aggregate_size = torch.cat((generation_classifier_labels, content_classifier_labels),dim=2).size() 
        classifier_labels = torch.zeros(aggregate_size, dtype=torch.float64).cuda()
        classifier_labels[:,:,0,:] = generation_classifier_labels[:,:,0,:] 
        classifier_labels[:,:,1,:] = content_classifier_labels[:,:,0,:] # 1: to 1 and to 0

        new_discrim_score = self.gamma*self.bce(classifier_labels[:,:,0,:], target_label[:,:,0,:].double()) 
        new_style_score = self.alpha*self.bce(classifier_labels[:,:,1,:], target_label[:,:,1,:].double()) # 1: to 1
        old_content_score = self.beta*self.mse(predicted_activs, true_activs)
        
        if return_loss_data:
            return LOSS_BOOSTING_COEFF * (new_discrim_score + new_style_score + old_content_score), \
                    {"gen_class_loss":new_discrim_score.item(),"content_class_loss":new_style_score.item(),"similarity_loss":old_content_score.item()}
        return LOSS_BOOSTING_COEFF * (new_discrim_score + new_style_score + old_content_score) 
        
#------------------------------------------------------------------------------------------------------

def load_models(args, input_activs_shape, input_targ_shape):
    npi_type = args.npi_type
    content_class_type = args.content_classifier_type
    generate_class_type = args.generation_classifier_type

    # Creating NPI Model
    npi_model = None
    if npi_type == "adversarial":
        npi_model = NPINetwork(input_activs_shape, input_targ_shape).float()
    elif args.npi_model_path is not None:
        raise NotImplementedError("NPI should be trained adversarially")
        npi_model = torch.load(args.npi_model_path)
        npi_model.eval()
    else:
        raise NotImplementedError("Requested model {} has not been implemented.".format(npi_type))
    npi_model.cuda()

    # Creating ContentClassifier Model
    content_class_model = None
    if content_class_type == 'adversarial':
        raise NotImplementedError("Content classifier should be pre-trained") 
        print("INITIALIZING NEW CONTENT CLASSIFIER NETWORK")
        content_class_model = ContentClassifier(input_activs_shape, input_targ_shape).float()
    elif content_class_type == 'pretrained' and args.content_classifier_path is not None:
        print("LOADING PRE-TRAINED CONTENT CLASSIFIER NETWORK")
        content_class_model = torch.load(args.content_classifier_path,map_location=torch.device('cpu'))
        content_class_model.eval()
    else:
        raise NotImplementedError("Requested model {} has not been implemented.".format(content_class_type))
    content_class_model.cuda()

    # Creating GenerationClassifier Model
    generate_class_model = None
    if generate_class_type == 'adversarial':
        generate_class_model = GenerationClassifier(input_activs_shape, input_targ_shape).float()
    elif generate_class_type == 'pretrained' and args.generation_classifier_path is not None:
        raise NotImplementedError("Generation classifier should be trained adversarially in tandem with NPI") 
        generate_class_model = torch.load(args.generation_classifier_path)
        generate_class_model.eval()
    else:
        raise NotImplementedError("Requested model {} has not been implemented.".format(generate_class_type))
    generate_class_model.cuda()

    return npi_model, content_class_model, generate_class_model


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
        curr_ep_losses = [x[1] for x in true_test_losses if x[0]==ep]
        curr_ep_accuracies = [x[1] for x in true_test_accuracies if x[0]==ep]
        curr_ep_false_losses = [x[1] for x in false_test_losses if x[0]==ep]
        curr_ep_false_accuracies = [x[1] for x in false_test_accuracies if x[0]==ep]

        # condense everything into lists of averages
        if curr_ep_losses: 
            avg_epoch_test_losses.append(sum(curr_ep_losses)/len(curr_ep_losses))
        else:
            avg_epoch_test_losses.append(0)
        if curr_ep_accuracies:
            avg_epoch_test_accuracies.append(sum(curr_ep_accuracies)/len(curr_ep_accuracies))
        else:
            avg_epoch_test_accuracies.append(0)
        if curr_ep_false_losses:
            avg_epoch_false_test_losses.append(sum(curr_ep_false_losses)/len(curr_ep_false_losses))
        else:
            avg_epoch_false_test_losses.append(0)
        if curr_ep_false_accuracies:
            avg_epoch_false_test_accuracies.append(sum(curr_ep_false_accuracies)/len(curr_ep_false_accuracies))
        else:
            avg_epoch_false_test_accuracies.append(0)

        if train_accuracies is not None:
            curr_ep_accuracies = [x[1] for x in train_accuracies if x[0]==ep]#train_accuracies[i*num_files:(i+1)*num_files]
            if curr_ep_accuracies: 
                avg_epoch_train_accuracies.append(sum(curr_ep_accuracies)/len(curr_ep_accuracies))
            else:
                avg_epoch_train_accuracies.append(0)

        if i == 0:
            num_files = len(curr_ep_losses)

    avg_epoch_train_losses = []
    if epoch_losses is not None:
        # make_classifier_plots : averaging epoch losses
        for i in range(epoch):
            curr_ep_losses = epoch_losses[i*num_files:(i+1)*num_files]
            if curr_ep_losses: 
                avg_epoch_train_losses.append(sum(curr_ep_losses)/len(curr_ep_losses))
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
    fig1.savefig(save_file_path+"visualization_epoch{}_{}_train_vs_test_losses.png".format(epoch, classifier_label)) 

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
    fig2.savefig(save_file_path+"visualization_epoch{}_{}_train_vs_test_accuracies.png".format(epoch, classifier_label))

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
        curr_ep_losses = [x[1] for x in test_losses if x[0]==ep]
        if i == 0:
            num_files = len(curr_ep_losses)
        if curr_ep_losses: 
            avg_epoch_test_losses.append(sum(curr_ep_losses)/len(curr_ep_losses))
        else:
            avg_epoch_test_losses.append(0)

    # make_npi_plots: obtaining avg train losses
    avg_epoch_train_losses = []
    for i in range(epoch):
        curr_ep_losses = epoch_losses[i*num_files:(i+1)*num_files]
        if curr_ep_losses: 
            avg_epoch_train_losses.append(sum(curr_ep_losses)/len(curr_ep_losses))
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
    fig1.savefig(save_file_path+"visualization_epoch{}_NPI_train_vs_test_losses.png".format(epoch)) 

    return avg_epoch_train_losses, avg_epoch_test_losses, test_epochs


def train_adversarial_NPI(args): # train NPI and Classifiers in-tandem
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
    GPT2_TEXT_INDEX = 8 # the text of what the gpt2 actually produced

    HEAD_START_NUM = args.head_start_num

    print("############################################################")
    print("<<<        USING THE FOLLOWING INPUT ARGUMENTS!!!        >>>")
    print(args)
    print("############################################################")

    # initialize function vars
    save_file_path = args.save_file_path
    train_file_path = args.train_file_path
    if not "pkl" in train_file_path: # train file path should have specific format
        train_file_path = train_file_path + ".pkl_"
    num_pkls = args.num_pkls 
    train_file_names = [str(pn) for pn in range(num_pkls)]#os.listdir(train_file_path)
    #train_file_names.sort()
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
        train_data, _ = load_training_data(train_file_path+train_file_names[0], args.perturbation_indices, split_ratio=.25, filter_unk=False, permitted_rows=None) # _, _, _ and .25

        # CREATE TRAIN LOADER
        train_loader = NPIDataLoader(train_data, batch_size=batch_size, pin_memory=True)

        # MODEL INITIALIZATION
        print("Creating ", npi_type, " npi")
        act0, _, targ0, _ = next(iter(train_loader)) 
        input_activs_shape = act0.size()
        input_targ_shape = (1,1) # targ0.size() <-- None

        npi_model, content_class_model, generate_class_model = load_models(args, input_activs_shape, input_targ_shape)

        print("Initializing GPT2WithNPI model with tokenizer -- not being placed on GPU until npi loss evaluation")
        gpt2_with_npi = GPT2LMWithNPI.from_pretrained(args.language_model_type) # lang model type may be 'gpt2' or 'gpt2-medium'
        gpt2_with_npi = gpt2_with_npi.cuda() 
        gpt2_with_npi.initialize_npi(args.perturbation_indices) 
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained(args.language_model_type) 

        # CREATE LOSS FUNCTION
        print("Initializing npi loss func")
        npi_objective = NPILoss(discrim_coeff, style_coeff, similarity_coeff, content_class_model, generate_class_model)
        npi_optimizer = optim.Adam(npi_model.parameters(), lr=args.npi_lr)

        print("Initializing classifier losses")
        #content_class_objective = torch.nn.BCELoss()
        #content_class_optimizer = optim.Adam(content_class_model.parameters(), lr=args.disc_lr)
        generate_class_objective = torch.nn.BCELoss()
        generate_class_optimizer = optim.Adam(generate_class_model.parameters(), lr=args.disc_lr)

        mse_loss = torch.nn.MSELoss() 
        bce_loss = torch.nn.BCELoss() 

        print("Setting Content Classifier and GPT-2 Parameters to requires_grad=False")
        for p in content_class_model.parameters():
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

        class_sample_meta_data = {"training data":{}, "testing data": {}} 
        npi_sample_meta_data = {"training data":{}, "testing data":{}} 

        print("Training")

        for epoch in range(num_epochs):
            gc.collect()
            print("############ Epoch == ", epoch, " ############")

            # ITERATE THROUGH ALL AVAILABLE TRAIING DATA (different pkl files)
            loop = tqdm(total=len(train_file_names), position=0, leave=False)
            for file_num, train_file_name in enumerate(train_file_names):
                gc.collect()
                train_data, test_data = load_training_data(train_file_path+train_file_name, args.perturbation_indices, split_ratio=.25, filter_unk=False, permitted_rows=None) # val_data, _ and .25

                # CREATE TRAIN / TEST LOADERS
                train_loader = NPIDataLoader(train_data, batch_size=batch_size, pin_memory=True)
                test_loader = NPIDataLoader(test_data, batch_size=batch_size, pin_memory=True)

                npi_batch_losses = []
                generate_class_batch_losses = []
                generate_class_train_batch_accuracies = []                

                # Looping through training batches
                for batch, (orig_activ, real_label, target_label, data_inds) in enumerate(train_loader):
                    
                    functional_batch_size = orig_activ.shape[0] 

                    # prepare the batch for model processing
                    input_activs_shape = orig_activ.size()
                    input_targ_shape = (1,1) # target_label.size() <-- this is None
                    orig_activ, real_label = orig_activ.cuda(async=True).float(), \
                                             real_label.cuda(async=True).float()

                    # ~~~~ TRAINING SEGMENT open ~~~~

                    curr_rows = train_loader.get_row_data(data_inds)
                    for i in range(len(curr_rows)):
                        curr_rows[i] = [None] * 4 + curr_rows[i][4:] 

                    # Get perturbed activations that we'll need throughout training iteration 
                    pred_activs = npi_model(orig_activ) 
                    gpt2_with_npi = gpt2_with_npi.cuda() 
                    pred_gpt2_outs, training_text = gpt2_with_npi.obtain_perturbed_GPT2WithNPI_outputs(pred_activs, args.perturbation_indices, \
                                                                  curr_rows, tokenizer=gpt2_tokenizer, max_seq_len=args.max_seq_len, \
                                                                  num_seq_iters=args.num_seq_iters, device=args.device,data_inds=data_inds) 

                    g_class_loss_item = None

                    if epoch >= HEAD_START_NUM: # NPI gets a headstart on the generation classifier in adversarial training
                        
                        # UPDATE CLASSIFIER WEIGHTS
                    
                        generate_class_model.train() 
                    
                        for p in npi_model.parameters():
                            p.requires_grad = False
                        for p in generate_class_model.parameters():
                            p.requires_grad = True
                     
                        generate_class_model.zero_grad() #generate_class_optimizer.zero_grad()

                        # labels 
                        y_real_GPT2 = torch.zeros(functional_batch_size).float().cuda() # 0 = real GPT2
                        y_fake_GPT2 = torch.ones(functional_batch_size).float().cuda() # 1 = fake GPT2
                        #y_real_GPT2, y_fake_GPT2 = Variable(y_real_GPT2), Variable(y_fake_GPT2)

                        # Now predict and get loss
                        real_gen_pred = generate_class_model(orig_activ) 
                        fake_gen_pred = generate_class_model(pred_gpt2_outs.detach()) 
                        # loss
                        real_loss = generate_class_objective(real_gen_pred.squeeze(), y_real_GPT2.squeeze()) 
                        fake_loss = generate_class_objective(fake_gen_pred.squeeze(), y_fake_GPT2.squeeze()) 
                        g_class_loss = LOSS_BOOSTING_COEFF * (real_loss + fake_loss) 
                        # record and .backward()
                        g_class_loss_item = g_class_loss.item()
                        generate_class_batch_losses.append(g_class_loss_item) 
                        g_class_loss.backward() 
                        generate_class_optimizer.step() 

                    # UPDATE NPI WEIGHTS

                    npi_model.train()
                    
                    for p in npi_model.parameters():
                        p.requires_grad = True
                    for p in generate_class_model.parameters():
                        p.requires_grad = False

                    npi_model.zero_grad() #npi_optimizer.zero_grad() 

                    npi_objective.generation_classifier_model = generate_class_model 
                    gpt2_with_npi.npi_model = npi_model 

                    # labels 
                    y_word = torch.ones(functional_batch_size).float().cuda() # ones here corresponds to having NO sexist slurs
                    y_real_GPT2 = torch.zeros(functional_batch_size).float().cuda()

                    # pred activations already calculated
                    resulting_gpt2_outs = pred_gpt2_outs 

                    # get classifications and loss 
                    content_classification = content_class_model(resulting_gpt2_outs)
                    gen_classification = generate_class_model(resulting_gpt2_outs)
                    # loss
                    discrim_loss = bce_loss(gen_classification.squeeze(), y_real_GPT2.squeeze())
                    style_loss = bce_loss(content_classification.squeeze(), y_word.squeeze())
                    similarity_loss = mse_loss(resulting_gpt2_outs, orig_activ)
                    npi_loss = LOSS_BOOSTING_COEFF * (discrim_coeff*discrim_loss + style_coeff*style_loss + similarity_coeff*similarity_loss) 
                    # now record and report state to terminal and then .backward()
                    npi_batch_losses.append(npi_loss.item())
                    if g_class_loss_item is not None: # will be None if we are still in the headstart
                        loop.set_description('epoch:{}, gen_class_loss:{:.2f}, npi_loss:{:.2f}, time_elapsed:{:.1f}'.format(epoch, g_class_loss_item,npi_loss.item(),(time.time()-start_time)))
                    else:
                        loop.set_description('epoch:{}, gen_class_loss:N/A, npi_loss:{:.2f}, time_elapsed:{:.1f}'.format(epoch, npi_loss.item(),(time.time()-start_time)))
                    #loop.update(1)
                    npi_loss.backward()
                    npi_optimizer.step()

                    # save meta data
                    if (epoch % save_freq == 0) and file_num == len(train_file_names) - 1 and batch == 0 and epoch >= HEAD_START_NUM: 
                        
                        class_sample_meta_data["training data"]["epoch {}".format(epoch)] = \
                                                {"real array classifications":real_gen_pred.squeeze().data.cpu().numpy(),
                                                  "NPI-produced array classifications":fake_gen_pred.squeeze().data.cpu().numpy(),
                                                  "training loss":g_class_loss.cpu().item()
                                                  }
                        npi_sample_meta_data["training data"]["epoch {}".format(epoch)] = \
                                                {"style loss":style_loss.cpu().item(),
                                                  "similarity loss":similarity_loss.cpu().item(),
                                                  "discrim loss":discrim_loss.cpu().item(),
                                                  "content classifier classifications":content_classification.squeeze().data.cpu().numpy(), 
                                                  "test_samples":training_text
                                                  }                        
                
                # This marks the end of looping through the training batches! :D 

                # collect more averages for meta data
                if npi_batch_losses:
                    npi_epoch_losses.append((sum(npi_batch_losses) / float(len(npi_batch_losses))))

                if generate_class_batch_losses: 
                    generate_class_epoch_losses.append((sum(generate_class_batch_losses) / float(len(generate_class_batch_losses))))

                if epoch % test_freq == 0 and generate_class_train_batch_accuracies and epoch >= HEAD_START_NUM: 
                    generate_class_train_accuracies.append((epoch, (sum(generate_class_train_batch_accuracies) / float(len(generate_class_train_batch_accuracies)))))

                # TESTING 

                if epoch % test_freq == 0 and epoch >= HEAD_START_NUM:# and epoch >= 1: # AFTER TRAINING PERFORM ANY REQUIRED TESTS 
                    print("Testing: START")
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

                    for test_batch,(test_x, test_t, test_y, test_inds) in enumerate(test_loader):
                        
                        # For testing we don't even deal with weirdly sized batches because that messes with averages
                        if test_x.shape[0] != batch_size:
                            continue

                        # Now we know functional_batch_size == batch_size
                        y_real_GPT2 = torch.zeros(batch_size).float().cuda() # 0 = real GPT2
                        y_fake_GPT2 = torch.ones(batch_size).float().cuda() # 1 = fake GPT2
                        y_word = torch.ones(batch_size).float().cuda()

                        test_x, test_t, test_y = test_x.cuda(async=True).float(), \
                                                 test_t.cuda(async=True).float(), \
                                                 test_y.cuda(async=True).float()
                        
                        curr_rows = test_loader.get_row_data(test_inds)
                        for i in range(len(curr_rows)):
                            curr_rows[i] = [None] * 4 + curr_rows[i][4:] 

                        test_deltas = npi_model(test_x) 
                        test_gpt2_outs, test_text = gpt2_with_npi.obtain_perturbed_GPT2WithNPI_outputs(test_deltas, args.perturbation_indices, \
                                                                              curr_rows, tokenizer=gpt2_tokenizer, max_seq_len=args.max_seq_len, \
                                                                              num_seq_iters = args.num_seq_iters, device=args.device)          
                        
                        generate_class_model.eval() 
                        test_real_gen_pred = generate_class_model(test_x) 
                        test_fake_gen_pred = generate_class_model(test_gpt2_outs) 
                        test_real_gen_loss = generate_class_objective(test_real_gen_pred.squeeze(), y_real_GPT2.squeeze()) 
                        test_fake_gen_loss = generate_class_objective(test_fake_gen_pred.squeeze(), y_fake_GPT2.squeeze()) 
                        test_g_class_loss = LOSS_BOOSTING_COEFF * (test_real_gen_loss + test_fake_gen_loss) 
                        # append losses and get accuracy 
                        generation_class_test_losses.append(test_g_class_loss.item()) # note this is the sum of real and fake loss
                        generation_false_class_test_losses.append(test_fake_gen_loss.item())
                        test_real_gen_acc = my_accuracy(test_real_gen_pred.squeeze(), y_real_GPT2.squeeze())
                        test_fake_gen_acc = my_accuracy(test_fake_gen_pred.squeeze(), y_fake_GPT2.squeeze())
                        test_avg_gen_acc = (test_real_gen_acc + test_fake_gen_acc)/2.
                        generate_class_test_batch_accuracies.append(test_avg_gen_acc)
                        generate_class_false_test_batch_accuracies.append(test_fake_gen_acc)

                        npi_model.eval() 
                        test_content_classification = content_class_model(test_gpt2_outs)
                        test_gen_classification = test_fake_gen_pred
                        test_discrim_loss = bce_loss(test_gen_classification.squeeze(), y_real_GPT2.squeeze())
                        test_style_loss = bce_loss(test_content_classification.squeeze(), y_word.squeeze())
                        test_similarity_loss = mse_loss(test_gpt2_outs, test_x)
                        test_npi_loss = LOSS_BOOSTING_COEFF * (discrim_coeff*test_discrim_loss + style_coeff*test_style_loss + similarity_coeff*test_similarity_loss)
                        # append losses and get accuracy 
                        npi_test_batch_losses.append(test_npi_loss.item())
                        # Don't forget the accuracy number from the classifier 
                        acc_from_content_class = my_accuracy(test_content_classification.squeeze(),y_word.squeeze())
                        content_class_false_test_batch_accuracies.append(acc_from_content_class)

                        
                        if file_num == len(train_file_names) - 1 and test_batch == 0: 
                            
                            class_sample_meta_data["testing data"]["epoch {}".format(epoch)] = \
                                                    {"real array classifications":test_real_gen_pred.squeeze().data.cpu().numpy(),
                                                      "NPI-produced array classifications":test_fake_gen_pred.squeeze().data.cpu().numpy(),
                                                      "testing loss":test_g_class_loss.cpu().item(),
                                                      "testing accuracy":test_avg_gen_acc
                                                      }
                            npi_sample_meta_data["testing data"]["epoch {}".format(epoch)] = \
                                                    {"style loss":test_style_loss.cpu().item(),
                                                      "similarity loss":test_similarity_loss.cpu().item(),
                                                      "discrim loss":test_discrim_loss.cpu().item(),
                                                      "content classifier classifications":test_content_classification.squeeze().data.cpu().numpy(),
                                                      "text samples":test_text
                                                      }

                    # Testing: Storing loss avgs
                    if npi_test_batch_losses: 
                        npi_test_losses.append((epoch, (sum(npi_test_batch_losses)/float(len(npi_test_batch_losses)))))
                    if content_class_test_losses: 
                        content_class_tests.append((epoch, (sum(content_class_test_losses)/float(len(content_class_test_losses)))))
                    if content_false_class_test_losses: 
                        content_false_class_tests.append((epoch, (sum(content_false_class_test_losses)/float(len(content_false_class_test_losses)))))
                    if generation_class_test_losses: 
                        generate_class_tests.append((epoch, (sum(generation_class_test_losses)/float(len(generation_class_test_losses)))))
                    if generation_false_class_test_losses: 
                        generate_false_class_tests.append((epoch, (sum(generation_false_class_test_losses)/float(len(generation_false_class_test_losses)))))
                    
                    # Testing: Storing accuracy avgs
                    if content_class_test_batch_accuracies: 
                        content_class_test_accuracies.append((epoch, (sum(content_class_test_batch_accuracies)/float(len(content_class_test_batch_accuracies)))))
                    if generate_class_test_batch_accuracies: 
                        generate_class_test_accuracies.append((epoch, (sum(generate_class_test_batch_accuracies)/float(len(generate_class_test_batch_accuracies)))))
                    if content_class_false_test_batch_accuracies: 
                        content_class_false_test_accuracies.append((epoch, (sum(content_class_false_test_batch_accuracies)/float(len(content_class_false_test_batch_accuracies)))))
                    if generate_class_false_test_batch_accuracies: 
                        generate_class_false_test_accuracies.append((epoch, (sum(generate_class_false_test_batch_accuracies)/float(len(generate_class_false_test_batch_accuracies)))))
                    
                    # Testing: STOP

                # report current state to terminal
                torch.cuda.empty_cache()
                if g_class_loss_item is not None:
                    loop.set_description('epoch:{}, gen_class_loss:{:.2f}, npi_loss:{:.2f}, time_elapsed:{:.1f}'.format(epoch, g_class_loss_item,npi_loss.item(),(time.time()-start_time)))
                else:
                    loop.set_description('epoch:{}, gen_class_loss:N/A, npi_loss:{:.2f}, time_elapsed:{:.1f}'.format(epoch, npi_loss.item(),(time.time()-start_time)))
                loop.update(1)

            print("end of regular epoch")

            if epoch % save_freq == 0 and epoch >= HEAD_START_NUM:

                # save the current version of the npi_model
                print("Saving NPI Model")
                out_path = save_file_path+"{}_npi_network_epoch{}.bin".format(npi_type, epoch) 
                torch.save(npi_model, out_path)

                print("Saving NPI Loss Summary")
                out_path = save_file_path + "{}_npi_loss_summaries_epoch{}.pkl".format(npi_type, epoch) 
                with open(out_path, 'wb') as outfile:
                    pkl.dump({"epoch_losses": npi_epoch_losses, 
                                "test_losses": npi_test_losses, 
                                "accuracies_from_content_class":content_class_false_test_accuracies, 
                                "sample_meta_data": npi_sample_meta_data, 
                             }, outfile)

                #print("Saving ContentClassifier Loss Summary") 
                #out_path = save_file_path + "{}_loss_summaries_epoch{}.pkl".format("ContentClassifier", epoch) 
                #with open(out_path, 'wb') as outfile:
                #    pkl.dump({"false_test_losses": content_false_class_tests, 
                #                "avg_test_losses": content_class_tests, 
                #                "false_test_accuracies": content_class_false_test_accuracies, 
                #                "avg_test_accuracies": content_class_test_accuracies, 
                #             }, outfile)

                print("Saving GenerationClassifier Model")
                out_path = save_file_path+"{}_network_epoch{}.bin".format('GenerationClassifier', epoch) 
                torch.save(generate_class_model, out_path)

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

                with open(save_file_path+"{}_averages_for_visualization_plots.pkl".format('NPI'), 'wb') as outfile:
                    pkl.dump({'avg_epoch_train_losses':npi_avg_epoch_train_losses, 
                              'avg_epoch_test_losses':npi_avg_epoch_test_losses, 
                              'test_epochs':npi_test_epochs, 
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

                with open(save_file_path+"{}_averages_for_visualization_plots.pkl".format('ContentClassifier'), 'wb') as outfile:
                    pkl.dump({'avg_epoch_train_losses':content_class_avg_epoch_train_losses, 
                              'avg_epoch_test_losses':content_class_avg_epoch_test_losses, 
                              'avg_epoch_false_test_losses':content_class_avg_epoch_false_test_losses, 
                              'avg_epoch_train_accuracies':content_class_avg_epoch_train_accuracies, 
                              'avg_epoch_test_accuracies':content_class_avg_epoch_test_accuracies, 
                              'avg_epoch_false_test_accuracies':content_class_avg_epoch_false_test_accuracies, 
                              'test_epochs':content_test_epochs, 
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

                with open(save_file_path+"{}_averages_for_visualization_plots.pkl".format('GenerationClassifier'), 'wb') as outfile:
                    pkl.dump({'avg_epoch_train_losses':gen_class_avg_epoch_train_losses, 
                              'avg_epoch_test_losses':gen_class_avg_epoch_test_losses, 
                              'avg_epoch_false_test_losses':gen_class_avg_epoch_false_test_losses, 
                              'avg_epoch_train_accuracies':gen_class_avg_epoch_train_accuracies, 
                              'avg_epoch_test_accuracies':gen_class_avg_epoch_test_accuracies, 
                              'avg_epoch_false_test_accuracies':gen_class_avg_epoch_false_test_accuracies, 
                              'test_epochs':gen_test_epochs, 
                              }, outfile)

                print("Saving Data Visualizations: STOP")
            torch.cuda.empty_cache()
            loop.close()

        print("SAVING META-DATA AFTER FULL TRAINING - NPI AND CLASSIFIER RETURNED (TO MAIN), NOT SAVED")
        out_path = save_file_path + "{}_loss_summaries_final.pkl".format(npi_type)
        with open(out_path, 'wb') as outfile: 
            pkl.dump({"epoch_losses": npi_epoch_losses, 
                        "test_losses": npi_test_losses, 
                        "accuracies_from_content_class":content_class_false_test_accuracies, 
                        "sample_meta_data": npi_sample_meta_data, 
                        }, outfile)

        #print("Saving ContentClassifier Loss Summary") 
        #out_path = save_file_path + "{}_loss_summaries_final.pkl".format("ContentClassifier")
        #with open(out_path, 'wb') as outfile:
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

        with open(save_file_path+"{}_final_averages_for_visualization_plots.pkl".format('NPI'), 'wb') as outfile:
            pkl.dump({'avg_epoch_train_losses':npi_avg_epoch_train_losses, 
                        'avg_epoch_test_losses':npi_avg_epoch_test_losses, 
                        'test_epochs':npi_test_epochs, 
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

        with open(save_file_path+"{}_final_averages_for_visualization_plots.pkl".format('ContentClassifier'), 'wb') as outfile:
            pkl.dump({'avg_epoch_train_losses':content_class_avg_epoch_train_losses, 
                        'avg_epoch_test_losses':content_class_avg_epoch_test_losses, 
                        'avg_epoch_false_test_losses':content_class_avg_epoch_false_test_losses, 
                        'avg_epoch_train_accuracies':content_class_avg_epoch_train_accuracies, 
                        'avg_epoch_test_accuracies':content_class_avg_epoch_test_accuracies, 
                        'avg_epoch_false_test_accuracies':content_class_avg_epoch_false_test_accuracies, 
                        'test_epochs':content_test_epochs, 
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

        with open(save_file_path+"{}_final_averages_for_visualization_plots.pkl".format('GenerationClassifier'), 'wb') as outfile:
            pkl.dump({'avg_epoch_train_losses':gen_class_avg_epoch_train_losses, 
                        'avg_epoch_test_losses':gen_class_avg_epoch_test_losses, 
                        'avg_epoch_false_test_losses':gen_class_avg_epoch_false_test_losses, 
                        'avg_epoch_train_accuracies':gen_class_avg_epoch_train_accuracies, 
                        'avg_epoch_test_accuracies':gen_class_avg_epoch_test_accuracies, 
                        'avg_epoch_false_test_accuracies':gen_class_avg_epoch_false_test_accuracies, 
                        'test_epochs':gen_test_epochs, 
                        }, outfile)

        print("Saving Data Visualizations: STOP")
    
        print("Epoch loss history == ", npi_epoch_losses)
        gc.collect()
        epoch_losses_cleaned = [x for x in npi_epoch_losses if x is not np.nan]
        return npi_model, content_class_model, generate_class_model, np.mean(epoch_losses_cleaned)

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
                        default=None, 
                        help="/path/to/training/dataset/")
    parser.add_argument("--npi-lr", 
                        type=float, 
                        default=1e-6, # -- CHANGED from 1e-4 05/11/2020 at 10:07pm
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
    parser.add_argument("--gpu-num", # NOTE: not implemented fully
                        type=int,
                        default=0,
                        help="Which GPU to use")
    parser.add_argument("--discrim-coeff", # gamma
                        type=float,
                        default=3.0,
                        help="Discriminator (or 'Generation Classifer') loss coefficient")
    parser.add_argument("--style-coeff", # alpha
                        type=float,
                        default=10.0,
                        help="Content classifier loss coefficient")
    parser.add_argument("--similarity-coeff", # beta
                        type=float,
                        default=1.0,
                        help="MSE similarity loss coefficient")
    parser.add_argument("--head-start-num",
                        type=int,
                        default=5,
                        help="Give the NPI this many epochs of head start on the discriminator")
    pparser.add_argument("--perturbation-indices",
                        type=str,
                        default="5,11",
                        help="indices for layers to extract from language model activations: string of numbers separated by commas")
    parser.add_argument("--max-seq-len",
                        type=int,
                        default=10,
                        help="Length of tokens list to pass through language model")
    parser.add_argument("--max-seq-iters",
                        type=int,
                        default=10,
                        help="Number of times to run text through language model iteratively")

    parser.add_argument("--no-cuda", action='store_true',
                        help="Avoid using CUDA when available")

    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = torch.cuda.device_count()
    torch.cuda.empty_cache()

    #discrim_coeffs = [args.discrim_coeff] # for grid search
    #style_coeffs = [args.style_coeff] 
    #similarity_coeffs = [args.similarity_coeff] 

    # format perturbation indices correctly
    args.perturbation_indices = [int(pi) for pi in args.perturbation_indices.split(',')]
    # construct file directory suffix
    dir_suffix = ""
    for pi in pis:
        dir_suffix = dir_suffix + "_" + str(pi)

    #best_min_epoch_loss = None # These var's for grid search
    #best_discrim_coeff = None
    #best_style_coeff = None
    #best_similarity_coeff = None

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

    #grid_search_iter = 0
    if True: #for coeffs in coeffs_hyperparam_list: # Commented out because we don't want a grid search
        if True:#for layers in layers_hyperparam_list:

            args.save_file_path = orig_save_file_path + "params_discco{}_styco{}_simco{}_layers{}/".format(args.discrim_coeff, args.style_coeff, \
                                                                                args.similarity_coeff, dir_suffix)
            # Finish making save directory
            if not os.path.exists(args.save_file_path):
                os.mkdir(args.save_file_path)

            try:
                torch.cuda.empty_cache()
                # Main event:
                npi_model, content_classifier_model, generation_classifier_model, avg_epoch_loss = train_adversarial_NPI(args)
            
                out_path = args.save_file_path+"{}_npi_vfinal.bin".format(args.npi_type)
                torch.save(npi_model, out_path)
                out_path = args.save_file_path+"content_classifier_vfinal.bin"
                torch.save(content_classifier_model, out_path)
                out_path = args.save_file_path+"generation_classifier_vfinal.bin"
                torch.save(generation_classifier_model, out_path)

                print("Avg epoch loss == ", avg_epoch_loss)

                del npi_model
                del content_classifier_model
                del generation_classifier_model

            except:
                raise

            finally:
                torch.cuda.empty_cache()
    print("DONE!!!")
    pass

