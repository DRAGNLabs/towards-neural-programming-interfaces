#           Data that we can use to train           #
#               our classifeir and NPI              #
#                                                   #
#          Fulda, Brown, Wingate, Robinson          #
#                       DRAGN                       #
#                    NPI Project                    #
#                       2020                        #

import pandas as pd
import argparse
import pickle as pkl
import copy as cp

import numpy as np

import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm
import faulthandler # Debugging segfaults

from src.utils import top_k_top_p_filtering

# define how sentence label vectors shall be indexed
FAKE_DATA_INDEX = 0
UNK_LABEL_INDEX = -2  # 1 + (num_word_categories * num_in_word_category)
GPT2_MODEL_INDEX = -1
# define how each data point in the data set will be indexed
ORIG_ACTIV_INDEX = 0  # activation arrays concatenated
ORIG_LABEL_INDEX = 1  # label of output text
TARG_LABEL_INDEX = 2  # this entry no longer used in our implementation
LANG_MODEL_INDEX = 3  # pretrained model name
META_DATA_INDEX = 4  # relevant meta data including text tokens
ORIG_TEXT_INDEX = 5  # input text that yields lang model output text
PRED_TEXT_INDEX = 6  # this entry no longer used in our implementation
TARG_TEXT_INDEX = 7  # target term/behavior
GPT2_TEXT_INDEX = 8  # the text of what the lang model actually produced

"""
The output of the GPT-2 model is a tuple of length 3
Last element of tuple is all_hidden_states, a list of length 25 for medium GPT-2 (13 for small)
Each element of all_hidden_states is a hidden-state tensor of size (1,seq_length,emb_dimension)
For many many sentences:
    Take seq_length-word sentence
    For i in range(num_iters):
        Pass sentence through GPT2
        Take selected indices of hidden states and concatenate them to BIG_ARRAY
        Get next predicted token from GPT2 and append to sentence
        Make sentence length seq_lenth again by removing first token
Then BIG_ARRAY is size (1,num_iters*seq_length*len(selected_indices),num_possible_labels)
    but we may want to reshape it to be ((1,num_iters*seq_length*len(selected_indices), num_possible_labels, 1)

"""


def construct_sentiment_data(model_layers, data_iter, data_len, pkl_name, model_name, gpu_device, window_size, num_iters, num_chunks):
    """Construct data based on the parameters given. There are lots
        num_chunks will determine the maximum number of .pkl data files to be generated
        of course you can always kill the process once you feel you have enough data
    """
    # Some very important constants
    PRED_INDICES = model_layers
    PRED_INDICES = [int(pi) for pi in PRED_INDICES]

    # a pkl file should only be so big for loading speed
    num_sentences_per_chunk = 4000 // len(PRED_INDICES)
    max_iters = 5 * num_iters  # default 50
    assert max_iters >= num_iters
    top_k = 1
    top_p = .9

    # Fix pkl_name:
    if ".pkl" not in pkl_name:
        pkl_name = pkl_name + ".pkl"
    pkl_name_base = pkl_name

    model = None
    tokenizer = None
    if 'gpt2' in model_name:
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    else:
        raise NotImplementedError(
            "model_name == {} not supported".format(model_name))

    # necessary to pull activation tensors
    model.transformer.output_hidden_states = True
    device = gpu_device
    if torch.cuda.is_available():
        print(F"Using cuda:{torch.cuda.current_device()}")
        print(F"gpu_device is {device}")
        model = model.cuda(gpu_device)
        

    try:
        # BEGINNING ##############################################################################################################################
        print("and so it begins", flush=True)

        """Now we begin the loop of a lifetime...---...---...---...---...---...---...---...---...---...---"""
        iterator = -1
        pkl_counter = 0
        dataset = []
        faulthandler.enable()
        pbar = tqdm(data_iter, total=data_len, mininterval=10, maxinterval=30, miniters=100)
        for line, score in pbar:

            # clean line to some extent
            #   (due to possible differences in corpora that could tip off the classifer)
            line = line.lower().strip().strip('.').strip()
            if len(line.split()) > 100 or len(line.split()) < 4:
                continue

            iterator += 1

            append_to_dataset = True  # naively assume we're gonna append ha ha so naive

            big_array = []  # nxmx1

            tokens = tokenizer.encode(line)
            tokens = tokens[-window_size:]
            num_tokens_needed = window_size - len(tokens)
            tokens = torch.tensor(tokens, dtype=torch.long, device=device)
            tokens = tokens.unsqueeze(0).repeat(1, 1)
            tokens = tokens.cuda(gpu_device)
            all_text_tokens = cp.deepcopy(tokens)

            # some constants to set first
            len_for_big_array = len(PRED_INDICES) * num_iters
            stop_itern = num_tokens_needed + max_iters

            # We loop through multiple times now
            purely_generated_tokens = []  # haven't generated anything yet
            i = -1
            # And a few other things we need defined outside the loop
            while True:
                i += 1

                # Now run the model
                hidden_states, _, all_hiddens = model(
                    input_ids=tokens[:, -window_size:])  # all_hiddens is a list of len
                # 25 or 13 with tensors of shape (gpt2 medium of small)
                # (1,sent_len,1024) or (1,sent_len,768)
                # Add to big_array
                if tokens.shape[1] >= window_size:
                    for pi in PRED_INDICES:
                        big_array.append(all_hiddens[pi].data)

                # Now we extract the new token and add it to the list of tokens
                next_token_logits = hidden_states[0, -1, :]
                filtered_logits = top_k_top_p_filtering(
                    next_token_logits, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(
                    F.softmax(filtered_logits, dim=-1), num_samples=1)
                next_token_list = next_token.tolist()
                purely_generated_tokens = purely_generated_tokens + next_token_list

                # ...update list of tokens
                if tokens.shape[1] < window_size:
                    tokens = torch.cat(
                        (tokens, next_token.unsqueeze(0)), dim=1).cuda(device)
                else:
                    tokens = torch.cat(
                        (tokens[:, 1:], next_token.unsqueeze(0)), dim=1).cuda(device)

                all_text_tokens = torch.cat(
                    (all_text_tokens, next_token.unsqueeze(0)), dim=1).cuda(device)

                if len(big_array) >= len_for_big_array and i >= stop_itern and len(
                        all_text_tokens.squeeze()) >= num_iters + window_size:
                    break  # if we have a target word and enough arrays

            num_gpt2_iters_run = i + 1
            big_array = big_array[-len_for_big_array:]

            # figure out true classification
            orig_classification = np.zeros(2)

            # refer to .csv for classification
            append_to_dataset = True
            # 0 || 1 means negative classification, 3 || 4 means positive classification
            if score == 0:
                orig_classification[0] = 1
            elif score == 1:
                orig_classification[1] = 1
            else:
                raise RuntimeError("Got a score that is not 0 or 1")

            # What will we call "original text" and "generated text"

            assert all_text_tokens.squeeze().tolist(
            )[-window_size:] == tokens.squeeze().tolist()

            orig_text_tokens = all_text_tokens[:,
                                               -window_size - num_iters:-num_iters]  # sent_len tokens that produced generated_text_tokens
            generated_text_tokens = tokens

            orig_tokens = orig_text_tokens.squeeze().tolist()
            gpt2_generated_tokens = generated_text_tokens.squeeze().tolist()

            orig_text = tokenizer.decode(orig_tokens)
            gpt2_generated_text = tokenizer.decode(gpt2_generated_tokens)

            # Now the big_array is a list of length (num_iters*len(PRED_INDICES)) of tensors with shape (1,sent_len,emb_dim)
            big_array = torch.cat(big_array, dim=1)
            big_array = big_array.permute(1, 2,
                                          0)  # shape is (2*sent_len*num_iters, emb_dim, 1) now, emb_dim will be 1024 or 768
            big_array = big_array.data.cpu().numpy()

            # We want to save this big_array in the data
            # ORIG_ACTIV_INDEX = 0
            # ORIG_LABEL_INDEX = 1
            # TARG_LABEL_INDEX = 2
            # LANG_MODEL_INDEX = 3
            # META_DATA_INDEX = 4
            # ORIG_TEXT_INDEX = 5
            # PRED_TEXT_INDEX = 6
            # TARG_TEXT_INDEX = 7
            # GPT2_TEXT_INDEX = 8
            if append_to_dataset:
                datum = [
                    big_array,  # ORIG ACTIV
                    orig_classification,  # ORIG LABEL
                    None,  # this no longer used
                    model_name,  # LANG MODEL: model_name an abstraction for 'gpt2'
                    {'num_gpt2_iters': num_gpt2_iters_run, \
                        'orig_tokens': orig_tokens, \
                        'gpt2_generated_tokens': gpt2_generated_tokens},  # META DATA
                    # ORIG TEXT (or what we're deeming 'originial text')
                    orig_text,
                    None,  # PRED TEXT this literally won't exist until we have an NPI
                    None,  # TARG TEXT Changed here.
                    gpt2_generated_text  # GPT2 TEXT: just generated right here by the GPT2 :D :D
                ]
                dataset.append(datum)

            # Check data len
            if len(dataset) >= num_sentences_per_chunk:
                # Then we want to save it
                # pkl stuff
                pkl_name = pkl_name_base + "_" + str(pkl_counter)
                with open(pkl_name, 'wb') as f:
                    pkl.dump(dataset, f)

                # More business to conduct every num_sentences_per_chunk data
                del dataset
                dataset = []
                pkl_counter += 1
                pbar.set_description(F"pkls written: {pkl_counter}")

            # Now if we have all the pickles we need... we break! (bc we done yay)
            if pkl_counter == num_chunks:
                break  # :)

    except:
        raise

    finally:
        torch.cuda.empty_cache()

    # shuffle now
    # random.shuffle(word_sent_list)
    torch.cuda.empty_cache()
    print(" ", flush=True)
    print("done")


if __name__ == "__main__":
    """
    Cycles through list of sentences provided by pkl files, making a big_array of hidden states
    for each sentence.
    For each sentence we perform:
        Take sentence
        For i in range(num_iters):
            Pass sentence through GPT2
            Take hidden states and concatenate them to big_array
            Get next predicted token from GPT2 and append to sentence
            Make sentence length sent_len again by removing first token

    Params:
        num_sentences (int): max number of sentences to get data for
        sent_len (int): sentence length
        num_iters (int): number of words to add onto the sentence cyclically
        pkl_name (str): name for pkl to which we save big_arrays
    Returns:
        (dict): dictionary mapping sentences to big_array's
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixed-sentence-file",
                        default="trainingandtest/training.1600000.processed.noemoticon.csv",
                        help="corpus from which to pull sentences; may be mixed with sentences that display target behavior to increase likelihood of production from GPT-2 and possibly expedite data set production"

                        )
    parser.add_argument("--save-pkl",
                        default="data/sentence_arrays.pkl",
                        help="base name of pickle files to write data set to"
                        )
    parser.add_argument("--pretrained-model",
                        default="gpt2",
                        help="pretrained model to use. For small GPT-2 use 'gpt2' and for medium GPT-2 use 'gpt2-medium'"
                        )
    parser.add_argument("--model-layers",
                        default="0,1,2,3,4,5,6,7,8,9,10,11,12",
                        help="Which layers to extract from language model? layer indices separated by commas\nRecommended: if spacial restrictions allow, use all available layers for data set generation and extract the needed layers at training using the extract_needed_layers function"
                        )
    parser.add_argument("--seq-len",
                        type=int,
                        default=10,
                        help="window size for inputs to lang model"
                        )
    parser.add_argument("--num-iters",
                        type=int,
                        default=10,
                        help="number of times to run lang model forward pass (extracting layers each time)"
                        )

    args = parser.parse_args()

    # Shortcuts in arguments
    mixed_sentence_file = args.mixed_sentence_file
    pkl_name = args.save_pkl
    pretrained_models = [args.pretrained_model]

    with open(mixed_sentence_file, 'r', newline='') as BIG_FILE:
        data = pd.read_csv(BIG_FILE)
        filereader = data.sample(frac=1)
        construct_sentiment_data(args.model_layers.split(
            ','), filereader, pkl_name, pretrained_models[0], args.seq_len, args.num_iters)

"""
"""
