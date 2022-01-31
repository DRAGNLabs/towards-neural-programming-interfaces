#           Data that we can use to train           #
#               our classifeir and NPI              #
#                                                   #
#          Fulda, Brown, Wingate, Robinson          #
#                       DRAGN                       #
#                    NPI Project                    #
#                       2020                        #

import argparse
import pickle as pkl
import random
import copy as cp

import numpy as np
# for NLP
import spacy
import torch
import torch.nn.functional as F
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from utils import top_k_top_p_filtering

nlp = spacy.load("en_core_web_sm")

# USER ADDITION
import pandas as pd

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
    parser.add_argument("--target-words",
                        default="sexist_slurs",
                        help="words to target, separated by commas; e.g. 'cat,dog,mouse'\n'sexist_slurs', 'positive_terms', and 'negative_terms' are special values for this argument"
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
    # we have a default special term_list
    if args.target_words == 'sexist_slurs':
        with open("data/sexist_terms.txt", 'r') as f:
            args.target_words = f.read().strip()

	# also support positive or negative terms
    #if args.target_words == 'positive_terms':
    #    with open("data/positive_terms.txt", 'r') as f:
    #        args.target_words = f.read().strip()
    #if args.target_words == 'negative_terms':
    #    with open("data/negative_terms.txt", 'r') as f:
    #        args.target_words = f.read().strip()

    term_list = args.target_words.split(',')  # list of target words
    term_list = [term.lower().strip() for term in term_list]  # clean words
    while '' in term_list:  # we don't want any '' empty strings
        term_list.remove('')

    # Some very important constants
    TARG = ['target words']
    WORDS = {'target words': term_list}
    PRED_INDICES = args.model_layers.split(',')
    PRED_INDICES = [int(pi) for pi in PRED_INDICES]

    # define variables for determining text processing

    # num_checks will determine the maximum number of .pkl data files to be generated
    #   of course you can always kill the process once you feel you have enough data
    #num_chunks = 25 * len(pretrained_models) * len(PRED_INDICES)
    #num_chunks = 57  # NOTE: This just for the default sexist slur NPI training, for which only 57 pickles are needed. Suggested to comment out
    num_chunks = 20
    # for other applications and just interrupt the process when you have a good amount of data
    # (depending on your machinery, time constraints)
    num_sentences_per_chunk = 4000 // len(PRED_INDICES)  # a pkl file should only be so big for loading speed
    num_sentences = num_chunks * num_sentences_per_chunk
    sent_len = args.seq_len
    num_iters = args.num_iters
    max_iters = 5 * num_iters  # default 50
    assert max_iters >= num_iters
    top_k = 1
    top_p = .9
    temperature = 1
    # define how sentence label vectors shall be indexed
    FAKE_DATA_INDEX = 0
    UNK_LABEL_INDEX = -2  # 1 + (num_word_categories * num_in_word_category)
    GPT2_MODEL_INDEX = -1
    # optional_s
    OPTIONAL_S = True  # n8 HACK

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

    # params to inject the word randomly into inputs to encourage its output
    INJECT_WORDNESS = False
    INJECT_WORD_RAND_CHANGES = False  # this one should likely be True if the first one is

    # Fix pkl_name:
    if ".pkl" not in pkl_name:
        pkl_name = pkl_name + ".pkl"
    pkl_name_base = pkl_name

    # Create tokenizers
    model_name = pretrained_models[0]
    gpt2_tokenizer = None
    if pretrained_models == [model_name]:
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    else:
        raise NotImplementedError("Only the following tokenizers are supported: {}".format(model_name))

    num_keywords = len(TARG)
    num_possible_labels = int(1 + num_keywords)

    model = None
    tokenizer = None
    if 'gpt2' in model_name:
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = gpt2_tokenizer
    else:
        raise NotImplementedError("model_name == {} not supported".format(model_name))

    model.transformer.output_hidden_states = True  # necessary to pull activation tensors
    device = torch.device("cpu")
    if torch.cuda.is_available():
        model = model.cuda()
        device = torch.device("cuda")

    try:
        # BEGINNING ##############################################################################################################################

        print("and so it begins", flush=True)
        dataset = []

        word_to_toks = {}
        word_to_toks[TARG[0]] = []
        for word in WORDS[TARG[0]]:
            word = word.lower()
            gpt2_word_toks = []
            gpt2_word_toks.append(gpt2_tokenizer.encode(word))  # list of veriable len but likely len 1 for small words
            gpt2_word_toks.append(gpt2_tokenizer.encode("this is " + word)[2:])  # token differs in context
            if OPTIONAL_S:  # for some verbs or nouns we may put an optional 's' on the end
                gpt2_word_toks.append(gpt2_tokenizer.encode(word + "s"))
                gpt2_word_toks.append(gpt2_tokenizer.encode("this is " + word + "s")[2:])
            word_to_toks[word] = gpt2_word_toks

            word_to_toks[TARG[0]] = word_to_toks[TARG[0]] + word_to_toks[word]

        # We want to count how many words we got :)
        word_counts = {}
        for word in TARG:
            word_counts[word] = 0
        word_counts['UNK'] = 0

        # And a few other things we need defined outside the loop
        pkl_counter = 0
        iterator = -1

        """Now we begin the loop of a lifetime...---...---...---...---...---...---...---...---...---...---"""

        with open(mixed_sentence_file, 'r', newline='') as BIG_FILE:
            data = pd.read_csv(BIG_FILE)
            filereader = data.sample(frac=1)

            for index, row in filereader.iterrows():
                # get score and last string
                score = int(row[0])
                line = row[-1]

                # clean line to some extent
                #   (due to possible differences in corpora that could tip off the classifer)
                line = line.lower().strip().strip('.').strip()
                if len(line.split()) > 100 or len(line.split()) < 4:
                    continue

                # Will we inject a word?
                if INJECT_WORD_RAND_CHANGES:
                    INJECT_WORDNESS = random.choice([True, False])

                iterator += 1

                append_to_dataset = True  # naively assume we're gonna append ha ha so naive

                big_array = []  # nxmx1

                tokens = gpt2_tokenizer.encode(line)
                tokens = tokens[-sent_len:]
                num_tokens_needed = sent_len - len(tokens)
                tokens = torch.tensor(tokens, dtype=torch.long, device=device)
                tokens = tokens.unsqueeze(0).repeat(1, 1)
                tokens = tokens.cuda()
                all_text_tokens = cp.deepcopy(tokens)
                ogog_tokens = cp.deepcopy(tokens)

                # some constants to set first
                found_words_dict = {}
                for word in TARG:
                    found_words_dict[word] = False
                len_for_big_array = len(PRED_INDICES) * num_iters
                stop_itern = num_tokens_needed + max_iters

                word_found_already = False  # This will tell us if we've found a word yet
                index_of_last_injection = -1  # This is for keeping track of word injection

                # We loop through multiple times now
                purely_generated_tokens = []  # haven't generated anything yet
                i = -1
                while True:
                    i += 1

                    # Now run the model
                    hidden_states, presents, all_hiddens = model(
                        input_ids=tokens[:, -sent_len:])  # all_hiddens is a list of len
                    # 25 or 13 with tensors of shape (gpt2 medium of small)
                    # (1,sent_len,1024) or (1,sent_len,768)
                    # Add to big_array
                    if tokens.shape[1] >= sent_len:
                        for pi in PRED_INDICES:
                            big_array.append(all_hiddens[pi].data)

                    # Now we extract the new token and add it to the list of tokens
                    next_token_logits = hidden_states[0, -1, :]
                    filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                    next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                    next_token_list = next_token.tolist()
                    next_word = tokenizer.decode(next_token_list)
                    purely_generated_tokens = purely_generated_tokens + next_token_list
                    # generated_sent = generated_sent + next_word + " "

                    # check if the next_token_list is the token we are looking for!!!
                    next_token_item = next_token_list[0]
                    generated_string = tokenizer.decode(purely_generated_tokens)
                    for word in TARG:
                        for subword in WORDS[word]:
                            if subword in generated_string:
                                found_words_dict[word] = True
                                word_found_already = True
                                time_since_last_injection = i - index_of_last_injection  # Let's make the target word be in the middle of the generated text
                                number_of_indices_more_we_need = num_iters - time_since_last_injection
                                stop_itern = i + number_of_indices_more_we_need  # this will ensure there are no injections in what we call the generated tokens

                    # ...update list of tokens
                    if tokens.shape[1] < sent_len:
                        tokens = torch.cat((tokens, next_token.unsqueeze(0)), dim=1).cuda()
                    else:
                        tokens = torch.cat((tokens[:, 1:], next_token.unsqueeze(0)), dim=1).cuda()

                    all_text_tokens = torch.cat((all_text_tokens, next_token.unsqueeze(0)), dim=1).cuda()

                    if len(big_array) >= len_for_big_array and i >= stop_itern and len(
                            all_text_tokens.squeeze()) >= num_iters + sent_len:
                        break  # if we have a target word and enough arrays

                    if i >= num_tokens_needed + max_iters and not word_found_already:
                        break

                    if INJECT_WORDNESS and i > num_tokens_needed and (
                            i - num_tokens_needed) % num_iters == 0 and not word_found_already:
                        word = random.choice(TARG)
                        tokens_to_inject = random.choice(word_to_toks['target words'])
                        num_tokens_to_inject = len(tokens_to_inject)
                        tokens_to_inject = torch.tensor(tokens_to_inject).long().unsqueeze(0).repeat(1, 1).cuda()
                        # Now we want to change the sentence a bit
                        #    iterate through the tokens and if any of them are nouns, then replace
                        #    NOTE: it is also valid to replace something other than nouns if that seems more reasonable :)
                        idx_to_replace = None
                        for tok_idx in range(tokens.shape[1]):
                            token_in_question = tokens.squeeze().tolist()[tok_idx]
                            word_in_question = tokenizer.decode([token_in_question]).strip()
                            assert type(word_in_question) is str
                            # Now we want to see if it's a noun
                            doc = nlp(word_in_question)
                            pos = [token.pos_ for token in doc]
                            if pos == ['NOUN'] and len(word_in_question) > 2:  # we have a viable noun
                                # we need this conditions because spacy thinks single letters are nouns
                                idx_to_replace = tok_idx

                        if idx_to_replace is not None:  # we actually found a NOUN to replace
                            # We knock off the first few tokens if num_tokens_to_inject > 1
                            tokens = torch.cat((tokens[:, num_tokens_to_inject - 1:idx_to_replace], tokens_to_inject,
                                                tokens[:, idx_to_replace + 1:]), dim=1)
                            tokens = tokens[:, -sent_len:]  # make sure it is the right length

                            idx_to_replace_in_all_toks = idx_to_replace - tokens.shape[1] + all_text_tokens.shape[1]
                            all_text_tokens = torch.cat((all_text_tokens[:, :idx_to_replace_in_all_toks],
                                                         tokens_to_inject,
                                                         all_text_tokens[:, idx_to_replace_in_all_toks + 1:]),
                                                        dim=1)  # all_text_tokens[-1,idx_to_replace_in_all_toks] = token_to_inject
                            # keep track of when this happened
                            index_of_last_injection = i

                num_gpt2_iters_run = i + 1
                big_array = big_array[-len_for_big_array:]

                # figure out true classification
                orig_classification = np.zeros(len(TARG) + 1)

                # count words and see if we should append to dataset
                #for i_word, word in enumerate(TARG):
                #    if found_words_dict[
                #        word]:  # means we found this word: so this is a term-postive labeled data point!
                #        # Label: [1, 0]
                #        orig_classification[i_word] = 1.
                #        word_counts[word] = word_counts[word] + 1
                #        # then check if we should append or not
                #        if word_counts[word] > num_sentences:
                #            append_to_dataset = False
                #if True not in list(found_words_dict.values()):  # means this is a term-negative labeled data point!
                #    # Label: [0, 1]
                #    orig_classification[i_word + 1] = 1.
                #    word_counts['UNK'] = word_counts['UNK'] + 1
                #    if word_counts['UNK'] > 1.2 * max(
                #            [word_counts[word] for word in TARG]) + 1:  # keep the UNKs down!!! We want balance!!!
                #        append_to_dataset = False
                #        word_counts['UNK'] = word_counts[
                #                                 'UNK'] - 1  # so we actually won't count this one since we're not appending

                # refer to .csv for classification
                append_to_dataset = True
                # 0 || 1 means negative classification, 3 || 4 means positive classification
                if score==0 or score==1:
                #if score==3 or score==4:
                    orig_classification[0] = 1
                else:
                    orig_classification[1] = 1

                # What will we call "original text" and "generated text"

                assert all_text_tokens.squeeze().tolist()[-sent_len:] == tokens.squeeze().tolist()

                orig_text_tokens = all_text_tokens[:,
                                   -sent_len - num_iters:-num_iters]  # sent_len tokens that produced generated_text_tokens
                generated_text_tokens = tokens

                orig_tokens = orig_text_tokens.squeeze().tolist()
                gpt2_generated_tokens = generated_text_tokens.squeeze().tolist()

                orig_text = gpt2_tokenizer.decode(orig_tokens)
                gpt2_generated_text = gpt2_tokenizer.decode(gpt2_generated_tokens)

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
                        orig_text,  # ORIG TEXT (or what we're deeming 'originial text')
                        None,  # PRED TEXT this literally won't exist until we have an NPI
                        TARG[0],  # TARG TEXT
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

                # Now if we have all the pickles we need... we break! (bc we done yay)
                if pkl_counter == num_chunks:
                    break  # :)

                # Now for helpful print statements
                if iterator % (num_sentences_per_chunk // (num_sentences_per_chunk // 100)) == 0:
                    print("iterations: {}; target words data: {}/{}, generic data: {}/{}, pkls written: {}".format(
                        iterator + 1, word_counts['target words'], num_sentences // 2, word_counts['UNK'],
                        num_sentences // 2, pkl_counter), \
                        flush=True)

    except:
        raise

    finally:
        torch.cuda.empty_cache()

    # shuffle now
    # random.shuffle(word_sent_list)
    torch.cuda.empty_cache()
    print(" ", flush=True)
    print("done")

"""
"""
