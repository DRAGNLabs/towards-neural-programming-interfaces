#           Data that we can use to train           #
#               our controller network              #
#                                                   #
#              Fulda, Brown,  Robinson              #
#                       DRAGN                       #
#                    NPI Project                    #
#                     May 2020                      #

#             See file                              #

import numpy as np 
import re
import pickle as pkl
import copy
import pdb
import gc
import random

from transformers import *
import run_generation as rg

import torch
import torch.nn.functional as F

from tqdm import trange
from tqdm import tqdm

import argparse
import os
import pdb
import copy as cp

# for NLP
import spacy
nlp = spacy.load("en_core_web_sm")

"""
The stuf returned by model is a tuple of length 3
Last element of tuple is all_hidden_states, a list of length 25
Each element of all_hidden_states is a hidden-state tensor of size (1,seq_length,num_possible_labels)
For many many sentences:
    Take 20-word sentence
    For i in range(20):
        Pass sentence through GPT2
        Take first and last hidden states and concatenate them to BIG_ARRAY
        Get next predicted token from GPT2 and append to sentence
        Make sentence length 20 again by removing first token
Then BIG_ARRAY is size (1,800,num_possible_labels) (but we may want to reshape it to be (800, num_possible_labels, 1))

See terminal for all the code I typed to do this minus the loops
"""

if __name__ == "__main__":
    """
    Cycles through list of sentences provided by pkl files, making a big_array of hidden states
    for each sentence.
    For each sentence we perform:
        Take sentence
        For i in range(num_iters):
            Pass sentence through GPT2
            Take first and last hidden states and concatenate them to big_array
            Get next predicted token from GPT2 and append to sentence
            Make sentence length sent_len again by removing first token
    Then big_array is size (1,2*sent_len*num_iters,1024) 
            (but we may want to reshape it to be (2*sent_len*num_iters, 1024, 1))

    Params:
        num_sentences (int): max number of sentences to get data for
        sent_len (int): sentence length 
        num_iters (int): number of words to add onto the sentence cyclically
        pkl_name (str): name for pkl to which we save big_arrays
    Returns:
        (dict): dictionary mapping sentences to big_array's
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mixed_sentence_file", 
                        default="/mnt/pccfs/not_backed_up/data/text_corpora/wiki_books_reddit_shuffled.txt",

                        )
    #parser.add_argument("--sentence_content_label_file", 
    #                    default="sentence_content_label_CAT.pkl", # n8 HACK
    #                    )
    #parser.add_argument("--processed_sentence_content_label_file", 
    #                    default="processed_sentence_content_labels_02142020.pkl", 
    #                    )
    #parser.add_argument("--dataset_config_dict_file", 
    #                    default="dataset_config_dictionary.pkl", 
    #                    )
    parser.add_argument("--save_pkl", 
                        default="/raid/remote/n8rob/slurs/cat_data/CAT_smGPT2_arrays.pkl",#"/home/nate/DATA/npi_proj/CAT_OUT_smGPT2_arrays_FIXED_with_toks.pkl", 
                        )
    #parser.add_argument("--reduced_sentence_content_label_file", 
    #                    default="", 
    #                    )
    #parser.add_argument("--new_reduced_label_save_filename", 
    #                    default="reduced_sentence_content_label_file.pkl", 
    #                    )
    #parser.add_argument("--num_words_to_reduce_to",  
    #                    type=int, 
    #                    default=100, 
    #                    )

    args = parser.parse_args()

    # Shortcuts in arguments
    mixed_sentence_file = args.mixed_sentence_file
    #sentence_content_label_file = args.sentence_content_label_file
    #processed_sentence_content_label_file = args.processed_sentence_content_label_file
    #dataset_config_dict_file = args.dataset_config_dict_file
    pkl_name = args.save_pkl
    pretrained_models = ['gpt2']
    
    # dist_save_dir = "/mnt/pccfs/backed_up/zac/zac_2020/npi_gan_dataset2/"
    #local_save_dir = args.local_save_dir
    #pkl_name= local_save_dir+"data_set/data_rows"#"sentence_arrays"
    #local_pkl_name = local_save_dir+"data_set/data_rows"

    #reduced_sentence_content_label_file = args.reduced_sentence_content_label_file
    #num_words_to_reduce_to = args.num_words_to_reduce_to

    # if not os.path.exists(dist_save_dir):
    #     os.mkdir(dist_save_dir)
    # if not os.path.exists(dist_save_dir+"data_set/"):
    #     os.mkdir(dist_save_dir+"data_set/")

    #if not os.path.exists(local_save_dir):
    #    os.mkdir(local_save_dir)
    #if not os.path.exists(local_save_dir+"data_set/"):
    #    os.mkdir(local_save_dir+"data_set/")
    
    # Some very important constants
    WORDS = ['cat']
    PRED_INDICES = list(range(13))#[2, -2]

    # define variables for determining text processing
    # generic_sentence_file="wiki"
    num_chunks=25 * len(pretrained_models) * len(PRED_INDICES)
    num_sentences_per_chunk=4000//len(PRED_INDICES)
    num_sentences = num_chunks * num_sentences_per_chunk
    sent_len=10
    num_iters=10
    max_iters=50
    assert max_iters >= num_iters
    top_k=1#.0
    top_p=.9
    temperature=1
    # define how sentence label vectors shall be indexed
    FAKE_DATA_INDEX = 0
    # num_in_word_category = 335
    # num_word_categories = 3
    UNK_LABEL_INDEX = -2#1 + (num_word_categories * num_in_word_category)
    GPT2_MODEL_INDEX = -1
    # optional_s
    OPTIONAL_S = False # n8 HACK

    # define how dataset will be indexed
    ORIG_ACTIV_INDEX = 0
    ORIG_LABEL_INDEX = 1
    TARG_LABEL_INDEX = 2
    LANG_MODEL_INDEX = 3
    META_DATA_INDEX = 4
    ORIG_TEXT_INDEX = 5
    PRED_TEXT_INDEX = 6
    TARG_TEXT_INDEX = 7
    GPT2_TEXT_INDEX = 8 # the text of what the gpt2 actually produced

    # Construct target classif'n now
    TARG_CLASSIFICATION = np.ones(len(WORDS)+3)
    TARG_CLASSIFICATION[0] = 0.
    TARG_CLASSIFICATION[-2] = 0.

    # this is a special option. So special
    INJECT_WORDNESS = True
    INJECT_WORD_RAND_CHANGES = True

    # Fix pkl_name:
    if ".pkl" not in pkl_name:
        pkl_name = pkl_name + ".pkl"
    pkl_name_base = pkl_name
    #pkl_name_word_base = pkl_name_base[:-4] + "_05042020_" + ".pkl"

    # Create tokenizers
    model_name = pretrained_models[0]
    gpt2_tokenizer = None
    if pretrained_models == [model_name]:
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    else:
        raise NotImplementedError("Only the following tokenizers are supported: {}".format(model_name))
				
    num_keywords = len(WORDS)
    num_possible_labels = int(1 + num_keywords + 2) # additional labels are for fake/not-fake, UNK/not-UNK, and GPT2/not-GPT2

    model = None
    tokenizer = None
    if model_name == 'gpt2':
        model = GPT2LMHeadModel.from_pretrained(model_name)#GPT2Model.from_pretrained(model_name)
        tokenizer = gpt2_tokenizer
    else:
        raise NotImplementedError("model_name == {} not supported".format(model_name))

    model.transformer.output_hidden_states = True
    device = torch.device("cpu") 
    if torch.cuda.is_available(): 
        model = model.cuda() 
        device = torch.device("cuda") 

    try:
        # IN THE BEGINNING ##############################################################################################################################

        # komya 
        print("we begin",flush=True)
        dataset = []

        word_to_toks = {}
        for word in WORDS:
            word = word.lower()
            gpt2_word_toks = []
            gpt2_word_toks.append(gpt2_tokenizer.encode(word)) # list of len 1
            gpt2_word_toks.append(gpt2_tokenizer.encode("this is "+word)[2:]) # token different in sentence context
            # n8 komya optional_s
            if OPTIONAL_S:
                gpt2_word_toks.append(gpt2_tokenizer.encode(word+"s"))
                gpt2_word_toks.append(gpt2_tokenizer.encode("this is "+word+"s")[2:])
            word_to_toks[word] = gpt2_word_toks

        # We want to count how many words we got :)
        word_counts = {}
        for word in WORDS:
            word_counts[word] = 0
        word_counts['UNK'] = 0

        # And a few other things we need defined outside the loop 
        pkl_counter = 0

        """Now we begin the massive loop of a lifetime...---...---...---...---...---...---...---...---...---...---"""

        iterator = -1

        with open(mixed_sentence_file,'r') as BIG_FILE:

            for line in BIG_FILE:
                
                # clean line to some extent
                line = line.lower().strip().strip('.').strip()
                if len(line.split()) > 100 or len(line.split()) < 4:
                    continue

                # Will we inject a word?
                if INJECT_WORD_RAND_CHANGES:
                    INJECT_WORDNESS = random.choice([True,False])
                
                iterator += 1

                append_to_dataset = True # naively assume we gonna append ha ha ha

                big_array = [] # nxmx1

                tokens = gpt2_tokenizer.encode(line)
                tokens = tokens[-sent_len:]
                num_tokens_needed = sent_len - len(tokens)
                tokens = torch.tensor(tokens, dtype=torch.long, device=device)
                tokens = tokens.unsqueeze(0).repeat(1, 1) 
                tokens = tokens.cuda()
                all_text_tokens = cp.deepcopy(tokens)
                ogog_tokens = cp.deepcopy(tokens)

                #sent = data_row[ORIG_TEXT_INDEX]
                #generated_sent = ""

                # some constants to set first
                found_words_dict = {}
                for word in WORDS:
                    found_words_dict[word] = False
                len_for_big_array = len(PRED_INDICES) * num_iters
                stop_itern = num_tokens_needed+max_iters

                word_found_already = False # This will tell us if we've found a word yet
                index_of_last_injection = -1 # This is for keeping track of word injection

                # We loop through now
                purely_generated_tokens = []
                i = -1
                while True:
                    i += 1

                    # Now run the model
                    hidden_states, presents, all_hiddens = model(input_ids=tokens[:,-sent_len:]) # all_hiddens is a list of len
                                                                    # 25 with tensors of shape 
                                                                    # (1,15,1024), where 20 is sent_len
                    #pdb.set_trace() 
                    # Add to big_array
                    if tokens.shape[1] >= sent_len:
                        for pi in PRED_INDICES:
                            #if pi == -1 or pi == len(all_hiddens)-1:
                            #    big_array.append(all_hiddens[pi].data[:,:-1,:])
                            #else:
                            big_array.append(all_hiddens[pi].data)

                    # Now we extract the new token and add it to the list of tokens
                    next_token_logits = hidden_states[0,-1,:]
                    filtered_logits = rg.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                    next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                    next_token_list = next_token.tolist()
                    purely_generated_tokens = purely_generated_tokens + next_token_list
                    try:
                        next_word = tokenizer.decode(next_token_list)
                    except:
                        print("****",next_token_list,"****")
                        raise
                    #sent = sent + " " + next_word # we just update this so sent remains accurate for dict
                    #generated_sent = generated_sent + next_word + " "

                    # check if the next_token_list is the token we are looking for!!!
                    next_token_item = next_token_list[0]
                    for word in WORDS:
                        generated_string = tokenizer.decode(purely_generated_tokens).lower()
                        if "cat" in generated_string.lower():
                            found_words_dict[word] = True
                            word_found_already = True
                            time_since_last_injection = i - index_of_last_injection # I want cat to be in the middle of the generated text I think
                            number_of_indices_more_we_need = num_iters - time_since_last_injection
                            stop_itern = i + number_of_indices_more_we_need # this will gurantee there are no injections in what we call the generated tokens
                    
                    # ...update list of tokens
                    if tokens.shape[1] < sent_len:
                        tokens = torch.cat((tokens,next_token.unsqueeze(0)),dim=1).cuda()
                    else:
                        tokens = torch.cat((tokens[:,1:],next_token.unsqueeze(0)),dim=1).cuda()

                    all_text_tokens = torch.cat((all_text_tokens,next_token.unsqueeze(0)),dim=1).cuda()
                    
                    if len(big_array) >= len_for_big_array and i >= stop_itern and len(all_text_tokens.squeeze()) >= num_iters+sent_len:
                        break # if we have a target word and enough arrays

                    if i >= num_tokens_needed+max_iters and not word_found_already:
                        break

                    if INJECT_WORDNESS and i > num_tokens_needed and (i-num_tokens_needed)%num_iters == 0 and not word_found_already:
                        #pdb.set_trace()
                        word = random.choice(WORDS)
                        #rand_index = random.choice([1,3]) # 'cat' or 'cats'
                        tokens_to_inject = random.choice(word_to_toks['cat'])
                        num_tokens_to_inject = len(tokens_to_inject)
                        tokens_to_inject = torch.tensor(tokens_to_inject).long().unsqueeze(0).repeat(1,1).cuda()
                        # token_to_inject
                        # Now we want to change the sentence a bit
                        #    iterate through the tokens and if any of them are nouns, then replace
                        idx_to_replace = None
                        for tok_idx in range(tokens.shape[1]):
                            token_in_question = tokens.squeeze().tolist()[tok_idx]
                            word_in_question = tokenizer.decode([token_in_question]).strip()
                            assert type(word_in_question) is str
                            # Now we want to see if it's a noun
                            doc = nlp(word_in_question)
                            pos = [token.pos_ for token in doc]
                            if pos == ['NOUN'] and len(word_in_question)>2 and word != word_in_question and word+'s' != word_in_question: # we have a viable noun
                                # we need all these conditions because spacy thinks single letters are nouns
                                idx_to_replace = tok_idx
                            #pdb.set_trace()

                        if idx_to_replace is not None: # we actually found a NOUN to replace
                            tokens = torch.cat((tokens[:,num_tokens_to_inject-1:idx_to_replace], tokens_to_inject, tokens[:,idx_to_replace+1:]),dim=1) #tokens[0,idx_to_replace] = tokens_to_inject
                            tokens = tokens[:,-sent_len:]
                            idx_to_replace_in_all_toks = idx_to_replace - tokens.shape[1] + all_text_tokens.shape[1]
                            all_text_tokens = torch.cat((all_text_tokens[:,:idx_to_replace_in_all_toks], tokens_to_inject, all_text_tokens[:,idx_to_replace_in_all_toks+1:]),dim=1)#all_text_tokens[-1,idx_to_replace_in_all_toks] = token_to_inject
                            # keep track of when this happened
                            index_of_last_injection = i
                        #pdb.set_trace()
                    """if INJECT_WORDNESS and i > num_tokens_needed and (i-num_tokens_needed)%num_iters == 0 and not word_found_already:
                        #pdb.set_trace()
                        word = random.choice(WORDS)
                        rand_index = random.choice([1,3]) # 'cat' or 'cats'
                        token_to_inject = word_to_toks['cat'][rand_index]
                        # Now we want to change the sentence a bit
                        #    iterate through the tokens and if any of them are nouns, then replace
                        idx_to_replace = None
                        for tok_idx in range(tokens.shape[1]):
                            token_in_question = tokens.squeeze().tolist()[tok_idx]
                            #word_in_question = tokenizer.decode([token_in_question]).strip()
                            try:
                                word_in_question = tokenizer.decode([token_in_question]).strip()
                            except:
                                print("****",token_in_question,"***",tokens,"****",flush=True)
                                raise
                            assert type(word_in_question) is str
                            # Now we want to see if it's a noun
                            doc = nlp(word_in_question)
                            pos = [token.pos_ for token in doc]
                            if pos == ['NOUN'] and len(word_in_question)>2 and word != word_in_question and word+'s' != word_in_question: # we have a viable noun
                                # we need all these conditions because spacy thinks single letters are nouns
                                idx_to_replace = tok_idx
                            #pdb.set_trace()

                        if idx_to_replace is not None: # we actually found a NOUN to replace
                            tokens[0,idx_to_replace] = token_to_inject
                            idx_to_replace_in_all_toks = idx_to_replace - tokens.shape[1] + all_text_tokens.shape[1]
                            all_text_tokens[-1,idx_to_replace_in_all_toks] = token_to_inject
                            # keep track of when this happened
                            index_of_last_injection = i
                        #pdb.set_trace()"""


                num_gpt2_iters_run = i+1
                big_array = big_array[-len_for_big_array:]
				
                # figure out true classification
                orig_classification = np.zeros(len(WORDS)+3) # FAKE_DATA_INDEX = 0, UNK_LABEL_INDEX = -2, GPT2_MODEL_INDEX = -1
                orig_classification[GPT2_MODEL_INDEX] = 1.
				
                # count words and see if we should append to dataset
                for i_word, word in enumerate(WORDS):
                    if found_words_dict[word]: # means we found this word
                        orig_classification[i_word+1] = 1.
                        word_counts[word] = word_counts[word] + 1
                        # then check if we should append or not
                        if word_counts[word] > num_sentences:
                            append_to_dataset = False
                if True not in list(found_words_dict.values()):
                    orig_classification[UNK_LABEL_INDEX] = 1.
                    word_counts['UNK'] = word_counts['UNK'] + 1
                    if word_counts['UNK'] > 1.2*max([word_counts[word] for word in WORDS])+1: # keep the UNKs down!!!!!!
                        append_to_dataset = False
                        word_counts['UNK'] = word_counts['UNK'] - 1 # so we actually won't count this one since we're not appending

                if not append_to_dataset: # whats the point of even being here
                    #continue
                    pass

                # What will we call "original text" and "generated text"
                # original text from before loop, actually
                assert all_text_tokens.squeeze().tolist()[-sent_len:] == tokens.squeeze().tolist()
                orig_text_tokens = all_text_tokens[:,-sent_len-num_iters:-num_iters]
                generated_text_tokens = tokens # komya

                orig_tokens = orig_text_tokens.squeeze().tolist()
                gpt2_generated_tokens = generated_text_tokens.squeeze().tolist()
                #try:
                #    assert len(orig_tokens) == sent_len and len(gpt2_generated_tokens) == num_iters
                #except:
                #    pdb.set_trace()
                #    raise
                #    append_to_dataset = False
                #    pass
                orig_text = gpt2_tokenizer.decode(orig_tokens)
                gpt2_generated_text = gpt2_tokenizer.decode(gpt2_generated_tokens)

                # Now the big_array is a list of length 40 (num_iters*2) of tensors with shape (1,sent_len,1024)
                big_array = torch.cat(big_array, dim=1)
                big_array = big_array.permute(1,2,0) # shape is (2*sent_len*num_iters, 1024, 1) now
                big_array = big_array.data.cpu().numpy()

                # We want to save this big_array in the data
                #ORIG_ACTIV_INDEX = 0
                #ORIG_LABEL_INDEX = 1
                #TARG_LABEL_INDEX = 2
                #LANG_MODEL_INDEX = 3
                #META_DATA_INDEX = 4
                #ORIG_TEXT_INDEX = 5
                #PRED_TEXT_INDEX = 6
                #TARG_TEXT_INDEX = 7
                #GPT2_TEXT_INDEX = 8
                if append_to_dataset:
                    datum = [
			        big_array, # ORIG ACTIV
			        orig_classification, # ORIG LABEL
		                TARG_CLASSIFICATION, # None, # TARG LABEL Ugh I could do this but I just don't use it in the single-control version
			        model_name, # LANG MODEL: model_name an abstraction for 'gpt2'
                                {'num_gpt2_iters':num_gpt2_iters_run,\
                                        'orig_tokens':orig_tokens,\
                                        'gpt2_generated_tokens':gpt2_generated_tokens}, # META DATA
			        orig_text, # ORIG TEXT (or what we're deeming 'originial text')
			        None, # PRED TEXT this literally won't exist until we have an NPI
			        WORDS[0], # TARG TEXT so this would need to be TWEAKED for multi-word applications. For now it's 'cat'
			        gpt2_generated_text # GPT2 TEXT: just generated right here by the GPT2
				    ]
                    #pdb.set_trace()
                    dataset.append(datum)

                # check for debugging
                #if len(dataset) == 50:
                #    pdb.set_trace()

                # Check data len
                if len(dataset) >= num_sentences_per_chunk:
                    # Then we want to save it
                    # pkl stuf
                    pkl_name = pkl_name_base + "_" + str(pkl_counter)
                    with open(pkl_name,'wb') as f:
                        pkl.dump(dataset,f)

                    # More business to conduct every num_sentences_per_chunk data
                    del dataset
                    dataset = []
                    pkl_counter += 1

                # Now if we have all the pickles we need... we break! (bc we done yay)
                if pkl_counter == num_chunks:
                    break # :)

                # Now for helpful print statements
                if iterator % (num_sentences_per_chunk//(num_sentences_per_chunk//100)) == 0:
                    print("iterations: {}; cat word data: {}/{}, generic data: {}/{}, pkls written: {}".format(iterator+1, word_counts['cat'], num_sentences//2, word_counts['UNK'], num_sentences//2, pkl_counter), \
                                flush=True)

    except:
        raise

    finally:
        torch.cuda.empty_cache()

    
    # shuffle now
    # random.shuffle(word_sent_list)
    torch.cuda.empty_cache()
    print(" ",flush=True)
    print("done")

"""
"""
