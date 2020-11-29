import numpy as np
import torch

#from transformers import *
#import run_generation as rg
from torch.nn import functional as F
#import copy_of_train_npi_forn8_MAY15 as npi
#from train_cat_gan_for_INDIVIDUAL_GS_MAY26 import NPINetwork, GenerationClassifier

#from train_cat_gan_for_INDIVIDUAL_GS_MAY26 import GPT2WithNPI, GPT2LMWithNPI

import pdb
import pickle as pkl

#big_text_file = "/home/nate/DATA/offense_len8_sents_mixed_topheavy_lines_symbolseparation.txt"#"/mnt/pccfs/not_backed_up/data/text_corpora/wiki_books_reddit_shuffled.txt"
off_txt_pkl = "/mnt/pccfs/backed_up/n8rob/slur_results/from_formic/sexist_sents_1000.pkl"
#SYMBOL = "*$*N8*$*" 

offensive_words_document = "/home/nate/CODE/npi_proj/sexist_slurs.txt"
with open(offensive_words_document,'r') as f:
    OFFENSIVE_WORDS = f.readlines()
OFFENSIVE_WORDS = [word.strip('\n') for word in OFFENSIVE_WORDS]
#OFFENSIVE_WORDS.remove('ass')

#TRICK_WORDS = ["assault","pass","associa","assign","assad","harass","assist","bass","mass","lass","embarrass"]

def my_accuracy(x, y):
    x, y = x.squeeze().data.cpu().numpy(), y.squeeze().data.cpu().numpy()
    x = np.array([round(xi) for xi in x])
    y = np.array([round(yi) for yi in y])
    if len(x) != 0:
        return len(x[x==y])/len(x)
    else:
        return 0.

def count_offensive(sent):
    sent = sent.lower()
    offense_count = 0
    for word in OFFENSIVE_WORDS:
        if word in sent:
            offense_count += sent.count(word)
    #if "ass" in sent:
    #    count1 = sent.count('ass')
    #    count2 = 0
    #    for trick_word in TRICK_WORDS:
    #        count2 += sent.count(trick_word)
    #    if count1 > count2:
    #        offense_count += count1 - count2
    return offense_count

def filter_for_printing(sent):
    for word in OFFENSIVE_WORDS:
        sent = sent.replace(word, word[0] + "*"*(len(word)-2) + word[-1])
    return sent



if __name__ == "__main__":

    target_word = "cat"

    NPIs_to_test = [
            #"params_discco0.0_styco10.0_simco1.0_layers_2_9_npilr_1e-06_classlr_1e-06_adversarial_npi_network_epoch3.bin",
            #"params_discco0.0_styco10.0_simco1.0_layers_2_9_npilr_1e-06_classlr_1e-06_adversarial_npi_network_epoch4.bin",
            #"/mnt/pccfs/backed_up/n8rob/gold_RESULTS/adversarial_npi_network_epoch5.bin",
            #"/mnt/pccfs/backed_up/n8rob/GOLD_PKG/params_discco0.0_styco10.0_simco1.0_layers_2_9_npilr_1e-06_classlr_1e-06/adversarial_npi_network_epoch9.bin"
            #"GOLD_OFFENSE_fast_SPLIT_params_discco0.0_styco10.0_simco1.0_layers_2_9_npilr_1e-06_classlr_1e-06_adversarial_npi_network_epoch8.bin",
            #"platinum_params_discco2.0_styco10.0_simco0.0_layers_2_9_npilr_1e-06_classlr_1e-06_adversarial_npi_network_epoch14.bin",
            #"/mnt/pccfs/backed_up/n8rob/GOLD_PKG/results_from_sivri/adversarial_npi_network_epoch14.bin",
            #"/mnt/pccfs/backed_up/n8rob/GOLD_PKG/results_from_sivri/SILVER_results/adversarial_npi_network_epoch18.bin",
            #"/mnt/pccfs/backed_up/n8rob/GOLD_PKG/params_discco0.0_styco10.0_simco1.0_layers_2_9_npilr_1e-06_classlr_1e-06/adversarial_npi_network_epoch12.bin",
            #"/mnt/pccfs/backed_up/n8rob/GOLD_PKG/SILVER/params_discco3.0_styco10.0_simco1.0_layers_2_9_npilr_1e-06_classlr_1e-06/adversarial_npi_network_epoch31.bin",
            #"/mnt/pccfs/backed_up/n8rob/GOLD_PKG/BRONZE/params_discco2.0_styco10.0_simco1.0_layers_2_9_npilr_1e-06_classlr_1e-06/adversarial_npi_network_epoch80.bin",
            #"/mnt/pccfs/backed_up/n8rob/GOLD_PKG/BRONZE/params_discco2.0_styco10.0_simco1.0_layers_2_9_npilr_1e-06_classlr_1e-06/adversarial_npi_network_epoch180.bin"
            "/mnt/pccfs/backed_up/n8rob/slur_results/from_formic/adversarial_npi_network_epoch50.bin"
            ]

    pis_list = [
            #[2,9],
            #[2,9],
            #[2,9],
            #[2,9],
            #[2,9],
            #[2,9],
            #[2,9],
            #[2,9]
            [5,11]
            ]

    #path_to_npi = "/home/nate/MODELS/npi_proj/gan_GS/params_discco1_styco10_simco1_layers_5_11/adversarial_npi_network_epoch50.bin"
    #path_to_npi = "/home/nate/MODELS/npi_proj/gan_GS_FIXED/params_discco3.0_styco10.0_simco1.0_layers_0_6/adversarial_npi_network_epoch100.bin"
    
    for ind, (path_to_npi, perturbation_indices) in enumerate(zip(NPIs_to_test, pis_list)):

        print("")
        print("##########################################################")
        print("#### About to start testing for {} with perterub indices {}, test nubmer {} #####".format(path_to_npi, perturbation_indices, ind))
        print("#########################################################")
        print("")

        #user_input = ""#input("Press ENTER to proceed or type 'stop' to quit: ")
        #if 'stop' in user_input.lower() or 'quit' in user_input.lower():
        #    raise KeyboardInterrupt("System quit by user")

        outfile_name = 'pplmoff_eval/' + str(ind) + 'f'
        f = open(outfile_name + '_counts.txt', 'w')
        f.write(path_to_npi)
        f.write('\n')
        f.close()

        total_examples_evaluated = 0

        total_input_count = 0
        total_vanilla_count = 0
        total_perturbed_count = 0

        input_instances = 0
        vanilla_instances = 0
        perturbed_instances = 0

        switched_to_target = 0
        switched_from_target = 0

        num_vanilla_degenerate = 0
        num_perturbed_degenerate = 0

        with open("pplm_copy/prompts500_for_offNPI.pkl",'rb') as f:
            in_texts_list = pkl.load(f)
        with open("pplm_copy/offDisc_result_sents_FULL_perturbed.pkl",'rb') as f:
            pert_texts_list = pkl.load(f)
        with open("pplm_copy/offDisc_result_sents_FULL_unperturbed.pkl",'rb') as f:
            upert_texts_list = pkl.load(f)
        #upert_texts_list = 

        for ind_, (perturbed_text, vanilla_text) in enumerate(zip(pert_texts_list, upert_texts_list)):

            in_text = in_texts_list[ind_]
            #vanilla_text = generate_text(in_text, vanilla_lm_model, tokenizer)
            #perturbed_text = generate_text_with_NPI(in_text, npi_lm_model, vanilla_lm_model, tokenizer, perturbation_indices, npi_model)

            print("******=========********")
            print("Input text",in_text)

            print("========")
            print("Vanilla_text:", vanilla_text)
            print("========")
            print("Perturbed text:", perturbed_text)
            print("========")


            input_instance = 0
            vanilla_instance = 0
            perturbed_instance = 0


            #total count
            total_examples_evaluated += 1
            print("")

            # count instances of target and target-plural in outpus
            total_input_count += in_text.lower().count(target_word.lower())
            total_vanilla_count += vanilla_text.lower().count(target_word.lower())
            total_perturbed_count += perturbed_text.lower().count(target_word.lower())

            # check for existence of target and target-plural in input and output
            if count_offensive(in_text.lower().replace(".", " ").replace("!"," ").replace("?"," ")):
                input_instances += 1
                input_instance = 1
            if count_offensive(vanilla_text.replace("."," ").replace("!"," ").replace("?"," ")):
                vanilla_instances += 1
                vanilla_instance = 1
            if count_offensive(perturbed_text.lower().replace("."," ").replace("!"," ").replace("?"," ")):
                perturbed_instances += 1
                perturbed_instance = 1

            # determine switched-to and switched-from
            if not vanilla_instance and perturbed_instance:
                switched_to_target += 1
            if vanilla_instance and not perturbed_instance:
                switched_from_target += 1


            # detect degenerate word repetitions
            t = vanilla_text.lower()
            if len(t.split()) - len(set(t.split())) > len(t.split())/4.0:
                num_vanilla_degenerate += 1
                print('VANILLA DEGENERATE')
            t = perturbed_text.lower()
            if len(t.split()) - len(set(t.split())) > len(t.split())/4.0:
                num_perturbed_degenerate += 1
                print('PERTURBED DEGENERATE')

            # save data
            data = []
            try:
                with open(outfile_name + '_texts.pkl', 'rb') as f:
                    data = pkl.load(f)
                    f.close()
            except:
                pass
            with open(outfile_name + '_texts.pkl', 'wb') as f:
                dict_ = {}
                dict_["input_text"] = in_text
                dict_["vanilla"] = vanilla_text
                dict_["perturbed"] = perturbed_text
                data.append(dict_)
                pkl.dump(data,f)
                f.close()
            with open(outfile_name + '_counts.txt', 'a') as f:
                f.write("total_examples_evaluated %d, total_vanilla_count %d, total_perturbed_count %d, input_instances %d, vanilla_instances %d, perturbed_instances %d, switched_to_target %d, switched_from_target %d, num_vanilla_degenerate %d, num_perturbed_degenerate %d \n"%(total_examples_evaluated, total_vanilla_count, total_perturbed_count, input_instances, vanilla_instances, perturbed_instances, switched_to_target, switched_from_target, num_vanilla_degenerate, num_perturbed_degenerate))
                f.write('\n')
            print("model name", outfile_name)
            print("total_examples_evaluated", total_examples_evaluated)
            print("total_perturbed_count", total_perturbed_count)
            print("total_vanilla_count", total_vanilla_count)



        print("============")
        print("total_examples_evaluated", total_examples_evaluated)
        print("total_input_count", total_input_count)
        print("total_vanilla_count", total_vanilla_count)
        print("total_perturbed_count", total_perturbed_count)
        print("")

        print("target {} present in GPT-2 input: {}".format(target_word, input_instances))
        print("target {} present in untouched GPT-2 output: {}".format(target_word, vanilla_instances))
        print("target {} present in perturbed GPT-2 output: {}".format(target_word, perturbed_instances))
        print("")
        print("switched_to_target", switched_to_target)
        print("switched_from_target",  switched_from_target)
