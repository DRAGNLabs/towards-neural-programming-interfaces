import pickle as pkl
import torch
from train_classifier import Classifier, extract_needed_layers
#import train_class
#from traindeepskip_class import Classifier
import numpy as np
from matplotlib import pyplot as plt
import pdb
import argparse

if __name__ == "__main__":

    """
    command line:
    nvidia-docker run -d --name test_chase -v /raid/remote/name:/raid/remote/name -v /mnt/server -e NVIDIA_VISIBLE_DEVICES=10 nvcr.io/nvidia/pytorch:19.10-py3 python3 /raid/remote/name.py &> foo.log &
        """

    """Political figure
    --file_path_base /raid/remote/name/ --data_path_base /raid/remote/data.pkl_ --test_start_num 34 --test_end_num 39 --epoch_num 20
    """
    """RACIST
    --file_path_base  /raid/remote/classifier/ --data_path_base /raid/remote/data/ --test_start_num 25 --test_end_num 29 --epoch_num 90
    """
    """OFFENSE
    --num_pkls 11 --first_perturbation_index 5 --second_perturbation_index 11 --save_file_path /raid/remote/name/ --train_file_path_base /raid/remote/data/
    """

    EPOCH_NUM_LIST = [4,6,8]#, 20, 30, 40] #, 40]
    FILE_PATH_LIST = ["./classifiers/layers_5_11/"] * len(EPOCH_NUM_LIST)

    for classifier_num in range(len(EPOCH_NUM_LIST)):
        EPOCH_NUM = EPOCH_NUM_LIST[classifier_num]
        TEST_NUMS = [0]#list(range(31,34))
        FILE_PATH = FILE_PATH_LIST[classifier_num]#"/raid/remote/n8rob/slurs/trump_classifiers3/layers_5_11/"
        DATA_PATH = "./data/B_smGPT2_arrays"
        
        PRED_INDS = [5,11]
        print("NEW FILE",FILE_PATH,"epoch num",EPOCH_NUM,flush=True)

        #print("Testing classifier from epoch {} in folder class_out_cat{} with prediction indices {}".format(EPOCH_NUM, ATTEMPT_NUM, PRED_INDS),flush=True)
    
        # get images
        #with open(FILE_PATH+"N8_classification_loss_summaries.pkl",'rb') as f:
        #    losses = pkl.load(f)
        #els = losses["epoch_losses"]
        #bls = losses["batch_losses"]
        #tls = [tup[1] for tup in losses["tests"]]

        #plt.plot(els,'ro')
        #plt.savefig(FILE_PATH+"epoch_loss_plot.png")#.format(ATTEMPT_NUM))
        #plt.plot(bls,'ro')
        #plt.savefig(FILE_PATH+"batch_loss_plot.png")#.format(ATTEMPT_NUM))
        #plt.plot(tls,'ro')
        #plt.savefig(FILE_PATH+"test_loss_plot.png")#.format(ATTEMPT_NUM))
        #plt.plot(tls[500:],'ro')
        #plt.savefig(FILE_PATH+"test_loss_partial.png")#.format(ATTEMPT_NUM))

        # Load classifier
        classifier = torch.load(FILE_PATH+"Classifier_classification_network_epoch{}.bin".format(EPOCH_NUM),map_location=torch.device('cpu')).cuda() # KOMYA

        collected_accs = []
        collected_alt_accs = []

        for test_num in TEST_NUMS:
        #for pklnum in range(1,71):
            # Load arrays
            with open(DATA_PATH+".pkl_{}".format(test_num),'rb') as f:
            #with open("../../DATA/npi_proj/CAT_IN_sentence_arrays_{}.pkl".format(pklnum),'rb') as f:
                money = pkl.load(f)

            score = 0
            alt_score = 0
            for i in range(len(money)):
                arr = extract_needed_layers(money[i][0],PRED_INDS) # HACK NOTE
                #arr = money[i][0] 
                arr = torch.Tensor(arr).cuda()#.to(torch.device('cuda:1'))
                #if CONV:
                #    arr = arr.unsqueeze(0)#.permute(0,3,1,2)
                sent = money[i][-1]
                truth = money[i][1][1]
                #truth = list(money[i][1][0,:,0])[0]
                #yhat = list(classifier(arr).cpu().data.numpy())[0]
                yhat = classifier(arr).squeeze().cpu().item()
                if truth == 1 and yhat >= .5:#.7:
                    score += 1
                elif truth == 0 and yhat < .5:#.7:
                    score += 1
                if truth == 1 and yhat >= .7:#.7:
                    alt_score += 1
                elif truth == 0 and yhat < .7:#.7:
                    alt_score += 1
                #else:
                #    print(sent.replace('\n','\\n'), "truth",truth,"yhat",yhat)
                if i % 100 == 0:
                    print(sent.replace('\n','\\n'))
                    print("truth",truth)
                    print("yhat",yhat)

            score = score/len(money)
            alt_score = alt_score/len(money)
            print("ACCURACY FOR TEST {}: {}".format(test_num,score)) #HACK
            print("ALT-ACCURACY FOR TEST {}: {}".format(test_num,alt_score))

            collected_accs.append(score)
            collected_alt_accs.append(alt_score)

        avg_acc = np.mean(collected_accs)
        avg_alt_acc = np.mean(collected_alt_accs)
        print('done')
        print("TOTAL ACCURACY OVERALL:",avg_acc)
        print("TOTAL ALT-ACCURACY OVERALL:",avg_alt_acc)

