#             Test content classifiers to             #
#            make sure they are working well          #
#           enough to be used in NPI training         #
#           (Accuracy should be very high:            #
#                 ideally 99% or more)                #
#                                                     #
#           Fulda, Brown, Wingate, Robinson           #
#                        DRAGN                        #
#                     NPI Project                     #
#                        2020                         #

import argparse
import pickle as pkl

import numpy as np
import torch

from npi.dataset.npi_dataset import extract_needed_layers
from npi.models.classifiers import StyleClassifier

def test_classifier(args):
    EPOCH_NUM_LIST = [int(pi) for pi in args.test_epochs.split(',')]
    FILE_PATH_LIST = [args.model_dir_path] * len(EPOCH_NUM_LIST)

    for classifier_num in range(len(EPOCH_NUM_LIST)):
        epoch_num = EPOCH_NUM_LIST[classifier_num]
        # these pickles are designated for testing!!
        test_nums = [int(pi) for pi in args.test_pkls.split(',')]
        file_path = FILE_PATH_LIST[classifier_num]
        data_path = args.data_path

        PRED_INDS = [int(pi) for pi in args.perturbation_indices.split(',')]
        print("NEW FILE", file_path, "epoch num", epoch_num, flush=True)

        # Load classifier
        print("Creating Classifier Model", flush=True)
        # TODO when modifiying PRED_IND, this may need to change
        classifier_model = StyleClassifier().float()

        #   We load the model from the CPU just in case it was trained on a different GPU than the one we are using
        classifier_model.load_state_dict(torch.load(F"{file_path}Classifier_classification_network_epoch{epoch_num}.bin",
                                                    map_location=torch.device('cpu')))
        classifier_model.cuda()
        classifier_model.eval()

        collected_accs = []
        # collected_alt_accs = []

        for test_num in test_nums:
            with open(data_path + ".pkl_{}".format(test_num), 'rb') as f:
                money = pkl.load(f)

            score = 0
            # alt_score = 0
            for i in range(len(money)):
                # TODO Figure out how to generalize and fix extract_needed_layers
                arr = extract_needed_layers(money[i][0], PRED_INDS)
                arr = torch.Tensor(arr).cuda()
                sent = money[i][-1]
                truth = money[i][1][1]
                yhat = classifier_model(arr).squeeze().cpu().item()
                if truth == 1 and yhat >= .5:
                    score += 1
                elif truth == 0 and yhat < .5:
                    score += 1
                # if truth == 1 and yhat >= .7:
                #     alt_score += 1
                # elif truth == 0 and yhat < .7:
                #     alt_score += 1

                if i % 100 == 99:
                    print(sent.replace('\n', '\\n'))
                    print("truth", truth)
                    print("yhat", yhat)

            score = score / len(money)
            # alt_score = alt_score/len(money)
            print("ACCURACY FOR TEST {}: {}".format(test_num, score))  # HACK
            # print("ALT-ACCURACY FOR TEST {}: {}".format(test_num,alt_score))

            collected_accs.append(score)
            # collected_alt_accs.append(alt_score)

        avg_acc = np.mean(collected_accs)
        # avg_alt_acc = np.mean(collected_alt_accs)
        print('done')
        print("TOTAL ACCURACY OVERALL:", avg_acc)
        # print("TOTAL ALT-ACCURACY OVERALL:",avg_alt_acc)
        print("\n================================================\n", flush=True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir-path",
                        default="classifiers/layers_5_11/",
                        help="path to directory containing classifiers")
    parser.add_argument("--data-path",
                        default="data/sentence_arrays",
                        help="path to data (standard file name witout pkl suffix, full or relative file path)")
    parser.add_argument("--test-pkls",
                        type=str,
                        # See NOTE in arg-parsing section of train_classifier.py (line 378)
                        default="53,54,55,56",
                        help="pkl numbers for data designated for testing: string of numbers separated by commas")
    parser.add_argument("--test-epochs",
                        type=str,
                        default="20,30,40,50,55,60,65,70",
                        help="epoch nums for class'n models we want to test: string of numbers separated by commas")
    parser.add_argument("--perturbation-indices",
                        type=str,
                        default="5,11",
                        help="indices for layers to extract from language model activations: string of numbers separated by commas")

    args = parser.parse_args()

    test_classifier(args)
