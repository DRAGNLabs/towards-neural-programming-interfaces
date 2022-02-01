#           Data that we can use to train           #
#               our classifeir and NPI              #
#                                                   #
#          Fulda, Brown, Wingate, Robinson          #
#                       DRAGN                       #
#                    NPI Project                    #
#                       2020                        #

import argparse
import torch
from .construct_sentiment_data import construct_sentiment_data
import pandas as pd

"""
Calls construct data on construct_sentiment_data for the dataset here:
https://github.com/JerryWei03/NewB
Note that pandas expect that the header of the file is appended:
view: number
text: str
"""

def construct_politics_data(model_layers, data, gpu_device_num, pkl_name):
    data_iter = zip(data['text'], data['view'])
    device = torch.device(F"cuda:{gpu_device_num}")
    construct_sentiment_data(model_layers=model_layers, data_iter=data_iter, data_len=len(data), gpu_device=device,
                                 pkl_name=pkl_name, model_name="gpt2", window_size=10, num_iters=10, num_chunks=40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--view",
                        default="both",
                        help="liberal conservative, or both. You can run this script simutaneously")
    args = parser.parse_args()

    model_layers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] 
    if args.view == "both":
        with open("NewB/train_orig.txt", 'r', newline='') as f:
            data = pd.read_csv(f, delimiter="\t")
            data["view"] = data["view"].replace(
                to_replace={0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1})
            data = data[data.view != 5]
            data = data.sample(frac=1)
            construct_politics_data(model_layers, data, 1, "data/politics/sentence_arrays.pkl")
    elif (args.view == "liberal"):
        with open("NewB/liberal.txt", 'r', newline='') as f:
            data = pd.read_csv(f, delimiter="\t")
            construct_politics_data(model_layers, data, 0, "data/politics/liberal/sentence_arrays.pkl")
    else:    
        with open("NewB/conservative.txt", 'r', newline='') as f:
            data = pd.read_csv(f, delimiter="\t")
            data["view"] = data["view"].replace(
                to_replace={0: 1})
            construct_politics_data(model_layers, data, 1, "data/politics/conservative/sentence_arrays.pkl")
