from argparse import ArgumentParser
import pickle as pkl
import torch
from npi.config import NPIConfig
from npi.dataset.npi_dataset import NPIDataSet
from npi.models import NPITrainingModels, Classifier
from npi.training.test_classifier import test_classifier

# from npi.training.test_npi import test_npi
from npi.training.train_classifier import train_classifier

from npi.training import NPITrainer


def train(args):
    model = args.model
    device = torch.device("cuda:0")  # TODO: Hyperparam
    num_epochs = 6  # TODO: Hyperparam
    split_ratio = 0.25  # TODO: Hyperparam
    config = NPIConfig(device, save_folder="../models/npi_models/")
    models = NPITrainingModels(
        config,
        content_classifier_path="../notebooks/politics/classifiers/layers_5_11/Classifier_classification_network_epoch0.bin",
    )
    if model == "npi":
        trainer = NPITrainer(config, batch_size=4, headstart=0)
        with open("../data/processed/politics/sentence_arrays.pkl_0", "rb") as datafile:
            dataset = pkl.load(datafile)
        max_train = len(dataset) - int(split_ratio * len(dataset))
        train_data = NPIDataSet(dataset[:max_train], config, return_row=True)
        test_data = NPIDataSet(dataset[max_train:], config, return_row=True)
        trainer.train_adversarial_npi(
            models, num_epochs, train_data, test_data
        )
    elif model == "classifier":
        train_classifier(args)
    else:
        print("Only can train classifier or npi.")


def test(args):
    model = args["model"]
    if model == "npi":
        # test_npi(args)
        pass
    elif model == "classifier":
        test_classifier(args)
    else:
        print("Only can train classifier or npi.")


def construct(args):
    print("Construct", args)


parser = ArgumentParser(prog="src")

parser.add_argument("--dataset", required=False)

action_parser = parser.add_subparsers(title="actions", required=True)
# parser.add_argument("action", default="test", choices=["train", "test", "construct"])
train_parser = action_parser.add_parser("train")
train_parser.add_argument("model", choices=["npi", "classifier"])
train_parser.set_defaults(run=train)

test_parser = action_parser.add_parser("test")
test_parser.add_argument("model", choices=["npi", "classifier"])
test_parser.set_defaults(run=test)

construct_parser = action_parser.add_parser("construct")
construct_parser.set_defaults(run=construct)

args = parser.parse_args()
args.run(args)
