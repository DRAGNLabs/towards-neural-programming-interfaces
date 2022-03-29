from argparse import ArgumentParser
import torch

from npi.config import NPIConfig
from npi.dataset.npi_dataset import NPIDatasetLoader
from npi.models import NPITrainingModels
from npi.training.test_classifier import test_classifier

from npi.training.test_npi import test_npi
from npi.training.train_classifier import train_classifier

from npi.training import NPITrainer

def train(args):
    model = args.model
    device = torch.device("cuda:0")  # TODO: Hyperparam
    num_epochs = 6  # TODO: Hyperparam
    split_ratio = 0.25  # TODO: Hyperparam
    batch_size = 5 # TODO: Hyperparam
    headstart = 0 # TODO: Hyperparam, set at 0 for debugging
    config = NPIConfig(device, model_save_folder="../models/npi_models/", dataset_folder="../data/processed", npi_name="politics")
    models = NPITrainingModels(
        config,
        style_classifier_path="../notebooks/politics/classifiers/layers_5_11/Classifier_classification_network_epoch0.bin",
    )
    if model == "npi":
        trainer = NPITrainer(config, batch_size=batch_size, headstart=headstart)
        dataset_loader = NPIDatasetLoader(config, split_ratio=split_ratio)
        trainer.train_adversarial_npi(models, num_epochs, dataset_loader)
    elif model == "classifier":
        train_classifier(args)
    else:
        print("Only can train classifier or npi.")


def test(args):
    model = args.model
    if model == "npi":
        test_npi(args)
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
