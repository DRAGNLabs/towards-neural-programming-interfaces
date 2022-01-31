from argparse import ArgumentParser
from src.training.test_classifier import test_classifier
from src.training.test_npi import test_npi
from src.training.train_classifier import train_classifier

from src.training.train_npi import train_adversarial_NPI

def train(args):
    model = args["model"]
    if(model == "npi"):
        train_adversarial_NPI(args)
    elif(model == "classifier"):
        train_classifier(args)
    else:
        print("Only can train classifier or npi.")

def test(args):
    model = args["model"]
    if(model == "npi"):
        test_npi(args)
    elif(model == "classifier"):
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
