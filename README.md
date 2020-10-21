>📋  A template README.md for code accompanying a Machine Learning paper

# Towards Neural Program Interfaces

This repository is the official implementation of "Towards Neural Program Interfaces" (https://arxiv.org/abs/2030.12345). 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  Describe how to set up the environment, e.g. pip/conda/docker commands, download datasets, etc...

## Dataset generation

To generate a dataset, run this command:

```data
python construct_data.py --word <word>
```

## Training classifier

To train a classifier model on the generated dataset, run this command:

```classifier
python train_classifier.py
```

## Evaluating classifier

To evaluate a classifier model, run this command:

```evaluate
python test_classifier.py
```

## Training NPI

To train an NPI model, run this command:

```train
python train_npi.py
```

## Evaluating NPI

To evaluate an NPI model, run this command:

```classifier
python evaluate_npi_fast.py
```

>📋  Describe how to train the models, with example commands on how to train the models in your paper, including the full training procedure and appropriate hyperparameters.

>📋  Describe how to evaluate the trained models on benchmarks reported in the paper, give commands that produce the results (section below).

## Pre-trained Models

You can download pretrained models here:

- [My awesome model](https://drive.google.com/mymodel.pth) trained on ImageNet using parameters x,y,z. 

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

>📋  Pick a licence and describe how to contribute to your code repository. 
