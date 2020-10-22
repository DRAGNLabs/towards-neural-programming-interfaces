# Towards Neural Program Interfaces

This repository is the official implementation of "Towards Neural Programming Interfaces" (to be published and presented in NeurIPS 2020 proceedings). 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Dependencies

To install dependencies and get ready to run scripts, simply run:

```setup
bash install_dependencies.sh
```
This bash script uses pip to install needed packages.

## Dataset generation

To generate a dataset, run this command:

```data
python construct_data.py --target-words <target words>
```
You may choose not to specify the target word option, in which case the default will be a set of sexist terms

## Training classifier

To train a classifier model on the generated dataset, run this command:

```classifier
python train_classifier.py
```
For this and other scripts you may specify keyword arguments as you see fit.

## Evaluating classifier

To evaluate a classifier model, run this command:

```evaluate
python test_classifier.py
```
And observe printed output. If classifier's performance is low, consider training again with a different class-learning-rate

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
