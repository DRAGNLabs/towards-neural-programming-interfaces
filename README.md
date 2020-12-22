# Towards Neural Programming Interfaces

This repository is the official implementation of "Towards Neural Programming Interfaces" (to be published and presented in NeurIPS 2020 proceedings). See https://arxiv.org/abs/2012.05983 for preprint.

## Dependencies

To install dependencies and get ready to run scripts, simply run:

```setup
bash install_dependencies.sh
```
This bash script uses pip to install needed packages.

## Dataset generation

To generate a dataset, run this command:

```data
python construct_data.py
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
python test_npi.py
```

## Pre-trained Models

Pre-trained models for "cat"-induction, "cat"-avoidance, racial-slur-avoidance, and sexist-slur-avoidance in folder

```pretrained
pretrained_models
```

## Results

Our model achieves the following performance on :


| Model name         | Target in output with NPI  | Target in output without |
| ------------------ |--------------------------- | ------------------------ |
| Sexist slur avoid. |          10.3%             |          90.2%           |
| Racist slur avoid. |           0.5%             |          52.1%           |
| Cat induction      |          48.8%             |           0.0%           |
| Cat avoid.         |          11.2%             |          38.8%           |

Running the scripts with default parameters as described here should reproduce the sexist slur results.
See our full paper for further details about these results and our methods.


Brigham Young University DRAGN Labs
Brigham Young University PCC Lab
