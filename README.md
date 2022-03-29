# Towards Neural Programming Interfaces

This repository is the official implementation of "Towards Neural Programming Interfaces", published and presented in NeurIPS 2020 proceedings. See https://arxiv.org/abs/2012.05983 for preprint.

## TODO:

* Generate data correctly for sentiment (get the tokens from the sentences and loop through them?)
* Figure out using N-hot encoding for training the NPI
* Using different layers for pertubations for multi-classification
* For multiclassification, we need to update the style classifier to support that.

## Dependencies

### Pipenv

A PipEnv is set up in this repo with a Pipfile. You can initialize the virtual environment easily with the `pipenv` package.
You can install that package with `pip3 install pipenv`.

After you have `pipenv` installed, you can initialize the virtual environment by running `pipenv install` in the root of this repository. That is all you need to do to initialize and install dependencies for this project!  
(Create an issue if you see issues).

To use the virtual environment in the shell, you can use `pipenv run <command>` to run a single command in the environment.
You can also create a shell with the environment using `pipenv shell`. This should be convienient for those `vim` users.

If you are using VS Code, VS Code should be able to detect the `pipenv` environment created. Upon starting VS Code, in the repository forlder, VS Code will ask you to choose a Python environment. Select the `pipenv` environment to allow VS Code to lint and autocomplete using the environment.

PyCharm should also be able to detect the virtual environment as well.

### Use Bash script
Whether or not you choose to use Pipenv, for now, there are dependencies that would need to be installed manually. Simply run the following:

 * If using pipenv:
    ```sh
    pipenv shell
    ./install_dependencies.sh
    ```
 * If installing all dependencies manually, just run `./install_dependencies.sh` as a bash command.

This bash script uses `pip3` to install needed packages using `requirements.txt`. It may or may not work properly depending on your environment. Therefore, it is recommended to use pipenv to create the environment.

## How to work with this project

### Jupyter Notebooks

As you are experimenting with this project, it is recommended to use Jupyter notebooks so you can quickly try out the repository and make modifications on the fly.

A good example of usage can be seen in `notebooks/politics/politics.ipynb`

### CLI

**This is a WIP**

After installing the environment, you will also be able to run scripts. You should be able to run scripts (if using `pipenv`) with these two options:

 * `pipenv run python -m npi <command>`

 * or:
    ```sh
    pipenv shell
    python -m npi <command>
    ```

## Pre-trained Models

Pre-trained models for "cat"-induction, "cat"-avoidance, racial-slur-avoidance, and sexist-slur-avoidance are in the `model/pretrained` folder

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
