#           Pre-train content classifer to          #
#                 use in NPI training               #
#                                                   #
#          Fulda, Brown, Wingate, Robinson          #
#                       DRAGN                       #
#                    NPI Project                    #
#                       2020                        #

"""
Neural Programming Interfaces (NPI)

Overview:
    Classifier Code:
        - Includes functionality for loading pretrained classifiers
        - Supported neural models:
            - GPT2
"""

import gc

import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from npi.config import NPIConfig
from npi.dataset import NPIDatasetLoader

from npi.modeling_neural_program_interfaces import *
from npi.models import NPITrainingModels

class NPIStyleTrainer:
    def __init__(
        self,
        config: NPIConfig,
        save_freq=5,
        test_freq=5,
        batch_size=5,
        class_lr=1e-3,
    ) -> None:
        self.config = config
        self.save_file_path = (
            config.model_save_folder + config.npi_name + "_style_model"
        )
        self.train_file_path = config.dataset_file

        self.device = config.device

        self.batch_size = batch_size
        self.test_freq = test_freq
        self.save_freq = save_freq
        self.class_lr = class_lr

    def train_classifier(
        self,
        npi_training_models: NPITrainingModels,
        dataset_loader: NPIDatasetLoader,
        num_epochs,
        continue_epoch=None,
    ):
        num_epochs = num_epochs
        total_epochs = continue_epoch + num_epochs if continue_epoch else num_epochs
        # READ IN DATASET
        (
            train_loader,
            test_loader,
            train_len,
            test_len,
        ) = dataset_loader.load_train_and_test_dataloaders(self.batch_size)

        # MODEL INITIALIZATION
        classifier_model = npi_training_models.load_style_classifier(
            epoch=continue_epoch
        )

        print("Initializing class loss", flush=True)
        class_objective = torch.nn.BCELoss()
        class_optimizer = optim.Adam(classifier_model.parameters(), lr=self.class_lr)

        # PERFORM MODEL TRAINING
        class_epoch_losses = []
        class_batch_losses = []
        class_tests = []

        print("Training", flush=True)
        epochs = (
            range(continue_epoch, total_epochs) if continue_epoch else range(num_epochs)
        )
        for epoch in epochs:
            try:
                gc.collect()
                class_batch_losses = []
                loop = tqdm(total=train_len, position=0, leave=False)
                for batch, (orig_activ, real_label, _) in enumerate(train_loader):
                    classifier_model = classifier_model.train()

                    # Catch any infs!!!!!
                    if -np.inf in orig_activ:
                        raise ValueError("Found inf in array")
                    # prepare the batch for model processing
                    orig_activ, real_label = (
                        orig_activ.cuda(device=self.device).float(),
                        real_label.cuda(device=self.device).float(),
                    )

                    # UPDATE CLASSIFIER WEIGHTS
                    for p in classifier_model.parameters():
                        p.requires_grad = True

                    # Find labels and loss
                    class_optimizer.zero_grad()
                    class_loss = None
                    labels = classifier_model(orig_activ)
                    class_loss = (
                        class_objective(labels.squeeze(), real_label[:, 1].squeeze())
                        * 1e8
                    )  # gradient boosting constant
                    # backprop
                    class_loss.backward()
                    class_batch_losses.append(class_loss.item())  # append to losses
                    class_optimizer.step()

                    if batch % self.test_freq == 0:
                        test_loss, test_accuracy = self.test_model(
                            test_loader,
                            classifier_model,
                            class_objective,
                        )
                        # report current state to terminal
                        class_tests.append((epoch, test_loss))
                        training_loss = sum(class_batch_losses) / float(
                            len(class_batch_losses)
                        )
                        loop.set_description(
                            f"epoch: {epoch} train_loss={training_loss:.2f} test_loss={test_loss:.2f} test_accuracy={test_accuracy:.2f}"
                        )
                    loop.update()

                # record average loss for epoch
                if len(class_batch_losses) > 0:
                    class_epoch_losses.append(
                        (sum(class_batch_losses) / float(len(class_batch_losses)))
                    )
                else:
                    class_epoch_losses.append(np.nan)

                if epoch % self.save_freq == 0:
                    # save model
                    npi_training_models.save_style_classifier_model(epoch)
                    # save training info

            except Exception as e:
                print(e)
                torch.cuda.empty_cache()
                print("Epoch train loss history == ", class_epoch_losses)
                raise

        # save model after training
        npi_training_models.save_style_classifier_model(epoch)

        loop.close()
        print("Epoch train loss history == ", class_epoch_losses)
        gc.collect()

        return classifier_model

    def test_model(self, test_loader, classifier_model, class_objective, output=False):
        # perform npi_model testing
        class_test_losses = []
        class_test_accuracy = []
        # run testing loop
        for test_x, test_truth, sentence in test_loader:
            test_x: torch.Tensor
            test_truth: torch.Tensor

            test_x, test_truth = (
                test_x.cuda(device=self.device).float(),
                test_truth.cuda(device=self.device).float(),
            )

            # Test values
            # lhat like labels above
            lhat: torch.Tensor = classifier_model(test_x)
            class_loss = (
                class_objective(lhat.squeeze(), test_truth[:, 1].squeeze()) * 1e8
            )
            class_test_losses.append(class_loss.item())

            # Check for accuracy in batch
            test_truth = test_truth.argmax(dim=1)
            score = (test_truth == (lhat.squeeze() > 0.5)).sum().item()
            class_test_accuracy.append(score / self.batch_size)
            if output:
                with open(self.config.model_save_folder + "style_classifier_output.log", 'a') as f:
                    print(sentence[-1].replace('\n', '\\n'), file=f)
                    print(F"truth={test_truth[-1]} actual={lhat.squeeze()[-1]}\n", file=f)

        return np.mean(class_test_losses), np.mean(class_test_accuracy)
