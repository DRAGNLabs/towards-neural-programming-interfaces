from copy import deepcopy
import gc
import time
import numpy as np
import pickle as pkl

import torch
import torch.optim as optim


from tqdm import tqdm
from npi.dataset.npi_dataset import NPIDataLoader, NPIDatasetLoader

from npi.config import NPIConfig
from npi.models import NPITrainingModels
from npi.training import NPILoss
from npi.visualization import make_training_plots


def accuracy_by_nate(x, y):
    """
    Accepts vector of ground truth labels and vector of generated labels
    Order not important, as long as dims are equal
        x, y are both 1-dim torch.Tensor objects or np.ndarray
    """
    x, y = x.squeeze().data.cpu().numpy(), y.squeeze().data.cpu().numpy()
    x = np.array([round(xi) for xi in x])
    y = np.array([round(yi) for yi in y])
    if len(x) != 0:
        return len(x[x == y]) / len(x)
    else:
        return 0.0


# TODO: Make generator optional
class NPITrainer:
    def __init__(
        self,
        config: NPIConfig,
        save_freq=10,
        test_freq=5,
        batch_size=5,
        headstart=5,
        loss_boosting_coeff=10000.0,
        discrim_coeff=3.0,
        style_coeff=10.0,
        similarity_coeff=1.0,
        npi_lr=1e-6,
        disc_lr=1e-6,
    ):
        # Set hyperparameters
        self.config = config
        self.save_freq = save_freq
        self.test_freq = test_freq
        self.batch_size = batch_size
        self.headstart = headstart
        self.loss_boosting_coeff = loss_boosting_coeff
        self.discrim_coeff = discrim_coeff
        self.style_coeff = style_coeff
        self.similarity_coeff = similarity_coeff
        self.npi_lr = npi_lr
        self.disc_lr = disc_lr

        # Initialize loss functions
        self.npi_objective = NPILoss(
            discrim_coeff, style_coeff, similarity_coeff, loss_boosting_coeff
        )
        self.generate_class_objective = torch.nn.BCELoss()
        self.bce_loss = torch.nn.BCELoss()
        self.mse_loss = torch.nn.MSELoss()
        self.list_average = lambda x: sum(x) / float(len(x))

        # Initialize data structures to store training results
        self.train_metadata = {
            "npi_losses": [],
            "generate_class_losses": [],
            "generate_class_accuracies": [],
        }
        self.train_batch_metadata = deepcopy(self.train_metadata)

        self.test_metadata = {
            "npi_test_losses": [],
            "content_class_test_losses": [],
            "generate_class_tests_losses": [],
            "generate_false_class_test_losses": [],
            "content_class_test_accuracies": [],
            "generate_class_test_accuracies": [],
            "generate_class_false_test_accuracies": [],
        }
        self.test_batch_metadata = deepcopy(self.test_metadata)

        # Seed torch
        torch.manual_seed(1)

    def train_generator_step(self, orig_activ, pred_gpt2_outs):
        self.generate_class_model.train()

        for p in self.npi_model.parameters():
            p.requires_grad = False
        for p in self.generate_class_model.parameters():
            p.requires_grad = True

        self.generate_class_model.zero_grad()  # generate_class_optimizer.zero_grad()

        # labels
        y_real_GPT2 = torch.zeros(self.batch_size).float().cuda()  # 0 = real GPT2
        y_fake_GPT2 = torch.ones(self.batch_size).float().cuda()  # 1 = fake GPT2
        # y_real_GPT2, y_fake_GPT2 = Variable(y_real_GPT2), Variable(y_fake_GPT2)

        # Now predict and get loss
        real_gen_pred = self.generate_class_model(orig_activ)
        fake_gen_pred = self.generate_class_model(pred_gpt2_outs.detach())
        # loss
        real_loss = self.generate_class_objective(
            real_gen_pred.squeeze(), y_real_GPT2.squeeze()
        )
        fake_loss = self.generate_class_objective(
            fake_gen_pred.squeeze(), y_fake_GPT2.squeeze()
        )
        g_class_loss = self.loss_boosting_coeff * (real_loss + fake_loss)

        g_class_loss.backward()
        self.generate_class_optimizer.step()

        return g_class_loss.item()

    def train_npi_step(self, orig_activ, pred_gpt2_outs):
        self.npi_model.train()

        for p in self.npi_model.parameters():
            p.requires_grad = True
        for p in self.generate_class_model.parameters():
            p.requires_grad = False

        self.npi_model.zero_grad()  # npi_optimizer.zero_grad()

        self.npi_objective.generation_classifier_model = self.generate_class_model

        # labels
        y_word = (
            torch.ones(self.batch_size).float().cuda()
        )  # ones here corresponds to having NO sexist slurs
        y_real_GPT2 = torch.zeros(self.batch_size).float().cuda()

        # get classifications and loss
        content_classification = self.content_class_model(pred_gpt2_outs)
        gen_classification = self.generate_class_model(pred_gpt2_outs)
        # loss
        discrim_loss = self.bce_loss(
            gen_classification.squeeze(), y_real_GPT2.squeeze()
        )
        style_loss = self.bce_loss(content_classification.squeeze(), y_word.squeeze())
        similarity_loss = self.mse_loss(pred_gpt2_outs, orig_activ)
        npi_loss = self.loss_boosting_coeff * (
            self.discrim_coeff * discrim_loss
            + self.style_coeff * style_loss
            + self.similarity_coeff * similarity_loss
        )

        npi_loss.backward()
        self.npi_optimizer.step()

        return npi_loss.item()

    def get_pred_gpt2_outs(self, orig_tokens, orig_text, generated_text, pred_activs):
        return self.gpt2_with_npi.obtain_perturbed_GPT2WithNPI_outputs(
            pred_activs,
            orig_tokens,
            orig_text,
            generated_text,
            tokenizer=self.gpt2_tokenizer,
            max_seq_len=self.config.max_seq_len,
            num_seq_iters=self.config.num_seq_iters,
            device=self.config.device,
        )

    def test_npi(self, test_loader):
        # print("Testing: START")
        # perform npi_model testing
        self.npi_model.eval()
        self.generate_class_model.eval()

        for orig_activ, orig_tokens, orig_text, generated_text in test_loader:

            # Now we know functional_batch_size == batch_size
            y_real_GPT2 = torch.zeros(self.batch_size).float().cuda()  # 0 = real GPT2
            y_fake_GPT2 = torch.ones(self.batch_size).float().cuda()  # 1 = fake GPT2
            y_word = torch.ones(self.batch_size).float().cuda()

            orig_activ = orig_activ.cuda(non_blocking=True).float()
            orig_tokens = orig_tokens.cuda(non_blocking=True)

            test_deltas = self.npi_model(orig_activ)
            test_gpt2_outs, test_text = self.get_pred_gpt2_outs(orig_tokens, orig_text, generated_text, test_deltas)

            test_real_gen_pred = self.generate_class_model(orig_activ)
            test_fake_gen_pred = self.generate_class_model(test_gpt2_outs)
            test_real_gen_loss = self.generate_class_objective(
                test_real_gen_pred.squeeze(), y_real_GPT2.squeeze()
            )
            test_fake_gen_loss = self.generate_class_objective(
                test_fake_gen_pred.squeeze(), y_fake_GPT2.squeeze()
            )
            test_g_class_loss = self.loss_boosting_coeff * (
                test_real_gen_loss + test_fake_gen_loss
            )

            # append losses and get accuracy
            self.test_batch_metadata["generate_class_tests_losses"].append(
                test_g_class_loss.item()
            )  # note this is the sum of real and fake loss
            self.test_batch_metadata["generate_false_class_test_losses"].append(
                test_fake_gen_loss.item()
            )
            test_real_gen_acc = accuracy_by_nate(
                test_real_gen_pred.squeeze(), y_real_GPT2.squeeze()
            )
            test_fake_gen_acc = accuracy_by_nate(
                test_fake_gen_pred.squeeze(), y_fake_GPT2.squeeze()
            )
            test_avg_gen_acc = (test_real_gen_acc + test_fake_gen_acc) / 2.0
            self.test_batch_metadata["generate_class_test_accuracies"].append(
                test_avg_gen_acc
            )
            self.test_batch_metadata["generate_class_false_test_accuracies"].append(
                test_fake_gen_acc
            )

            test_content_classification = self.content_class_model(test_gpt2_outs)
            test_gen_classification = test_fake_gen_pred
            test_discrim_loss = self.bce_loss(
                test_gen_classification.squeeze(), y_real_GPT2.squeeze()
            )
            test_style_loss = self.bce_loss(
                test_content_classification.squeeze(), y_word.squeeze()
            )
            test_similarity_loss = self.mse_loss(test_gpt2_outs, orig_activ)
            test_npi_loss = self.loss_boosting_coeff * (
                self.discrim_coeff * test_discrim_loss
                + self.style_coeff * test_style_loss
                + self.similarity_coeff * test_similarity_loss
            )
            # append losses and get accuracy
            self.test_batch_metadata["npi_test_losses"].append(test_npi_loss.item())
            # Don't forget the accuracy number from the classifier
            acc_from_content_class = accuracy_by_nate(
                test_content_classification.squeeze(), y_word.squeeze()
            )
            self.test_batch_metadata["content_class_test_accuracies"].append(
                acc_from_content_class
            )

    def visualize_training(self):
        make_training_plots(
            "NPI average loss per epoch",
            self.config.model_save_folder,
            self.train_metadata["npi_losses"],
            self.test_metadata["npi_test_losses"],
        )
        make_training_plots(
            "Discriminator average loss per epoch",
            self.config.model_save_folder,
            self.train_metadata["generate_class_losses"],
            self.test_metadata["generate_class_tests_losses"],
            self.test_metadata["generate_false_class_test_losses"],
        )
        make_training_plots(
            "Discriminator average accuracy per epoch",
            self.config.model_save_folder,
            self.train_metadata["generate_class_accuracies"],
            self.test_metadata["generate_class_test_accuracies"],
            self.test_metadata["generate_class_false_test_accuracies"],
        )

    def train_adversarial_npi(
        self,
        npi_training_models: NPITrainingModels,
        num_epochs,
        dataset_loader: NPIDatasetLoader,
    ):

        # Initialize model
        (
            self.npi_model,
            self.generate_class_model,
            self.content_class_model,
        ) = npi_training_models.load_training_models()

        # TODO: See if it is okay to initialize gpt2 here instead of every loop
        self.gpt2_with_npi, self.gpt2_tokenizer = npi_training_models.load_gpt2()
        self.gpt2_with_npi.npi_model = self.npi_model
        self.gpt2_with_npi.cuda(device=self.config.device)

        # Initialize Data to train
        (
            self.train_loader,
            self.test_loader,
            train_len,
            test_len,
        ) = dataset_loader.load_train_and_test_dataloaders(self.batch_size)

        # Initialize optimizer
        self.npi_optimizer = optim.Adam(self.npi_model.parameters(), lr=self.npi_lr)
        self.generate_class_optimizer = optim.Adam(
            self.generate_class_model.parameters(), lr=self.disc_lr
        )

        # Set up NPILoss
        self.npi_objective.content_classifier_model = self.content_class_model
        self.npi_objective.generation_classifier_model = self.generate_class_model

        print("Training")

        for epoch in range(num_epochs):
            gc.collect()
            print("############ Epoch == ", epoch, " ############")

            # Looping through training batches
            loop = tqdm(total=train_len, position=0, leave=False)
            for (
                orig_activ,
                orig_tokens,
                orig_text,
                generated_text,
            ) in self.train_loader:
                orig_activ = orig_activ.cuda(non_blocking=True).float()

                self.npi_model.eval()  # TODO: See if this is needed
                pred_gpt2_outs, _ = self.get_pred_gpt2_outs(
                    orig_tokens, orig_text, generated_text, self.npi_model(orig_activ)
                )

                g_class_loss_item = None
                if epoch >= self.headstart:
                    g_class_loss_item = self.train_generator_step(
                        orig_activ, pred_gpt2_outs
                    )
                    self.train_batch_metadata["generate_class_losses"].append(
                        g_class_loss_item
                    )

                npi_loss_item = self.train_npi_step(orig_activ, pred_gpt2_outs)
                self.train_batch_metadata["npi_losses"].append(npi_loss_item)

                if g_class_loss_item is not None:
                    loop.set_description(
                        f"epoch:{epoch}, gen_class_loss:{g_class_loss_item:.2f}, npi_loss:{npi_loss_item:.2f}"
                    )
                else:
                    loop.set_description(
                        f"epoch:{epoch}, gen_class_loss:N/A, npi_loss:{npi_loss_item:.2f}"
                    )
                loop.update(1)

            # collect training averages for meta data
            for key, value in self.train_metadata.items():
                if self.train_batch_metadata[key]:
                    value.append(
                        (epoch, self.list_average(self.train_batch_metadata[key]))
                    )

            # TESTING
            if epoch % self.test_freq == 0 and epoch >= self.headstart:
                print("Testing")
                self.test_npi(self.test_loader)
                for key, value in self.test_metadata.items():
                    # collect training averages for meta data
                    if self.test_batch_metadata[key]:
                        value.append(
                            (epoch, self.list_average(self.test_batch_metadata[key]))
                        )
                out_path = f"{self.config.model_save_folder}{self.config.npi_type}_npi_train_summaries_epoch{epoch}.pkl"
                with open(out_path, "wb") as outfile:
                    pkl.dump(self.train_metadata, outfile)
                out_path = f"{self.config.model_save_folder}npi_test_summaries_epoch{epoch}.pkl"
                with open(out_path, "wb") as outfile:
                    pkl.dump(self.test_metadata, outfile)

            # report current state to terminal
            torch.cuda.empty_cache()

            print("end of regular epoch")

            if epoch % self.save_freq == 0 and epoch >= self.headstart:
                npi_training_models.save_models(epoch)
                self.visualize_training()
        torch.cuda.empty_cache()
        loop.close()
