import torch

import os
from npi.config import NPIConfig
from npi.models.classifiers import StyleClassifier, Discriminator
from npi.models.npi import GPT2LMWithNPI, NPINetwork
from npi.models.test_npi import generate_text, generate_text_with_NPI
from npi.transformers.modeling_gpt2 import GPT2LMHeadModel
from npi.transformers.tokenization_gpt2 import GPT2Tokenizer


class NPITrainingModels:
    def get_existing_models(self, prefix: str):
        model_list = [
            self.config.model_save_folder + filename
            for filename in os.listdir(self.config.model_save_folder)
            if filename.startswith(prefix)
        ]
        model_list.sort()
        return model_list

    def __init__(
        self,
        config: NPIConfig,
        npi_model_epoch=-1,
        discrim_model_epoch=-1,
        style_model_epoch=-1,
    ):
        """
        Initializes class with the configuration.
        Arguments:
            config: an NPIConfig object
            npi_model_epoch: the epoch number of the trained NPI to load. -1 signifies latest epoch.
            discrim_model_epoch: the epoch number of the trained discriminator to load. -1 signifies latest epoch.
            style_model_epoch:  the epoch number of the trained style classifier to load. -1 signifies latest epoch.
        """
        self.config = config
        self.npi_model_path = (
            self.config.model_save_folder + self.config.npi_name + "_npi_model"
        )
        self.style_classifier_path = (
            self.config.model_save_folder + self.config.npi_name + "_style_model"
        )
        self.discrim_model_path = (
            self.config.model_save_folder + self.config.npi_name + "_discrim_model"
        )
        self.input_activs_shape = (config.n, config.m, 1)
        self.input_targ_shape = (1, 1)

        self.npi_models = self.get_existing_models(self.config.npi_name + "_npi_model")
        self.style_models = self.get_existing_models(
            self.config.npi_name + "_style_model"
        )
        self.discrim_models = self.get_existing_models(
            self.config.npi_name + "_discrim_model"
        )
        self.npi_model_epoch = npi_model_epoch
        self.discrim_model_epoch = discrim_model_epoch
        self.style_model_epoch = style_model_epoch

        self.npi_model = None
        self.style_class_model = None
        self.discrim_model = None

        self.gpt2_with_npi = None
        self.gpt2_tokenizer = None

    def _load_model(
        self,
        model,
        epoch,
        new_epoch,
        model_path,
        model_name,
        model_initializer,
        require_pretrained=False,
    ):
        path = None
        if new_epoch:
            model = model_initializer()
            path = f"{model_path}_{new_epoch:03d}.pth"
        elif not model:
            model = model_initializer()
            if epoch == -1:
                existing_paths = self.get_existing_models(
                    self.config.npi_name + "_" + model_name
                )
                path = existing_paths[epoch] if existing_paths else None
            elif epoch:
                path = f"{model_path}_{epoch:03d}.pth"
        if path:
            print(f"Loading {model_name} weights from {path}")
            model.load_state_dict(torch.load(path, map_location="cpu"))
        elif require_pretrained:
            raise ValueError(
                f"{model_name} is not pretrained even though it is needed."
            )
        model.eval()
        return model.cuda(device=self.config.device)

    def load_npi_model(self, epoch=None) -> NPINetwork:
        self.npi_model = self._load_model(
            self.npi_model,
            self.npi_model_epoch,
            epoch,
            self.npi_model_path,
            "npi_model",
            lambda: NPINetwork(self.input_activs_shape).float(),
        )
        return self.npi_model

    def load_discriminator(self, epoch=None) -> Discriminator:
        self.discrim_model = self._load_model(
            self.discrim_model,
            self.discrim_model_epoch,
            epoch,
            self.discrim_model_path,
            "discrim_model",
            lambda: Discriminator(
                self.input_activs_shape, self.input_targ_shape
            ),
        )
        return self.discrim_model

    def load_style_classifier(self, epoch=None, require_pretrained=False) -> StyleClassifier:
        self.style_class_model = self._load_model(
            self.style_class_model,
            self.style_model_epoch,
            epoch,
            self.style_classifier_path,
            "style_model",
            lambda: StyleClassifier(self.config.n, self.config.m, 1).float(),
            require_pretrained,
        )
        return self.style_class_model

    def load_gpt2(self):
        if self.gpt2_with_npi and self.gpt2_tokenizer:
            return self.gpt2_with_npi, self.gpt2_tokenizer
        print(
            "Initializing GPT2WithNPI model with tokenizer -- not being placed on GPU until npi loss evaluation"
        )
        self.gpt2_with_npi = GPT2LMWithNPI.from_pretrained(
            self.config.gpt_model
        )  # lang model type may be 'gpt2' or 'gpt2-medium'
        self.gpt2_with_npi = self.gpt2_with_npi.cuda(device=self.config.device)
        self.gpt2_with_npi.initialize_npi(
            self.config.perturbation_indices, lang_model_type=self.config.gpt_model
        )
        self.gpt2_tokenizer = GPT2Tokenizer.from_pretrained(self.config.gpt_model)

        return self.gpt2_with_npi, self.gpt2_tokenizer

    def load_training_models(self):
        return (
            self.load_npi_model(),
            self.load_discriminator(),
            self.load_style_classifier(require_pretrained=True),
        )

    def save_models(self, epoch="_unk"):
        print("Saving NPI Model")
        out_path = f"{self.npi_model_path}_{epoch:03d}.pth"
        torch.save(self.npi_model.state_dict(), out_path)

        print("Saving GenerationClassifier Model")
        out_path = f"{self.discrim_model_path}_{epoch:03d}.pth"
        torch.save(self.discrim_model.state_dict(), out_path)

    def save_style_classifier_model(self, epoch="_unk"):
        print(f"Saving Style Classifier epoch {epoch}")
        out_path = f"{self.style_classifier_path}_{epoch:03d}.pth"
        torch.save(self.style_class_model.state_dict(), out_path)

    def gpt2_generate_text(
        self,
        in_text,
        lm_model=GPT2LMHeadModel.from_pretrained("gpt2"),
        num_generation_iters=100,
        max_seq_len=10,
        num_samples=1,
        top_k=1,
        top_p=0.0,
    ):
        _, gpt2_tokenizer = self.load_gpt2()
        return generate_text(
            in_text,
            lm_model,
            gpt2_tokenizer,
            num_generation_iters,
            max_seq_len,
            num_samples,
            top_k,
            top_p,
        )

    def npi_generate_text(
        self,
        in_text,
        vanilla_lm_model=GPT2LMHeadModel.from_pretrained("gpt2"),
        num_generation_iters=100,
        num_seq_iters=10,
        max_seq_len=10,
        num_samples=1,
        temperature=1,
        top_k=1,
        top_p=0.0,
    ):
        lm_model, gpt2_tokenizer = self.load_gpt2()
        npi_model = self.load_npi_model()
        return generate_text_with_NPI(
            in_text,
            lm_model,
            vanilla_lm_model,
            gpt2_tokenizer,
            self.config.perturbation_indices,
            npi_model,
            num_generation_iters,
            num_seq_iters,
            max_seq_len,
            num_samples,
            temperature,
            top_k,
            top_p,
        )
