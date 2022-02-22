import torch
from npi.config import NPIConfig
from npi.models.classifiers import Classifier, GenerationClassifier
from npi.models.npi import GPT2LMWithNPI, NPINetwork
from npi.transformers.tokenization_gpt2 import GPT2Tokenizer


class NPITrainingModels:
    def __init__(
        self,
        config: NPIConfig,
        npi_model_path=None,
        content_classifier_path=None,
        generation_classifier_path=None,
    ):
        self.config = config
        self.npi_model_path = npi_model_path
        self.content_classifier_path = content_classifier_path
        self.generation_classifier_path = generation_classifier_path
        self.input_activs_shape = None  # TODO: Get hyperparam from config
        self.input_targ_shape = None  # TODO: Get hyperparam from config

        self.npi_model = None
        self.content_class_model = None
        self.generate_class_model = None

    def load_npi_model(self, reload=False):
        if reload or not self.npi_model:
            self.npi_model = NPINetwork(self.input_activs_shape).float()
            if self.npi_model_path is not None:
                self.npi_model.load_state_dict(
                    torch.load(self.npi_model_path, map_location="cpu")
                )
                self.npi_model.eval()
            self.npi_model.cuda(device=self.config.device)

        return self.npi_model

    def load_discriminator(self, reload=False):
        if reload or not self.generate_class_model:
            self.generate_class_model = GenerationClassifier(
                self.input_activs_shape, self.input_targ_shape
            ).float()
            if self.generation_classifier_path is not None:
                self.generate_class_model.load_state_dict(
                    torch.load(self.generation_classifier_path, map_location="cpu")
                )
                self.generate_class_model.eval()
            self.generate_class_model.cuda(device=self.config.device)

        return self.generate_class_model

    def load_content_classifier(self, reload=False):
        if reload or not self.content_class_model:
            self.content_class_model = Classifier()
            if self.content_classifier_path is not None:
                print("LOADING PRE-TRAINED CONTENT CLASSIFIER NETWORK")
                self.content_class_model.load_state_dict(
                    torch.load(
                        self.content_classifier_path, map_location=torch.device("cpu")
                    )
                )
                self.content_class_model.eval()
            else:
                raise NotImplementedError(
                    "Classifier should be pretrained. Pass in the path to the classifer."
                )
            self.content_class_model.cuda(device=self.config.device)

        return self.content_class_model

    def load_gpt2(self):
        print(
            "Initializing GPT2WithNPI model with tokenizer -- not being placed on GPU until npi loss evaluation"
        )
        gpt2_with_npi = GPT2LMWithNPI.from_pretrained(
            self.config.gpt_model
        )  # lang model type may be 'gpt2' or 'gpt2-medium'
        gpt2_with_npi = gpt2_with_npi.cuda(device=self.config.device)
        gpt2_with_npi.initialize_npi(
            self.config.perturbation_indices, lang_model_type=self.config.gpt_model
        )
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained(self.config.gpt_model)

        return gpt2_with_npi, gpt2_tokenizer

    def load_training_models(self):
        return (
            self.load_npi_model(),
            self.load_discriminator(),
            self.load_content_classifier(),
        )

    def save_models(self, epoch="_unk"):
        print("Saving NPI Model")
        out_path = f"{self.config.save_folder}{self.config.npi_type}_npi_network_epoch{epoch}.bin"
        torch.save(self.npi_model.state_dict(), out_path)

        print("Saving GenerationClassifier Model")
        out_path = (
            f"{self.config.save_folder}GenerationClassifier_network_epoch{epoch}.bin"
        )
        torch.save(self.generate_class_model.state_dict(), out_path)
