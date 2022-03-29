import os
import torch

LM_CONFIG_DICT = {'gpt2': {'total_layers': 13, 'm': 768},
                'gpt2-medium': {'total_layers': 25, 'm': 1024}}
class NPIConfig:
    def __init__(
        self,
        device: torch.device = None,
        gpt_model="gpt2",
        npi_type="adversarial",
        model_save_folder="models/npi_models/",
        dataset_folder="data/processed/",
        npi_name="cat_induction",
        batch_size=10,
        perturbation_indices=[5, 11],
        window_size=10,
        max_seq_len=10,
        num_seq_iters=10,
        top_k=1,
        top_p=0.9,
    ):
        self.gpt_model = gpt_model
        self.npi_type = npi_type
        self.model_save_folder = (
            model_save_folder
            if model_save_folder.endswith("/")
            else model_save_folder + "/"
        )
        self.dataset_folder = (
            dataset_folder if dataset_folder.endswith("/") else dataset_folder + "/"
        )
        os.makedirs(self.model_save_folder, exist_ok = True)
        os.makedirs(self.dataset_folder, exist_ok = True)

        self.npi_name = npi_name
        self.dataset_file = self.dataset_folder + self.npi_name + ".tar"
        self.batch_size = batch_size
        self.perturbation_indices = perturbation_indices
        self.window_size = window_size
        self.max_seq_len = max_seq_len
        self.num_seq_iters = num_seq_iters
        self.device = device

        self.top_k = top_k
        self.top_p = top_p

        lm_config = LM_CONFIG_DICT[gpt_model]
        self.num_total_layers = lm_config['total_layers']
        self.m = lm_config['m']

        # Calculate n from pis
        self.n = len(perturbation_indices) * max_seq_len * num_seq_iters  # could be 200

        self.k = 1 #TODO: configure this when actually in use.
