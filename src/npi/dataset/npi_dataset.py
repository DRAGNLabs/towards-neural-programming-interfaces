import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader

from npi.config import NPIConfig

torch.manual_seed(1)

LM_CONFIG_DICT = {'gpt2': {'total_layers': 13, 'm': 768},
                'gpt2-medium': {'total_layers': 25, 'm': 1024}}

ORIG_ACTIV_INDEX = 0
ORIG_LABEL_INDEX = 1
TARG_LABEL_INDEX = 2
LANG_MODEL_INDEX = 3
META_DATA_INDEX = 4

def extract_needed_layers(array, pis, num_total_layers=13, seq_len=10, num_iters=10):
    """
    Accepts array of size (1300, 768, 1) (as an example)
        (really the size is ((num_total_layers*seq_len*num_iters), m, 1)
        and prediction indices
    Returns array of size (n, m, 1)

    * accepts and returns a numpy array
    """

    # First note that the pis may have some negative numbers in them - we fix this
    for i in range(len(pis)):
        if pis[i] < 0:
            pis[i] = num_total_layers + pis[i]  # Should be good now

    original_length = array.shape[0]
    assert original_length == num_total_layers * seq_len * num_iters  # Should contain all layers originally (recommended)
    all_layers_len_for_one_iter = original_length / num_iters
    # We construct a mask for the large array
    mask = np.array([False for _ in range(original_length)])
    for i in range(original_length):
        position = i % all_layers_len_for_one_iter
        corresponding_pi = position // seq_len
        if corresponding_pi in pis:
            mask[i] = True

    # Now we should have it
    extracted_layer_array = array[mask]
    # Shape should be (n, m, 1)
    return extracted_layer_array


class NPIDataSet(Dataset):
    def __init__(self, dataset, config: NPIConfig, return_row=False, permitted_rows=None, start_index=0):
        """
        Assumes input dataset is of the form:
            [[language_model_activations, 
              activations_classification, 
              target_classification (no longer used), 
              language_model_type, 
              meta_data,
              ...
            ],  
            ...]
        With objects of the following types:
            language_model_activations : nxmx1 ndarray representing flattened activation sequences (required)
            activations_classification : small ndarray representing the sentiment/content classification of the original activations (required)
            target_classification : (not required)
            language_model_type : str naming the language model being controlled (optional - assumed None)
            meta_data : dict recording desired metadata (required for NPI training later)
        """
        self.return_row = return_row
        self.seq_len = config.max_seq_len 
        self.num_iters = config.num_seq_iters  

        lm_config = LM_CONFIG_DICT[config.gpt_model]
        self.num_total_layers = lm_config['total_layers']
        self.m = lm_config['m']

        # Calculate n from pis
        self.n = len(config.perturbation_indices) * self.seq_len * self.num_iters  # could be 200

        # self.masking_coeff = 1e12

        if permitted_rows is None:
            self.dataset = dataset
        else:
            self.dataset = []
            for i in range(len(dataset)):
                if start_index + i in permitted_rows:
                    self.dataset.append(dataset[i])

        for i in range(len(self.dataset)):
            # mask the inf values in the activations to simply be VERY VERY LARGE values
            # self.dataset[i][ORIG_ACTIV_INDEX][self.dataset[i][ORIG_ACTIV_INDEX] == np.inf] = self.masking_coeff
            # self.dataset[i][ORIG_ACTIV_INDEX][self.dataset[i][ORIG_ACTIV_INDEX] == -1.*np.inf] = -1.*self.masking_coeff

            # cast everything as torch tensors, extract needed layers
            self.dataset[i][ORIG_ACTIV_INDEX] = torch.from_numpy(
                extract_needed_layers(
                    self.dataset[i][ORIG_ACTIV_INDEX], pis=config.perturbation_indices, num_total_layers=self.num_total_layers, seq_len=config.max_seq_len, num_iters=config.num_seq_iters)).double()
            self.dataset[i][ORIG_LABEL_INDEX] = torch.from_numpy(
                np.array(self.dataset[i][ORIG_LABEL_INDEX])).double()
            self.dataset[i][TARG_LABEL_INDEX] = torch.tensor([])  # empty tensor
        

    def __getitem__(self, i):
        acts = self.dataset[i][ORIG_ACTIV_INDEX]
        true_label = self.dataset[i][ORIG_LABEL_INDEX]
        targ = self.dataset[i][TARG_LABEL_INDEX]  # None
        return (acts, true_label, targ, i) if self.return_row else (acts, true_label, targ)

    def __len__(self):
        return len(self.dataset)

    def get_row_data(self, i):
        return self.dataset[i].copy()


class NPIDataLoader(DataLoader):
    def __init__(self, data, batch_size, pin_memory):
        super(NPIDataLoader, self).__init__(data, batch_size=batch_size, pin_memory=pin_memory, drop_last=True)
        self.data = data

    def get_row_data(self, dataset_indices):
        dataset_indices = dataset_indices.tolist()
        rows = []
        for index in dataset_indices:
            rows.append(self.data.get_row_data(index))
        return rows
