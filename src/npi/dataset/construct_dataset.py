import copy
import numpy as np
import torch
import torch.nn.functional as F

from tqdm import tqdm
import webdataset as wds
from npi.config import NPIConfig
from npi.transformers.modeling_gpt2 import GPT2LMHeadModel
from npi.transformers.tokenization_gpt2 import GPT2Tokenizer
from npi.utils import top_k_top_p_filtering


class NPIDatasetConstructor:
    def __init__(self, config: NPIConfig) -> None:
        self.save_file = config.dataset_file
        if "gpt2" in config.gpt_model:
            self.model = GPT2LMHeadModel.from_pretrained(config.gpt_model)
            self.tokenizer = GPT2Tokenizer.from_pretrained(config.gpt_model)
        else:
            raise NotImplementedError(f"model_name == {config.gpt_model} not supported")

        self.device = config.device
        if torch.cuda.is_available():
            print(f"Using cuda:{torch.cuda.current_device()}")
            print(f"gpu_device is {self.device}")
            self.model = self.model.cuda(device=self.device)

        self.window_size = config.window_size
        self.perturb_indicies = config.perturbation_indices
        self.num_iters = config.num_seq_iters
        self.max_iters = 5 * self.num_iters
        self.top_k = config.top_k
        self.top_p = config.top_p

    def construct_target_word_dataset(self):
        # TODO: Logic for creating target word dataset.
        pass

    def construct_dataset(self, data_iter, data_len):
        sink = wds.TarWriter(self.save_file)

        # necessary to pull activation tensors
        self.model.transformer.output_hidden_states = True

        pbar = tqdm(
            data_iter, total=data_len, mininterval=10, maxinterval=30, miniters=100
        )
        try:
            for index, (line, cls) in enumerate(pbar):
                if index == data_len:  # Break if reached amount of data to generate.
                    break

                # clean line to some extent
                #   (due to possible differences in corpora that could tip off the classifer)
                line = line.lower().strip().strip(".").strip()
                if len(line.split()) > 100 or len(line.split()) < 4:
                    continue

                big_array = []  # nxmx1

                tokens = self.tokenizer.encode(line)
                tokens = tokens[-self.window_size :]
                num_tokens_needed = self.window_size - len(tokens)
                tokens = torch.tensor(tokens, dtype=torch.long, device=self.device)
                tokens = tokens.unsqueeze(0).repeat(1, 1)
                tokens = tokens.cuda(device=self.device)
                all_text_tokens = copy.deepcopy(tokens)

                # some constants to set first
                len_for_big_array = len(self.perturb_indicies) * self.num_iters

                # We loop through multiple times now
                purely_generated_tokens = []  # haven't generated anything yet
                i = -1
                while True:
                    i += 1

                    tokens, next_token = self.generate_tokens(big_array, tokens)
                    purely_generated_tokens = (
                        purely_generated_tokens + next_token.tolist()
                    )
                    all_text_tokens = torch.cat(
                        (all_text_tokens, next_token.unsqueeze(0)), dim=1
                    ).cuda(self.device)

                    if (  # TODO Figure out this condition. It can vary.
                        len(big_array) >= len_for_big_array
                        and len(all_text_tokens.squeeze())
                        >= self.num_iters + self.window_size
                    ):
                        break

                num_gpt2_iters_run = i + 1
                big_array = big_array[-len_for_big_array:]

                # TODO: This currently only allows binary classification
                # figure out true classification
                orig_classification = np.zeros(2)
                if cls == 0:
                    orig_classification[0] = 1
                elif cls == 1:
                    orig_classification[1] = 1
                else:
                    raise RuntimeError("Got a score that is not 0 or 1")

                # What will we call "original text" and "generated text"

                assert (
                    all_text_tokens.squeeze().tolist()[-self.window_size :]
                    == tokens.squeeze().tolist()
                )

                orig_text_tokens = all_text_tokens[
                    :, -self.window_size - self.num_iters : -self.num_iters
                ]  # sent_len tokens that produced generated_text_tokens
                generated_text_tokens = tokens

                orig_tokens = orig_text_tokens.squeeze().tolist()
                gpt2_generated_tokens = generated_text_tokens.squeeze().tolist()

                orig_text = self.tokenizer.decode(orig_tokens)
                gpt2_generated_text = self.tokenizer.decode(gpt2_generated_tokens)

                # Now the big_array is a list of length (num_iters*len(perturb_indicies)) of tensors with shape (1,sent_len,emb_dim)
                big_array = torch.cat(big_array, dim=1)
                big_array = big_array.permute(
                    1, 2, 0
                )  # shape is (2*sent_len*num_iters, emb_dim, 1) now, emb_dim will be 1024 or 768
                big_array = big_array.data.cpu().numpy()

                sink.write(
                    {
                        "__key__": "sample%06d" % index,
                        "orig_activ.npy": big_array,
                        "orig_label.npy": orig_classification,
                        "orig.txt": orig_text,
                        "generated.txt": gpt2_generated_text,
                        "target.txt": "target words",  # TODO: This may not be needed, or needed for word avoidance/induction
                        "orig_tokens.pyd": orig_tokens,
                        "metadata.pyd": {
                            "num_gpt2_iters": num_gpt2_iters_run,
                            "gpt2_generated_tokens": gpt2_generated_tokens,
                        },
                    }
                )

                # TODO: Print out how many of each classified data is constructed.
                pbar.update()

        except:
            raise
        finally:
            torch.cuda.empty_cache()
            print("Closing dataset writer")
            sink.close()

        torch.cuda.empty_cache()
        print(" ", flush=True)
        print("done")

    def generate_tokens(self, big_array, tokens):
        # Now run the model
        hidden_states, _, all_hiddens = self.model(
            input_ids=tokens[:, -self.window_size :]
        )  # all_hiddens is a list of len

        # 25 or 13 with tensors of shape (gpt2 medium or small)
        # (1,sent_len,1024) or (1,sent_len,768)
        # Add to big_array
        if tokens.shape[1] >= self.window_size:
            for pi in self.perturb_indicies:
                big_array.append(all_hiddens[pi].data)

        # Now we extract the new token and add it to the list of tokens
        next_token_logits = hidden_states[0, -1, :]
        filtered_logits = top_k_top_p_filtering(
            next_token_logits, top_k=self.top_k, top_p=self.top_p
        )
        next_token = torch.multinomial(
            F.softmax(filtered_logits, dim=-1), num_samples=1
        )

        # ...update list of tokens
        if tokens.shape[1] < self.window_size:
            tokens = torch.cat((tokens, next_token.unsqueeze(0)), dim=1).cuda(
                self.device
            )
        else:
            tokens = torch.cat((tokens[:, 1:], next_token.unsqueeze(0)), dim=1).cuda(
                self.device
            )

        return tokens, next_token
