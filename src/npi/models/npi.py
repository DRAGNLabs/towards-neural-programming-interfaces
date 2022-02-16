import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss

from npi.transformers import GPT2Model, GPT2LMHeadModel
from npi.utils import top_k_top_p_filtering

# NPI Neural Model Code -------------------------------------------------------------------------------
class NPINetwork(nn.Module):
    def __init__(self, input_activs_shape):
        """
        input_activs_shape: tuple of (n, m, 1)
            n x m x 1 slices contain the elements of the original activations, flattened into a 2D array
        """
        super(NPINetwork, self).__init__()
        print("NPI INITIALIZATION")
        self.n = input_activs_shape[0]
        self.m = input_activs_shape[1]
        self.k = input_activs_shape[2]

        # Setting Scaling Factors
        fact1 = 2**2
        fact2 = 2**3
        fact3 = 2**3

        # Defining first npi layer
        self.first_linear = nn.Sequential(
            nn.Linear((self.n) * self.m * self.k, self.n // fact1),
            nn.ReLU(),
        )
        self.second_linear = nn.Sequential(
            nn.Linear(self.n // fact1, self.n // fact1),
            nn.ReLU(),
        )
        self.third_linear = nn.Sequential(
            nn.Linear(self.n // fact1, self.n // fact2),
            nn.ReLU(),
        )
        self.fourth_linear = nn.Sequential(
            nn.Linear(self.n // fact2, self.n // fact2),
            nn.ReLU(),
        )
        self.fourth_linear_residual = nn.Sequential(
            nn.Linear(self.n // fact2, self.n // fact3),
            nn.ReLU(),
        )
        self.fifth_linear = nn.Sequential(
            nn.Linear(self.n // fact3, self.n // fact2),
            nn.ReLU(),
        )
        self.sixth_linear = nn.Sequential(
            nn.Linear(self.n // fact2, self.n // fact1),
            nn.ReLU(),
        )
        self.seventh_linear = nn.Sequential(
            nn.Linear(self.n // fact1, self.n // fact1),
            nn.ReLU(),
        )
        self.last_linear = nn.Sequential(
            nn.Linear(self.n // fact1, self.n * self.m * self.k),
        )

    def forward(self, orig_activs):
        metadata = {
            "ordered_hidden_activations": [],
            "final_out_preview": None,
            "final_out_returned": None,
            "concatenated_input": None,
        }
        combined = orig_activs  # torch.cat((target_label, orig_activs), dim=1)
        first_out = self.first_linear(combined.view(-1, (self.n) * self.m * self.k))
        second_out = self.second_linear(first_out)
        third_out = self.third_linear(second_out)
        fourth_out = self.fourth_linear(third_out)
        # fourth_out_resid = self.fourth_linear_residual(third_out+fourth_out)
        # fifth_out = self.fifth_linear(fourth_out_resid)
        # sixth_out = self.sixth_linear(third_out+fifth_out)
        # seventh_out = self.seventh_linear(second_out+sixth_out)
        # out_linear = self.last_linear(first_out+seventh_out)
        fourth_out_resid = self.fourth_linear_residual(fourth_out)
        fifth_out = self.fifth_linear(fourth_out_resid)
        sixth_out = self.sixth_linear(fifth_out)
        seventh_out = self.seventh_linear(sixth_out)
        out_linear = self.last_linear(seventh_out)
        final_out = out_linear.view(-1, self.n, self.m, self.k)

        # metadata['ordered_hidden_activations'] = [first_out.detach().data.cpu().numpy(),
        #                                          second_out.detach().data.cpu().numpy(),
        #                                          third_out.detach().data.cpu().numpy(),
        #                                          fourth_out.detach().data.cpu().numpy(),
        #                                          fourth_out_resid.detach().data.cpu().numpy(),
        #                                          fifth_out.detach().data.cpu().numpy(),
        #                                          sixth_out.detach().data.cpu().numpy(),
        #                                          seventh_out.detach().data.cpu().numpy(),
        #                                          ]
        # metadata['final_out_preview'] = out_linear.detach().data.cpu().numpy()
        # metadata['final_out_returned'] = final_out.detach().data.cpu().numpy()
        # metadata['concatenated_input'] = combined.detach().data.cpu().numpy()

        return final_out  # , metadata


class GPT2WithNPI(GPT2Model):
    r"""
    Modified from GPT2Model class in transformers module (from HuggingFace)

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the last layer of the model.
        **past**:
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            that contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2Model.from_pretrained('gpt2')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

    """

    def __init__(self, config):  # NPI added functionality
        super(GPT2WithNPI, self).__init__(config)  # NPI added functionality

        GPT2Model.__init__(self, config)  # NPI added functionality

    def initialize_npi(self, prediction_indices):
        self.perturbation_indices = prediction_indices  # NPI added functionality
        self.output_hidden_states = True

    def forward(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        activation_perturbations=None,
    ):
        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        if position_ids is None:
            position_ids = torch.arange(
                past_length,
                input_ids.size(-1) + past_length,
                dtype=torch.long,
                device=input_ids.device,
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        # Attention mask.
        if attention_mask is not None:
            attention_mask = attention_mask.view(-1, input_shape[-1])
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
            # this attention mask is more simple than the triangular masking of causal attention
            # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            attention_mask = attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (
                    head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                )
                head_mask = head_mask.expand(self.config.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.n_layer

        inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = ()
        all_attentions = []
        all_hidden_states = ()
        # print("GPT2WithNPI: Total num layers == ", len(self.h))
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (
                    hidden_states.view(*output_shape),
                )

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
            )

            hidden_states, present = outputs[:2]
            for j, index in enumerate(self.perturbation_indices):
                if i == index:
                    # GPT2 MODEL: perturbing activation layer == i
                    # GPT2 MODEL: hidden_states size == hidden_states.size()
                    # GPT2 MODEL: activation_perturbations[j] size == activation_perturbations[j].size()
                    hidden_states = hidden_states + activation_perturbations[j]
            presents = presents + (present,)

            if self.output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states, presents)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            # let the number of heads free (-1) so we can extract attention even after head pruning
            attention_output_shape = (
                input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
            )
            all_attentions = tuple(
                t.view(*attention_output_shape) for t in all_attentions
            )
            outputs = outputs + (all_attentions,)
        return outputs  # last hidden state, presents, (all hidden_states), (attentions)


class GPT2LMWithNPI(GPT2LMHeadModel):
    r"""
    Modified from GPT2LMHeadModel class in transformers module (from HuggingFace)

        **labels**: (`optional`) ``torch.LongTensor`` of shape ``(batch_size, sequence_length)``:
            Labels for language modeling.
            Note that the labels **are shifted** inside the model, i.e. you can set ``lm_labels = input_ids``
            Indices are selected in ``[-1, 0, ..., config.vocab_size]``
            All labels set to ``-1`` are ignored (masked), the loss is only
            computed for labels in ``[0, ..., config.vocab_size]``

    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **loss**: (`optional`, returned when ``labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Language modeling loss.
        **prediction_scores**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, config.vocab_size)``
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        **past**:
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            that contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.

    Examples::

        import torch
        from transformers import GPT2Tokenizer, GPT2LMHeadModel

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=input_ids)
        loss, logits = outputs[:2]

    """

    def __init__(self, config):  # , npi_config):
        super(GPT2LMWithNPI, self).__init__(config)

        GPT2LMHeadModel.__init__(self, config)  # NPI added functionality

    def initialize_npi(self, prediction_indices, lang_model_type="gpt2"):
        self.perturbation_indices = prediction_indices  # NPI added functionality
        # self.output_hidden_states = True
        self.transformer = GPT2WithNPI.from_pretrained(
            lang_model_type
        )  # (config, self.npi, self.prediction_indices) # NPI added functionality
        self.transformer.initialize_npi(prediction_indices)
        self.npi_model = None

    def forward(
        self,
        input_ids,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
        activation_perturbations=None,
    ):
        """
        target_classification : nx1x1 target classification vector  # NPI added functionality
        """
        transformer_outputs = self.transformer(
            input_ids,
            past=past,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            activation_perturbations=activation_perturbations,
        )  # NPI added functionality
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )
            outputs = (loss,) + outputs

        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)

    def obtain_perturbed_GPT2WithNPI_outputs(
        self,
        npi_batched_perturbations,
        perturbation_indices,
        data_rows,
        tokenizer=None,
        max_seq_len=10,
        num_seq_iters=10,
        device=None,
    ):
        # obtain perturbed GPT2WithNPI outputs: START
        LANG_MODEL_ACTS_IND = 0
        ACTS_CLASSIF_IND = 1
        TARG_CLASSIF_IND = 2
        LANG_MODEL_TYPE_IND = 3
        META_DATA_IND = 4
        ORIG_TEXT_IND = 5
        PRED_TEXT_IND = 6
        TARG_TEXT_INDEX = 7
        GPT2_TEXT_INDEX = 8  # the text of what the gpt2 actually produced
        top_k = 1
        top_p = 0.9
        temperature = 1.0
        masking_coeff = 1e12

        batched_deltas_shape = npi_batched_perturbations.size()
        b = batched_deltas_shape[0]
        n = batched_deltas_shape[1]
        m = batched_deltas_shape[2]
        k = batched_deltas_shape[3]

        gpt2_perturbed_outs = []
        npi_resulting_text = []
        # iterating over batches
        for j in range(b):
            # create input_ids
            tokens = data_rows[j][META_DATA_IND]["orig_tokens"]
            tokens = torch.tensor(tokens, dtype=torch.long)  # , device=device)
            tokens = tokens.unsqueeze(0).repeat(1, 1)
            tokens = tokens.cuda(device=device)

            # create list of un-flattened activation_perturbations from current batch elem
            # creating curr_perturbs
            reshaped = npi_batched_perturbations[j, :, :, 0].view(1, n, m, 1)
            # chunking with reshaped size == reshaped.size()
            chunked = torch.chunk(
                reshaped, num_seq_iters * len(self.perturbation_indices), dim=1
            )
            # ^ each hidden layer in the hugging face repo has shape (batch, seq_len, hidden_size)
            # casting chunked as list
            curr_perturbs = [x.view(1, max_seq_len, m) for x in chunked]

            # initializing big_array
            #   obtain flattened representation of the resulting perturbed forward pass in GPT-2
            big_array = []  # nxmx1
            sent = data_rows[j][ORIG_TEXT_IND]
            generated_sent = ""
            # iteratively producing big_array
            for i in range(num_seq_iters):

                # Now run the model
                logits, presents, all_hiddens = self.forward(
                    input_ids=tokens[:, -max_seq_len:],
                    activation_perturbations=curr_perturbs[
                        i
                        * len(self.perturbation_indices) : (i + 1)
                        * len(self.perturbation_indices)
                    ],
                )
                # all_hiddens is a list of len
                # 25 or 13 with tensors of shape (gpt2 medium of small)
                # (1,sent_len,1024) or (1,sent_len,768)
                # Add to big_array
                for index in self.perturbation_indices:
                    big_array.append(all_hiddens[index])  # .data)

                # Now we extract the new token and add it to the list of tokens
                next_token_logits = logits[0, -1, :] / temperature
                filtered_logits = top_k_top_p_filtering(
                    next_token_logits, top_k=top_k, top_p=top_p
                )
                next_token = torch.multinomial(
                    F.softmax(filtered_logits, dim=-1), num_samples=1
                )
                next_token_list = next_token.tolist()
                next_word = tokenizer.decode(next_token_list)
                sent = (
                    sent + " " + next_word
                )  # we just update this so sent remains accurate for dict
                generated_sent = generated_sent + next_word + " "

                # ...update list of tokens
                tokens = torch.cat((tokens, next_token.unsqueeze(0)), dim=1).cuda()
            if tokenizer is not None:
                npi_sent_for_data_set = tokenizer.decode(
                    [x.item() for x in tokens[:, -max_seq_len:].flatten()]
                )
                npi_resulting_text.append(
                    [
                        data_rows[j][ORIG_TEXT_IND],
                        data_rows[j][GPT2_TEXT_INDEX],
                        data_rows[j][TARG_TEXT_INDEX],
                        npi_sent_for_data_set,
                        sent,
                    ]
                )

            del tokens

            # Now the big_array is a list of length (max_seq_len*2) of tensors with shape (1,max_seq_len,1024) or (1,max_seq_len,768)
            # completing big_array
            big_array = torch.cat(big_array, dim=1)
            big_array = big_array.permute(1, 2, 0).view(1, n, m, 1)

            # mask the inf values in the activations to simply be VERY VERY LARGE values
            # big_array[big_array == float("Inf")] = masking_coeff
            # big_array[big_array == -1.*float("Inf")] = -1.*masking_coeffs

            # store for later concatenation
            gpt2_perturbed_outs.append(big_array)
            #   iteration stop
        # create the end-result of npi_perturbations
        # casting output as single torch tensor
        resulting_gpt2_activations = torch.cat(gpt2_perturbed_outs, dim=0)
        # obtain perturbed GPT2WithNPI outputs: STOP
        return resulting_gpt2_activations, npi_resulting_text  # this is batched
