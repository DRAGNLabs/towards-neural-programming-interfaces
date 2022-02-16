#           Some objects defined here               #
#          and needed in other scripts              #
#                                                   #
#          Fulda, Brown, Wingate, Robinson          #
#                       DRAGN                       #
#                    NPI Project                    #
#                       2020                        #

#                    !!!NOTE!!!                     #
# This script not as well-polished as others.
# A lot here is extraneous.
# Don't read through this script to understand our modeling.
# (Refer to the others in this repo
# ~ Dec 2020

"""
Neural Program Interfaces

NOTE:
    target_classification functionality in all classes other than the NeuralProgramInterface class is decrimented
     - the NPI functions by caching activations, and thus will not use target_classification in real time 
       forward() calls
"""
import os
import pickle as pkl

# from modeling_gpt2 import GPT2Model, GPT2LMHeadModel
# from modeling_xlnet import XLNetModel, XLNetLMHeadModel
# from modeling_transfo_xl import TransfoXLModel, TransfoXLLMHeadModel
# import modeling_bert
import torch
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from tqdm import trange
from npi.transformers import GPT2LMHeadModel, GPT2Model, GPT2Tokenizer
#from transformers.models.gpt2 import GPT2Tokenizer

torch.manual_seed(1)


class GPT2WithNPI(GPT2Model):
    r"""
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

        # self.npi = npi # NPI added functionality
        # self.prediction_indices = prediction_indices # NPI added functionality

        GPT2Model.__init__(self, config)  # NPI added functionality
        pass

    def initialize_npi(self, npi, prediction_indices):
        self.npi = npi  # NPI added functionality
        self.prediction_indices = prediction_indices  # NPI added functionality
        pass

    def forward(self, input_ids, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                target_classification=None):
        """
        target_classification : nx1x1 target classification vector  # NPI added functionality
        """
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
            position_ids = torch.arange(past_length, input_ids.size(-1) + past_length, dtype=torch.long,
                                        device=input_ids.device)
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
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.n_layer, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(
                    -1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype)  # switch to fload if need + fp16 compatibility
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
        print("GPT2WithNPI: Total num layers == ", len(self.h))
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states.view(*output_shape),)

            outputs = block(hidden_states,
                            layer_past=layer_past,
                            attention_mask=attention_mask,
                            head_mask=head_mask[i])

            hidden_states, present = outputs[:2]
            if i in self.prediction_indices:
                # hidden_states = self.npi(hidden_states[0], target_classification, hidden_states[1:]) # NPI added functionality
                hidden_states[0] = self.npi(i)  # NPI added functionality
                print("GPT2WithNPI: layer num == ", i)
                print("GPT2WithNPI: hidden_states[0] shape == ", hidden_states[0].size)
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
            attention_output_shape = input_shape[:-1] + (-1,) + all_attentions[0].shape[-2:]
            all_attentions = tuple(t.view(*attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)
        return outputs  # last hidden state, presents, (all hidden_states), (attentions)


class GPT2LMWithNPI(GPT2LMHeadModel):
    r"""
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
        # self.prediction_indices = npi_config['prediction_indices'] # NPI added functionality
        # self.npi = NeuralProgramInterface(npi_config, 'gpt2') # NPI added functionality
        # self.transformer = GPT2WithNPI(config, self.npi, self.prediction_indices) # NPI added functionality
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # self.init_weights()
        # self.tie_weights()
        GPT2LMHeadModel.__init__(self, config)  # NPI added functionality

    def initialize_npi(self, npi_config):
        self.prediction_indices = npi_config['prediction_indices']  # NPI added functionality
        self.npi = NeuralProgramInterface(npi_config, 'gpt2-medium')  # NPI added functionality
        self.transformer = GPT2WithNPI.from_pretrained(
            'gpt2-medium')  # (config, self.npi, self.prediction_indices) # NPI added functionality
        self.transformer.initialize_npi(self.npi, self.prediction_indices)

    def forward(self, input_ids, past=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None, target_classification=None):
        """
        target_classification : nx1x1 target classification vector  # NPI added functionality
        """
        transformer_outputs = self.transformer(input_ids,
                                               past=past,
                                               attention_mask=attention_mask,
                                               token_type_ids=token_type_ids,
                                               position_ids=position_ids,
                                               head_mask=head_mask,
                                               target_classification=target_classification)  # NPI added functionality
        hidden_states = transformer_outputs[0]

        lm_logits = self.lm_head(hidden_states)

        outputs = (lm_logits,) + transformer_outputs[1:]
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)),
                            shift_labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), lm_logits, presents, (all hidden_states), (attentions)


class NeuralProgramInterface:  # NPI added functionality
    def __init__(self, npi_config, model_type):
        """
        npi_config : {'npi_model': npi_neural_weights, 
                      'flattened_shape':[n (int), m (int), batch_size (int)], 
                      'prediction_indices': list((int)), 
                      'max_seq_len': (int), 
                      } 

        USAGE DIRECTIONS / STEPS:
            1) __init__
            2) generate_text / forward()
            4) repeat from Step 2
        """
        self.curr_sequence_index = 0
        self.max_seq_len = npi_config['max_seq_len']
        self.num_layers_per_seq_elem = len(npi_config['prediction_indices'])

        self.sequential_indices = []
        for _ in range(self.max_seq_len):
            self.sequential_indices = self.sequential_indices + npi_config['prediction_indices']

        self.npi_model = npi_config['npi_model']
        self.n, self.m, self.batch_size = npi_config['flattened_shape']
        self.model_type = model_type

        self.cached_activs = [None for _ in range(len(self.sequential_indices))]
        self.cached = False

    def flatten_activations(self, activations):  # , target_label):
        # each hidden layer in the hugging face repo has shape (batch, seq_len, hidden_size)
        # for ALL models.
        # We assume activations is indexed by self.sequential_indices, appearing in model-layer order
        reshaped = [x.view(self.batch_size, self.max_seq_len, self.m) for x in activations]
        flattened = torch.cat(reshaped, dim=1)
        return flattened.view(self.batch_size, self.n, self.m, 1)

    def unflatten_activations(self, flattened):
        resaphed = flattened.view(self.batch_size, self.n, self.m, 1)
        # each hidden layer in the hugging face repo has shape (batch, seq_len, hidden_size)
        # for ALL models
        chunked = torch.chunk(resaphed, self.max_seq_len, dim=0)
        return [x.view(self.batch_size, self.max_seq_len, self.m) for x in chunked]

    def cache_activations(self, orig_activs, target_classification):
        self.cached = False
        self.cached_activs = [None for _ in range(len(self.sequential_indices))]
        # flatten the activations
        flattened_activations = self.flatten_activations(orig_activs)
        # call self.npi_model on flattened activations
        reimagined_activations = self.npi_model(flattened_activations, target_classification)
        # unflatten the npi_model's output and store the unflattened output as self.cached_activs
        self.cached_activs = self.unflatten_activations(reimagined_activations)
        self.cached = True
        pass

    def uncache_activations(self, orig_activs, target_classification):
        self.cached = False
        del self.cached_activs

    def top_k_top_p_filtering(self, logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (vocabulary size)
                top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
            From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
        """
        # Note: this was taken/modified from a decrimented huggingface implementation
        # print("Logits == ", logits)
        assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
        top_k = min(top_k, logits.size(-1))  # Safety check
        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value
        return logits

    def generate_text(self, in_text, lm_model, tokenizer,
                      length, num_samples=1, temperature=0, top_k=0,
                      top_p=0.0, is_xlnet=False, device='cuda'):
        # Note: this was taken/modified from a decrimented huggingface implementation
        tokenized = tokenizer._tokenize(in_text)
        embedded_text = [tokenizer._convert_token_to_id(tok) for tok in tokenized]
        context = torch.tensor(embedded_text, dtype=torch.long, device=device)
        context = context.unsqueeze(0).repeat(num_samples, 1)
        generated = context
        with torch.no_grad():
            for _ in trange(length):
                inputs = {'input_ids': generated}

                outputs = lm_model(
                    **inputs)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
                next_token_logits = outputs[0][0, -1, :] / temperature
                filtered_logits = self.top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                generated = torch.cat((generated[:, 1:], next_token.unsqueeze(0)),
                                      dim=1)  # Note: added [:,1:] to enforce sliding window for passage through npi
        out = generated[0, len(embedded_text):].tolist()
        decoded_toks = [tokenizer._convert_id_to_token(embedded) for embedded in out]
        text = tokenizer.convert_tokens_to_string(decoded_toks)
        return text

    def forward(self, layer_index):
        if layer_index == self.sequential_indices[self.curr_sequence_index] and self.cached:
            if self.curr_sequence_index != self.max_seq_len * self.num_layers_per_seq_elem - 1:
                self.curr_sequence_index += 1
                return self.cached_activs[self.curr_sequence_index - 1]
            else:
                self.cached = False
                self.curr_sequence_index = 0
                return self.cached_activs[-1]
        else:
            raise ValueError("NeuralProgramInterface :: requested layer activation not in cached activations")


def evaluate_npi_text_generation(npi_model_path, flattened_shape, prediction_indices,
                                 max_seq_len, model_type, data_path, save_path):
    """
    npi_model_path: path to trained_npi_model, 
    flattened_shape:[n (int), m (int), batch_size (int)], 
    prediction_indices: list((int representing which layers of a given language model you wish to control)), 
    max_seq_len: (int representing how many tokens to generate/to use for context)
    model_type: str representing one of ['gpt2-medium', 'xlnet-large-cased', 'transfo-xl-wt103']
    data_path: str representing the path to the evaluation data set, assumed to have the format:
        [[language_model_activations, 
            activations_classification, 
            target_classification, 
            language_model_type, 
            meta_data, 
            original_text, 
            predicted_text,
        ],  
        ...]
        With objects of the following types:
            language_model_activations : nxmx1 ndarray representing flattened activation sequences (required)
            activations_classification : 1xmx1 ndarray representing the sentiment/content classification of the original activations (optional - assumed None)
            target_classification : 1xmx1 ndarray representing the desired sentiment/content classification of generated activations (required)
            language_model_type : str naming the language model being controlled (optional - assumed None)
            meta_data : dict recording desired metadata (optional - assumed None)
            original_text : str representing the context passed into the pretrained model to generate language_model_activations
            predicted_text : None - THIS WILL BE OVER-WRITTEN in this method
    save_path: str representing the path to which the over-written data set ought to be saved - this directory will be created if it doesn't already exist
    """
    ORIG_ACTIV_INDEX = 0
    ORIG_LABEL_INDEX = 1
    TARG_LABEL_INDEX = 2
    LANG_MODEL_INDEX = 3
    META_DATA_INDEX = 4
    ORIG_TEXT_INDEX = 5
    PRED_TEXT_INDEX = 6

    npi_config = {'npi_model': torch.load(npi_model_path),
                  'flattened_shape': flattened_shape,
                  'prediction_indices': prediction_indices,
                  'max_seq_len': max_seq_len,
                  }

    npi = NeuralProgramInterface(npi_config, model_type)
    npi.npi_model.cuda()

    tokenizer = None
    host_lm = None
    if model_type == 'gpt2-medium':
        tokenizer = GPT2Tokenizer.from_pretrained(model_type)
        # host_lm = GPT2LMHeadModel.from_pretrained(model_type)
        host_lm = GPT2LMWithNPI.from_pretrained(model_type)
    # elif model_type == 'xlnet-large-cased':
    #     tokenizer = XLNetTokenizer.from_pretrained(model_type)
    #     host_lm = XLNetLMHeadModel.from_pretrained(model_type)
    # elif model_type == 'transfo-xl-wt103':
    #     tokenizer = TransfoXLTokenizer.from_pretrained(model_type)
    #     host_lm = TransfoXLLMHeadModel.from_pretrained(model_type)
    else:
        raise ValueError("evaluate_npi_text_generation :: model_type == {} not supported".format(model_type))

    host_lm = host_lm.cuda()
    device = torch.device("cuda")

    dataset = None
    with open(data_path, 'rb') as datafile:
        dataset = pkl.load(datafile)

    for i in range(len(dataset)):
        acts = torch.from_numpy(dataset[i][ORIG_ACTIV_INDEX]).cuda()
        targ = torch.from_numpy(dataset[i][TARG_LABEL_INDEX]).cuda()
        orig_text = dataset[i][ORIG_TEXT_INDEX]

        npi.cache_activations(acts, targ)

        controlled_text = npi.generate_text(orig_text, targ, host_lm, tokenizer, max_seq_len)
        dataset[i][PRED_TEXT_INDEX] = controlled_text

        del acts
        del targ
        npi.uncache_activations

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    with open(save_path, 'wb') as outfile:
        pkl.dump(dataset, outfile, protocol=pkl.HIGHEST_PROTOCOL)

    torch.cuda.empty_cache()
    pass


"""
tokenizer = GPT2Tokenizer.from_pretrained(model_type)
host_lm = GPT2LMHeadModel.from_pretrained(model_type)
tokenizer = XLNetTokenizer.from_pretrained(model_type)
host_lm = XLNetLMHeadModel.from_pretrained(model_type)
tokenizer = TransfoXLTokenizer.from_pretrained(model_type)
host_lm = TransfoXLLMHeadModel.from_pretrained(model_type)
"""
