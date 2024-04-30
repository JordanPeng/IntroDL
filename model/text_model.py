import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset


# Define the LoRA-enhanced model class
class LoraLLaMA(torch.nn.Module):
    def __init__(self, model_name, rank=4, device='cuda'):
        super().__init__()
        self.text_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        custom_tokens = ['[VST]', '[VET]']
        num_added_toks = self.tokenizer.add_tokens(custom_tokens)
        self.voice_start_token_id = self.tokenizer.convert_tokens_to_ids('[VST]')
        self.voice_end_token_id = self.tokenizer.convert_tokens_to_ids('[VET]')

        self.text_model.resize_token_embeddings(len(self.tokenizer))

        self.rank = rank
        # self.lora_parameters = torch.nn.ParameterList()  # List to store LoRA parameters
        self.device = device
        # self.lora_name_map = {}
        self.apply_lora()
        # self.add_all_lora_params_to_lora_parameters()

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.text_model.config.model_type)

    def apply_lora(self):
        for name, param in self.text_model.named_parameters():
            if 'self_attn.q_proj' in name:
                self._apply_lora_to_param(param, name, 'query')
            elif 'self_attn.k_proj' in name:
                self._apply_lora_to_param(param, name, 'key')
            elif 'self_attn.v_proj' in name:
                self._apply_lora_to_param(param, name, 'value')
            elif 'self_attn.o_proj' in name:
                self._apply_lora_to_param(param, name, 'output')
        # Freeze all original parameters of the model
        for param in self.text_model.parameters():
            param.requires_grad = False

    def _apply_lora_to_param(self, param, name, param_type):
        A = torch.randn((param.size(0), self.rank)) * 0.01
        B = torch.randn((self.rank, param.size(1))) * 0.01
        A = nn.Parameter(A)
        B = nn.Parameter(B)
        # self.lora_parameters.append(A)
        # self.lora_parameters.append(B)
        name = name.replace('.weight', '')
        name = name.replace('.', '_')

        setattr(self, name + f'_lora_A', A)
        setattr(self, name + f'_lora_B', B)

        # self.lora_name_map[name] = (A, B)

    # def add_all_lora_params_to_lora_parameters(self):
    #     for name in vars(self).keys():
    #         if 'lora' in name:
    #             self.lora_parameters.append(getattr(self, name))

    def extract_text_embeddings(self, text_ids):
        inputs_embeds = self.text_model.get_input_embeddings()(text_ids)
        return inputs_embeds


    # this is wrong because if we directly change the weight of a object, it it like detaching the weight from the computing graph,
    # the chain will be broken by this way. so the weight will not be updated during the training
    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, labels=None):
        if inputs_embeds is None:
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids)
        for name, module in self.text_model.named_modules():
            if isinstance(module, torch.nn.Linear) and 'self_attn' in name and '_proj' in name:
                # A, B = self.lora_name_map[name]
                no_dot_name = name.replace('.', '_')
                A = getattr(self, no_dot_name + '_lora_A')
                B = getattr(self, no_dot_name + '_lora_B')

                lora_adjustment = A @ B
                original_weight = module.weight.data
                module.weight = nn.Parameter(original_weight + lora_adjustment)
        return self.text_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
