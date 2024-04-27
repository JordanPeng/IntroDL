import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader
from datasets import load_dataset


# Define the LoRA-enhanced model class
class LoraLLaMA(torch.nn.Module):
    def __init__(self, model_name, rank=4, device='cuda'):
        super().__init__()
        self.text_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.rank = rank
        self.lora_parameters = torch.nn.ParameterList()  # List to store LoRA parameters
        self.device = device
        self.lora_name_map = {}
        self.apply_lora()


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
        A = Parameter(torch.Tensor(param.size(0), self.rank).normal_(0, 0.02)).to(self.device)
        B = Parameter(torch.Tensor(self.rank, param.size(1)).normal_(0, 0.02)).to(self.device)
        self.lora_parameters.append(A)
        self.lora_parameters.append(B)
        name = name.replace('.weight', '')
        self.lora_name_map[name] = (A, B)

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, labels=None):
        if inputs_embeds is None:
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids)
        for name, module in self.text_model.named_modules():
            if isinstance(module, torch.nn.Linear) and 'self_attn' in name and '_proj' in name:
                A, B = self.lora_name_map[name]
                lora_adjustment = A @ B
                original_weight = module.weight.data
                module.weight = Parameter(original_weight + lora_adjustment)
        return self.text_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
