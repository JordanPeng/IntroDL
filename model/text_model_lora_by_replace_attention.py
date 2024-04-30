from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers.cache_utils import Cache
import math


class LlamaAttentionWithLoRA(LlamaAttention):
    """Extended Multi-headed attention from LLaMA model with LoRA implementation."""

    def __init__(self, config, layer_idx=None, lora_dim=32, max_length=512):
        super().__init__(config, layer_idx)
        self.lora_dim = lora_dim
        self.lora_q_A = nn.Parameter(torch.randn(self.hidden_size, lora_dim))
        self.lora_q_B = nn.Parameter(torch.randn(lora_dim, self.num_heads * self.head_dim))
        self.lora_k_A = nn.Parameter(torch.randn(self.hidden_size, lora_dim))
        self.lora_k_B = nn.Parameter(torch.randn(lora_dim, self.num_key_value_heads * self.head_dim))
        self.lora_v_A = nn.Parameter(torch.randn(self.hidden_size, lora_dim))
        self.lora_v_B = nn.Parameter(torch.randn(lora_dim, self.num_key_value_heads * self.head_dim))

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs,
    ):
        bsz, q_len, _ = hidden_states.size()

        # LoRA Adjustments for Q, K, V
        lora_q_adjustment = self.lora_q_A @ self.lora_q_B
        lora_k_adjustment = self.lora_k_A @ self.lora_k_B
        lora_v_adjustment = self.lora_v_A @ self.lora_v_B

        # Apply LoRA adjustments
        query_states = self.q_proj(hidden_states) + hidden_states @ lora_q_adjustment
        key_states = self.k_proj(hidden_states) + hidden_states @ lora_k_adjustment
        value_states = self.v_proj(hidden_states) + hidden_states @ lora_v_adjustment

        # Following the original attention mechanism calculations
        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        # Further calculations as in the original LlamaAttention class
        if past_key_value is not None:
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights += attention_mask

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights, past_key_value


# Define the LoRA-enhanced model class
class LoraLLaMA(torch.nn.Module):
    def __init__(self, model_name, rank=4, device='cuda'):
        super().__init__()
        self.text_model = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto')
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        custom_tokens = ['[VST]', '[VET]']
        num_added_toks = self.tokenizer.add_tokens(custom_tokens)
        self.voice_start_token_id = self.tokenizer.convert_tokens_to_ids('[VST]')
        self.voice_end_token_id = self.tokenizer.convert_tokens_to_ids('[VET]')

        self.text_model.resize_token_embeddings(len(self.tokenizer))
        self.max_length = 512
        self.rank = rank
        self.device = device
        self.apply_lora()

    def get_tokenizer(self):
        return AutoTokenizer.from_pretrained(self.text_model.config.model_type)

    def apply_lora(self):
        new_attn_0 = LlamaAttentionWithLoRA(self.text_model.config, layer_idx=0, lora_dim=8, max_length=512)
        new_attn_0.load_state_dict(self.text_model.model.layers[0].self_attn.state_dict(), strict=False)

        # setattr(self.text_model.model.layers[0].self_attn, "self_attn",
        #         new_attn_0)  # this is error because it rather add a new attribute to the object

        new_attn_1 = LlamaAttentionWithLoRA(self.text_model.config, layer_idx=1, lora_dim=8, max_length=512)
        new_attn_1.load_state_dict(self.text_model.model.layers[1].self_attn.state_dict(), strict=False)
        # setattr(self.text_model.model.layers[1].self_attn, "self_attn",
        #         new_attn_1)

        self.text_model.model.layers[0].self_attn = new_attn_0
        self.text_model.model.layers[1].self_attn = new_attn_1

        self.lora_params = [param for name, param in self.text_model.named_parameters() if \
                            ('lora_q' in name or 'lora_k' in name or 'lora_v' in name)]

    def extract_text_embeddings(self, text_ids):
        inputs_embeds = self.text_model.get_input_embeddings()(text_ids)
        return inputs_embeds

    def forward(self, input_ids=None, attention_mask=None, inputs_embeds=None, labels=None):
        if inputs_embeds is None:
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids)
        return self.text_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
