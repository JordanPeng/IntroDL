from model.audio_model import AudioEncoder
# from model.text_model import LoraLLaMA
from model.text_model_lora_by_replace_attention import LoraLLaMA
import torch


class JointModel(torch.nn.Module):
    def __init__(self, audio_model_name, text_model_name, rank=8, device='cuda', use_peft=False):
        super().__init__()
        self.audio_model = AudioEncoder(audio_model_name)
        self.text_model = LoraLLaMA(text_model_name, rank=rank, device=device)
        self.audio_projection = torch.nn.Linear(768, 768)

    def forward(self, joint_embedding, joint_mask, joint_labels):
        outputs = self.text_model(inputs_embeds=joint_embedding, attention_mask=joint_mask, labels=joint_labels)
        return outputs

    def forward_audio(self, audio_input_values):
        output = self.audio_model(audio_input_values)
        last_hidden_state = output.hidden_states[-1]
        return self.audio_projection(last_hidden_state)
