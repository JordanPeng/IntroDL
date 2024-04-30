from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
class AudioEncoder(torch.nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.audio_processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.audio_model = Wav2Vec2ForCTC.from_pretrained(model_name)

    def forward(self, audio_input_values):
        output = self.audio_model(audio_input_values, output_hidden_states=True)
        return output
