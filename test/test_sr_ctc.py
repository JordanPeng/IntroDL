import torch
from torch import nn, optim
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torchaudio
from datasets import load_dataset

# Load the pre-trained Wav2Vec 2.0 model and its processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
model.to("cuda")

# Load the LibriSpeech dataset
dataset = load_dataset("librispeech_asr", "clean", split="train.100")

# Preprocess the audio files to match the input expected by Wav2Vec2
def preprocess(batch):
    audio = batch["audio"]
    inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt", padding=True)

    # Extract non-padded lengths
    input_lengths = [len(input_ids) for input_ids in inputs.input_values]

    batch["input_values"] = inputs.input_values.squeeze()
    batch["input_lengths"] = torch.tensor(input_lengths)
    batch["labels"] = processor(batch["text"], return_tensors="pt").input_ids.squeeze()
    return batch

dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# CTC Loss
ctc_loss = nn.CTCLoss(blank=processor.tokenizer.pad_token_id).to("cuda")

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=5e-5)

# Training configuration
num_epochs = 3

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in dataset:
        optimizer.zero_grad()

        input_values = batch["input_values"].to("cuda")
        input_lengths = batch["input_lengths"].to("cuda")
        targets = batch["labels"].to("cuda")

        # Forward pass
        logits = model(input_values).logits

        # Compute the lengths of the logits
        logits_lengths = torch.full((logits.shape[0],), logits.shape[1], dtype=torch.long).to("cuda")

        # Calculate CTC Loss
        loss = ctc_loss(logits.log_softmax(2), targets, logits_lengths, input_lengths)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Average Loss: {total_loss / len(dataset)}")

# Note: Add inference logic here if needed
