import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2Tokenizer

import torchaudio
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
import random
# Load the pre-trained Wav2Vec 2.0 model and its processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")

model.to("cuda")

# Load the LibriSpeech dataset
dataset = load_dataset("librispeech_asr", "clean", split="test")


# Preprocess the audio files to match the input expected by Wav2Vec2
def preprocess(batch):
    audio = batch["audio"]
    inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt", padding=True)
    # Extract non-padded lengths
    input_lengths = [len(input_ids) for input_ids in inputs.input_values]
    batch["input_values"] = inputs.input_values.squeeze()
    batch["input_lengths"] = torch.tensor(input_lengths)
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"], return_tensors="pt").input_ids.squeeze()
    return batch


def collate_fn(batch):
    input_values = [item['input_values'] for item in batch]
    labels = [item['labels'] for item in batch]

    input_values_padded = pad_sequence(input_values, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
    labels_padded = pad_sequence(labels, batch_first=True, padding_value=processor.tokenizer.pad_token_id)
    targets_length = torch.full((labels_padded.data.shape[0],), labels_padded.data.shape[1], dtype=torch.long)

    return {
        "input_values": input_values_padded,
        "labels": labels_padded,
        "targets_length": targets_length
    }


dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

dataset.set_format(type="torch", columns=["input_values", "input_lengths", "labels"])
dataset = dataset.filter(lambda example: example["input_values"].shape[0] <= 128000)
# DataLoader
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

# CTC Loss
ctc_loss = nn.CTCLoss(blank=processor.tokenizer.pad_token_id).to("cuda")

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=2e-4)

# Training configuration
num_epochs = 0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()

        input_values = batch["input_values"].to("cuda")
        input_lengths = batch["targets_length"].to("cuda")
        targets = batch["labels"].to("cuda")

        # Forward pass
        output = model(input_values, output_hidden_states=True)
        logits = output.logits

        # Compute the lengths of the logits
        logits_lengths = torch.full((logits.shape[0],), logits.shape[1], dtype=torch.long).to("cuda")

        prediction = torch.argmax(logits, dim=-1)
        # transcription = tokenizer.batch_decode(prediction)[0]
        # print(transcription)
        # Calculate CTC Loss
        log_logits = logits.log_softmax(2).permute(1, 0, 2)

        loss = ctc_loss(log_logits, targets, logits_lengths, input_lengths)

        # Backward pass
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Average Loss: {total_loss / len(dataset)}")

# Note: Add inference logic here if needed
def calculate_wer(reference, hypothesis):
    ref_words = reference.split()
    hyp_words = hypothesis.split()
    # Counting the number of substitutions, deletions, and insertions
    substitutions = sum(1 for ref, hyp in zip(ref_words, hyp_words) if ref != hyp)
    deletions = len(ref_words) - len(hyp_words)
    insertions = len(hyp_words) - len(ref_words)
    # Total number of words in the reference text
    total_words = len(ref_words)
    # Calculating the Word Error Rate (WER)
    wer = (substitutions + deletions + insertions) / total_words
    return wer

dataset = load_dataset("librispeech_asr", "clean", split="test")
wer = 0
for batch in dataset:
    audio = batch["audio"]
    inputs = processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt", padding=True)
    with torch.no_grad():
        logits = model(inputs.input_values.to("cuda")).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    wer += calculate_wer(batch["text"], transcription)
print(f"WER: {wer / len(dataset)}")