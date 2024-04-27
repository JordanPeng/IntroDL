import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import Adam
from torch.nn.parameter import Parameter
from Project.model.text_model import LoraLLaMA
from torch.optim import Adam

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "JackFram/llama-68m"  # Example model
model = LoraLLaMA(model_name, device=device)
# print(model.lora_parameters)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
# Hyperparameters and optimizer
learning_rate = 2e-5


# Dummy dataset and dataloader


def prepare_input_output(examples):
    # Tokenize the text
    inputs = tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512, return_tensors="pt")

    # Creating input_ids for the first half of the sequences
    half_seq_length = inputs.input_ids.size(1) // 2
    input_ids_half = inputs.input_ids[:, :half_seq_length]

    # Labels are the full input_ids
    labels_full = inputs.input_ids.clone()

    # Mask the first half of the labels so the loss is calculated only on the second half
    labels_full[:, :half_seq_length] = -100

    # Create attention masks that match the first half of each sequence
    # Attention mask should be 1 for all tokens in the first half and 0 for the rest
    attention_masks = torch.zeros_like(inputs.input_ids)
    attention_masks[:, :half_seq_length] = 1

    return {
        "input_ids": input_ids_half,
        "labels": labels_full,
        "attention_mask": attention_masks
    }


def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch]).to(device)
    labels = torch.stack([item['labels'] for item in batch]).to(device)
    attention_masks = torch.stack([item['attention_mask'] for item in batch]).to(device)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_masks}


dataset = load_dataset("wikipedia", "20220301.en", split='train[:1000]')  # smaller slice for example
processed_dataset = dataset.map(prepare_input_output, batched=True)
processed_dataset.set_format(type='torch', columns=['input_ids', 'labels', 'attention_mask'])
dataloader = DataLoader(processed_dataset, batch_size=4, collate_fn=collate_fn)

optimizer = Adam(model.lora_parameters, lr=2e-5)

# Manual training loop
model.train()
num_epochs = 10
for epoch in range(num_epochs):
    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()
        outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'])
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")
