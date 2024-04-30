import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


def prepare_dataloader(audio_processor, text_tokenizer, batch_size=2, split="train"):
    dataset = load_dataset("librispeech_asr", "clean", split=split)

    def preprocess(batch):
        audio = batch["audio"]
        inputs = audio_processor(audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt",
                                 padding=True)
        # Extract non-padded lengths
        input_lengths = [len(input_ids) for input_ids in inputs.input_values]
        batch["audio_input_values"] = inputs.input_values.squeeze()
        batch["audio_input_lengths"] = torch.tensor(input_lengths)
        with audio_processor.as_target_processor():
            batch["audio_labels"] = audio_processor(batch["text"], return_tensors="pt").input_ids.squeeze()
        batch['text_ids'] = text_tokenizer(batch['text'].lower(), return_tensors="pt").input_ids.squeeze()
        batch['text_lengths'] = len(batch['text_ids'])
        return batch

    def collate_fn(batch):
        input_values = [item['audio_input_values'] for item in batch]
        labels = [item['audio_labels'] for item in batch]

        input_values_padded = pad_sequence(input_values, batch_first=True,
                                           padding_value=audio_processor.tokenizer.pad_token_id)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=audio_processor.tokenizer.pad_token_id)
        targets_length = torch.full((labels_padded.data.shape[0],), labels_padded.data.shape[1], dtype=torch.long)
        text_padded = pad_sequence([item['text_ids'] for item in batch], batch_first=True,
                                   padding_value=text_tokenizer.pad_token_id)

        text_lengths = torch.tensor([item['text_lengths'] for item in batch])

        return {
            "audio_input_values": input_values_padded,
            "audio_labels": labels_padded,
            "audio_targets_length": targets_length,
            "text_ids": text_padded,
            "text_lengths": text_lengths
        }

    dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
    dataset.set_format(type="torch", columns=["audio_input_values", "audio_input_lengths", "audio_labels", "text_ids",
                                              "text_lengths"])
    dataset = dataset.filter(lambda example: example["audio_input_values"].shape[0] <= 128000)

    # DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    return loader
