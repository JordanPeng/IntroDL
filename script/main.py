import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
import argparse

from script.prepare_data import prepare_dataloader
from model.joint_model import JointModel


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    joint_model = JointModel(args.audio_model, args.text_model,  device=device, rank=args.lora_rank)
    joint_model.to(device)
    audio_model = joint_model.audio_model
    text_model = joint_model.text_model

    data_loader = prepare_dataloader(audio_model.audio_processor, text_model.tokenizer, args.batch_size,
                                     split="train.100")

    optimizer = torch.optim.Adam([
        # {'params': joint_model.audio_model.parameters(), 'lr': 1e-4},
        {'params': joint_model.text_model.lora_params, 'lr': 2e-3},
        {'params': joint_model.audio_projection.parameters(), 'lr': 1e-4}
    ])
    joint_model.train()
    num_epochs = 20
    best_loss = 999
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in data_loader:
            optimizer.zero_grad()

            batch['audio_input_values'] = batch['audio_input_values'].to(device)
            batch['audio_labels'] = batch['audio_labels'].to(device)
            batch['audio_targets_length'] = batch['audio_targets_length'].to(device)
            batch['text_ids'] = batch['text_ids'].to(device)

            audio_embeddings = joint_model.forward_audio(batch['audio_input_values'])

            # add special token embeddings
            voice_start_embed = text_model.extract_text_embeddings(
                torch.tensor(text_model.voice_start_token_id).to(device))
            voice_end_embed = text_model.extract_text_embeddings(
                torch.tensor(text_model.voice_end_token_id).to(device))

            start_token_expanded = voice_start_embed.unsqueeze(0).unsqueeze(0).expand(args.batch_size, -1, -1)
            end_token_expanded = voice_end_embed.unsqueeze(0).unsqueeze(0).expand(args.batch_size, -1, -1)

            audio_embeddings_with_token = torch.cat([start_token_expanded, audio_embeddings, end_token_expanded], dim=1)
            audio_embedding_mask = torch.ones(audio_embeddings_with_token.shape[:-1]).to(device)

            # create tensor for text model to forward
            text_ids = batch['text_ids']
            # extend the text_id length to make joint embedding length 512
            text_ids = torch.cat(
                [text_ids, torch.full((args.batch_size, 512 - text_ids.shape[1] - audio_embeddings_with_token.shape[1]),
                                      text_model.tokenizer.pad_token_id).to(device)], dim=1)

            inputs_embeds = text_model.extract_text_embeddings(text_ids)
            joint_embedding = torch.cat([audio_embeddings_with_token, inputs_embeds], dim=1)

            text_start_token_mask = torch.ones((args.batch_size, 1)).to(device)
            # minus 1 because the first token is the text start token
            text_remain_length_mask = torch.zeros((args.batch_size, text_ids.shape[1] - 1)).to(device)
            joint_mask = torch.cat([audio_embedding_mask, text_start_token_mask, text_remain_length_mask], dim=1)

            audio_labels = torch.full((args.batch_size, audio_embeddings_with_token.shape[1]), -100).to(device)
            joint_labels = torch.cat([audio_labels, text_ids], dim=1)
            # change the padding token in joint_labels to -100
            joint_labels[joint_labels == text_model.tokenizer.pad_token_id] = -100

            outputs = joint_model.forward(joint_embedding, joint_mask, joint_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if total_loss / len(data_loader) < best_loss:
            print("saving model, previous loss: {:.4f}, current loss: {:.4f}".format(best_loss, total_loss))
            best_loss = total_loss
            torch.save(joint_model.state_dict(),
                       "/pycharm_projects/IntroDL_Project/IntroDL/save/epoch_{}_loss_{:.4f}.pt".format(epoch,
                                                                                                       total_loss))
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(data_loader)}")
        # save the best model


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


def test(best_joint_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    joint_model = JointModel(args.audio_model, args.text_model,  device=device, rank=args.lora_rank)
    joint_model.to(device)
    joint_model.load_state_dict(torch.load(best_joint_model))
    joint_model.eval()

    data_loader = prepare_dataloader(joint_model.audio_model.audio_processor, joint_model.text_model.tokenizer,
                                     args.batch_size, split="test")
    wer=0
    for batch in data_loader:
        batch['audio_input_values'] = batch['audio_input_values'].to(device)
        batch['audio_labels'] = batch['audio_labels'].to(device)
        batch['audio_targets_length'] = batch['audio_targets_length'].to(device)
        batch['text_ids'] = batch['text_ids'].to(device)

        audio_embeddings = joint_model.forward_audio(batch['audio_input_values'])

        # add special token embeddings
        voice_start_embed = joint_model.text_model.extract_text_embeddings(
            torch.tensor(joint_model.text_model.voice_start_token_id).to(device))
        voice_end_embed = joint_model.text_model.extract_text_embeddings(
            torch.tensor(joint_model.text_model.voice_end_token_id).to(device))

        start_token_expanded = voice_start_embed.unsqueeze(0).unsqueeze(0).expand(args.batch_size, -1, -1)
        end_token_expanded = voice_end_embed.unsqueeze(0).unsqueeze(0).expand(args.batch_size, -1, -1)

        audio_embeddings_with_token = torch.cat([start_token_expanded, audio_embeddings, end_token_expanded], dim=1)
        audio_embedding_mask = torch.ones(audio_embeddings_with_token.shape[:-1]).to(device)

        # create tensor for text model to forward
        text_ids = batch['text_ids']
        # extend the text_id length to make joint embedding length 512
        text_ids = torch.cat(
            [text_ids, torch.full((args.batch_size, 512 - text_ids.shape[1] - audio_embeddings_with_token.shape[1]),
                                  joint_model.text_model.tokenizer.pad_token_id).to(device)], dim=1)

        inputs_embeds = joint_model.text_model.extract_text_embeddings(text_ids)
        joint_embedding = torch.cat([audio_embeddings_with_token, inputs_embeds], dim=1)

        text_start_token_mask = torch.ones((args.batch_size, 1)).to(device)
        # minus 1 because the first token
        text_remain_length_mask = torch.zeros((args.batch_size, text_ids.shape[1] - 1)).to(device)
        joint_mask = torch.cat([audio_embedding_mask, text_start_token_mask, text_remain_length_mask], dim=1)

        audio_labels = torch.full((args.batch_size, audio_embeddings_with_token.shape[1]), -100).to(device)
        joint_labels = torch.cat([audio_labels, text_ids], dim=1)
        # change the padding token in joint_labels to -100
        joint_labels[joint_labels == joint_model.text_model.tokenizer.pad_token_id] = -100

        outputs = joint_model.forward(joint_embedding, joint_mask, joint_labels)
        output_ids = outputs.logits[:, audio_labels.shape[1]:, :].argmax(dim=-1)
        # output_text = joint_model.text_model.tokenizer.decode(output_ids[0])
        # calculate word error rate
        for i in range(args.batch_size):
            reference_text = joint_model.text_model.tokenizer.decode(text_ids[i])
            hypothesis_text = joint_model.text_model.tokenizer.decode(output_ids[i])
            wer_item = calculate_wer(reference_text, hypothesis_text)
            wer += wer_item
    wer = wer / len(data_loader)
    print(f"Word Error Rate: {wer}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--audio_model", type=str, default="facebook/wav2vec2-base-960h")
    args.add_argument("--text_model", type=str, default="JackFram/llama-68m")
    args.add_argument("--batch_size", type=int, default=2)
    args.add_argument("--lora_rank", type=int, default=8)

    args = args.parse_args()
    # main(args)

    test("/pycharm_projects/IntroDL_Project/IntroDL/save/epoch_19_loss_10104.2066.pt")
