import torch
from datasets import load_dataset
from torch.utils.data import DataLoader

def main():
    dataset = load_dataset("facebook/multilingual_librispeech", "german", split='train.1h', streaming=True)
    dataloader = DataLoader(dataset, batch_size=32)
    



if __name__ == "__main__":
    main()