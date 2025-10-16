import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class CustomDataset(Dataset):
    def __init__(self, tokenizer, dataset:str, max_length=512, split="train"):
        """
        Dataset for next-token prediction.

        Args:
            texts (List[str]): List of raw text strings.
            tokenizer (BltTokenizer): Your tokenizer object.
            max_length (int): Max sequence length (including BOS and EOS if added).
        """
        ds = load_dataset()
        texts = ds[split]['text']
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        for text in texts:
            tokens = tokenizer.encode(text)
            if len(tokens) > max_length:
                # Truncate if needed
                tokens = tokens[:max_length]
            self.samples.append(tokens)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = self.samples[idx]
        input_ids = tokens[:-1]

        labels = tokens[1:]

        # # Pad if needed
        # input_ids = self.pad_sequence(input_ids)
        # labels = self.pad_sequence(labels)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
    

    def __len__(self):
        return len(self.samples)

    # def pad_sequence(self, seq):
    #     pad_id = self.tokenizer.pad_id if self.tokenizer.pad_id is not None else 0
    #     padded = seq + [pad_id] * (self.max_length - len(seq))
    #     return padded[:self.max_length]
