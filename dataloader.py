from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from models.constant import SpecialTokens
from models.tokenizer import Tokenizer
from dataset import CustomDataset

def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=SpecialTokens.PAD_ID)
    labels = pad_sequence(labels, batch_first=True, padding_value=SpecialTokens.PAD_ID)

    return {"input_ids": input_ids, "labels": labels}

class CustomDataloader:
    def __init__(self, tokenizer: Tokenizer, dataset: str, batch_size: int, max_length: int = 512, split: str = "train", shuffle: bool = False):
        dataset = CustomDataset(tokenizer, dataset, max_length, split)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)

    def __iter__(self):
        return iter(self.dataloader)

    def __len__(self):
        return len(self.dataloader)

    def __getitem__(self, index):
        return self.dataloader[index]
