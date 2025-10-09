from models.entropy_model import EntropyModel, EntropyModelArgs
import torch
import torch.nn as nn
from dataset import CustomDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from models.tokenizer import Tokenizer
from models.constant import SpecialTokens
from tqdm import tqdm
import os
import time

epochs = 10
batch_size = 128
vocab_size = 260
early_stopping = 10

def collate_fn(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=SpecialTokens.PAD_ID)
    labels = pad_sequence(labels, batch_first=True, padding_value=SpecialTokens.PAD_ID)

    return {"input_ids": input_ids, "labels": labels}

def train():
    args = EntropyModelArgs()
    model = EntropyModel(args)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    tokenizer = Tokenizer()
    train_dataset = CustomDataset(tokenizer, max_length=args.max_seqlen, split="train")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    val_dataset = CustomDataset(tokenizer, max_length=args.max_seqlen, split="validation")
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model_params = sum(p.numel() for p in model.parameters())
    print(f"Entropy model params: {model_params / 1e6:.4f}M")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Training on device:", device)
    model = model.to(device)

    best_val_loss = float("inf")
    epochs_not_improved = 0

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        total_train_loss = 0.0

        train_loop = tqdm(train_dataloader, desc=f"[Epoch {epoch}] Training", leave=False)
        for batch in train_loop:
            optimizer.zero_grad()
            x, y = batch["input_ids"].to(device), batch["labels"].to(device)
            logits = model(x)
            logits = logits.view(-1, vocab_size)
            labels = y.view(-1)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_dataloader)

        model.eval()
        total_val_loss = 0.0
        val_loop = tqdm(val_dataloader, desc=f"[Epoch {epoch}] Validation", leave=False)
        with torch.no_grad():
            for batch in val_loop:
                x, y = batch["input_ids"].to(device), batch["labels"].to(device)
                logits = model(x)
                logits = logits.view(-1, vocab_size)
                labels = y.view(-1)
                loss = criterion(logits, labels)
                total_val_loss += loss.item()
                val_loop.set_postfix(loss=loss.item())

        avg_val_loss = total_val_loss / len(val_dataloader)
        time_elapsed = time.time() - start_time

        print(f"Epoch {epoch} - train_loss: {avg_train_loss:.4f} - val_loss: {avg_val_loss:.4f} - time: {time_elapsed:.2f}s")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_not_improved = 0
        else:
            epochs_not_improved += 1

        if epochs_not_improved == early_stopping:
            print(f"Early stopping at epoch {epoch}")
            break

    os.makedirs("pretrained", exist_ok=True)
    torch.save(model.state_dict(), "pretrained/entropy_model.pth")

if __name__ == "__main__":
    train()
