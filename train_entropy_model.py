from models.entropy_model import EntropyModel, EntropyModelArgs
import torch
import torch.nn.functional as F
import torch.nn as nn
from dataset import CustomDataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from models.tokenizer import Tokenizer
from models.constant import SpecialTokens

epochs = 10
batch_size = 4
vocab_size = 260

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
    dataset = CustomDataset(tokenizer,max_length=args.max_seqlen,split="train")
    dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=True, collate_fn= collate_fn)
    optimizer = torch.optim.AdamW(model.parameters(),lr=1e-3)
    model_params = sum(p.numel() for p in model.parameters())
    print(f"Entropy model params: {model_params/1e6:.4f}M")


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    for epoch in range(1,epochs+1):
        model.train()
        total_loss = 0.0
        for batch in dataloader:
            optimizer.zero_grad()
            x, y = batch["input_ids"].to(device), batch["labels"].to(device)
            logits = model(x)
            logits = logits.view(-1,vocab_size)
            labels = y.view(-1)
            loss = criterion(logits,labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch} - loss: {avg_loss:.4f}")
    
    torch.save(model.state_dict(),"pretrained/entropy_model.pth")

if __name__ == "__main__":
    train()        


            

        



