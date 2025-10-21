import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseTransformer, BaseTransformerArgs

class EntropyModelArgs(BaseTransformerArgs):
    vocab_size: int = 260
    n_layers: int = 3
    n_heads: int = 8
    max_seqlen: int = 256  #maybe longer seq_len?


class EntropyModel(BaseTransformer):
    def __init__(self, args:EntropyModelArgs):
        super().__init__(args)
        self.token_emb = nn.Embedding(args.vocab_size,args.dim)
        self.language_head = nn.Linear(args.dim, args.vocab_size) 
        
    def forward(self, x: torch.Tensor):
        tok_emb = self.token_emb(x) #bs, seq_len, dim
        out = super().forward(tok_emb,mask=True)
        return self.language_head(out) #bs,seq_len,vocab_size
    

