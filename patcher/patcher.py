import torch
import torch.nn.functional as F
from models.entropy_model import EntropyModel, EntropyModelArgs

def load_entropy_model(path=None):
    pass

#This is the official implementation
def entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits,dim=-1)
    probs = torch.exp(log_probs)
    p_log_p = probs * log_probs

    #why negative, the formula in the paper is positive
    #also, why su
    entropy = -p_log_p.sum(dim=-1) 
    return entropy

class Patcher():
    def __init__(self,patch_size: int, entropy_threshold: int):
        self.args = EntropyModelArgs
        self.entropy_model = EntropyModel(self.args)
        self.patch_size = patch_size
        self.entropy_theshold = entropy_threshold

    def calculate_entropy(self,tokens: torch.Tensor):
        bs, slen, dim = tokens.size()
        logits = self.entropy_model(tokens)
        entropies = entropy(logits)
        return entropies

    ##TODO: implement create mask from entropy to feed into local encoder, maybe implement a little different from their code