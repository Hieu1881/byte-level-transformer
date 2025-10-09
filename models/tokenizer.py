import torch
from .constant import SpecialTokens

class Tokenizer():
    def __init__(self):
        self.bos_id = SpecialTokens.BOS_ID
        self.eos_id = SpecialTokens.EOS_ID
        self.offset = SpecialTokens.SPECIAL_TOKENS_OFFSET
        self.pad_id = SpecialTokens.PAD_ID
        
    def encode(self,text):
        tokens = bytes(text, encoding='utf-8', errors='ignore')
        tokens = [int(token)+self.offset for token in tokens]
        tokens.insert(0,self.bos_id)
        tokens.append(self.eos_id)
        return tokens
    
    def decode(self,tokens):
        for token in tokens:
            return bytes(
                [token - self.offset for token in tokens if token - self.offset >= 0] #filter out special tokens
            ).decode(encoding='utf-8',errors='ignore')

