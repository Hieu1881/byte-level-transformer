import torch
import torch.nn as nn
import torch.nn.functional as F
from  pydantic import BaseModel, ConfigDict
from .constant import SpecialTokens
from typing import Optional, Tuple, Union
from torch.nn import RMSNorm

from enum import Enum

class InitStdFactor(str, Enum):
    DISABLED = "disabled"  # Init std is divided by 1.0
    GLOBAL_DEPTH = "global_depth"  # Init std is divided by sqrt(2*n_layers)
    CURRENT_DEPTH = "current_depth"  # Init std is divided by sqrt(2*depth)
    DIM_RATIO = "dim_ratio"  # Init std is divided by model_dim/4096


class BaseTransformerArgs(BaseModel):
    model_config = ConfigDict(extra='forbid')
    dim: int = 512
    n_layers: int = 8
    head_dim: int | None = None
    n_heads: int | None = None
    n_kv_heads: int | None = None

    ffn_dim_multiplier: float | None = None

    multiple_of: int = 256

    norm_eps: float = 1e-5

    rope_theta: float = 10000.0
    rope_use_fp32_in_outer_product: bool = False

    init_base_std: float | None = None
    init_std_factor: InitStdFactor = InitStdFactor.DISABLED

    max_seqlen: int = 1024

    attn_impl: str | None = "sdpa"
    attn_bias_type: str | None = None
    # Special token config
    eos_id: int | None = SpecialTokens.EOS_ID
    bos_id: int | None = SpecialTokens.BOS_ID


def apply_rotary_emb(x,cos,sin):
    assert x.ndim == 4, "shape must be 4 (bs,nheads,seq_len,dim)"
    d = x.shape[-1] // 2
    x1, x2 = x[:,:,:,:d], x[:,:,:,d:] #[bs,nheads,deq_len,d]
    y1 = x1 * cos + x2 * sin 
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1,y2],dim=-1)
    out = out.to(x.dtype)
    return out

def repeat_kv(x, n_rep):
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    if n_rep == 1:
        return x
    bs, n_kv_heads, slen, head_dim = x.shape
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

def norm(x):
    return F.rms_norm(x,(x.size(-1),)) #This doesnt have learnable params??


class Attention(nn.Module):
    def __init__(self, dim, n_head,n_kv_head):
        super().__init__()
        self.dim = dim
        self.n_head = n_head
        self.n_kv_head = n_kv_head
        # self.layer_idx = layer_idx     #No kv cache implemented yet
        assert self.dim // n_head == 0
        self.head_dim = dim // n_head

        self.c_q = nn.Linear(self.dim,self.n_head * self.head_dim, bias=False)
        self.c_k = nn.Linear(self.dim,self.n_kv_head * self.head_dim, bias=False)
        self.c_v = nn.Linear(self.dim,self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = nn.Linear(self.dim, self.dim, bias= False)

    def forward(self,x,cos_sin,mask=None, kv_cache=None):
        bs, slen, dim = x.shape

        q = self.c_q(x).view(bs,slen,self.n_head,self.head_dim)
        k = self.c_k(x).view(bs,slen,self.n_kv_head,self.head_dim)
        v = self.c_v(x).view(bs,slen,self.n_kv_head,self.head_dim)

        cos, sin = cos_sin
        q = apply_rotary_emb(q,cos,sin)
        k = apply_rotary_emb(k,cos,sin)
        q, k = norm(q), norm(k)
        transpose = lambda x: x.transpose(1,2)
        q, k, v = map(transpose,(q,k,v))

        q_slen = q.size(2)
        kv_slen = k.size(2)

        nrep = self.n_head // self.n_kv_head
        k,v = repeat_kv(k,nrep), repeat_kv(v,nrep)
        att = F.scaled_dot_product_attention(q,k,v,is_causal=True)

        out = self.c_proj(att).transpose(1,2).contiguous().view(bs,slen,-1)
        return out


class FeedForward(nn.Module):
    def __init__(self,dim:int,hidden_dim:int, multiple_of: int):
        super().__init__()
        hidden_dim = int(2*hidden_dim / 3)
        hidden_dim = multiple_of* ((hidden_dim+multiple_of-1)//multiple_of)

        self.hidden_dim = hidden_dim

        self.w1 = nn.Linear(dim,hidden_dim,bias=False)
        self.w2 = nn.Linear(hidden_dim,dim,bias=False)
        self.w3 = nn.Linear(dim,hidden_dim,bias=False)

    def forward(self, x: torch.Tensor):
        x1 = self.w1(x)
        x3 = self.w3(x)

        out = self.w2(F.silu(x1)*x3)
        return out

    def reset_parameters(self, init_std=None, factor=1.0):
        in_init_std = init_std or (self.dim ** (-0.5)) / factor
        out_init_std = init_std or (self.hidden_dim ** (-0.5)) / factor

        nn.init.trunc_normal_(
            self.w1.weight,
            mean=0.0,
            std=in_init_std,
            a=-3 * in_init_std,
            b=3 * in_init_std,
        )
        nn.init.trunc_normal_(
            self.w2.weight,
            mean=0.0,
            std=out_init_std,
            a=-3 * out_init_std,
            b=3 * out_init_std,
        )
        nn.init.trunc_normal_(
            self.w3.weight,
            mean=0.0,
            std=in_init_std,
            a=-3 * in_init_std,
            b=3 * in_init_std,
        )

class Block(nn.Module):
    def __init__(self,args:BaseTransformerArgs):
        super().__init__()
        self.n_heads = args.n_heads

        assert args.dim % args.n_heads == 0
        self.head_dim = args.dim // args.n_heads

        self.attention = Attention(
            dim=args.dim,
            n_head=args.n_heads,
            n_kv_head=args.n_heads,           
        )

        self.ffwd = FeedForward(
            dim=args.dim,
            hidden_dim=4*args.dim,
            multiple_of= args.multiple_of          
        )


    def forward(self,x: torch.Tensor, cos_sin: torch.Tensor, mask: Union[torch.Tensor,bool]) -> torch.Tensor:
        attn_output = self.attention(
            x=self.norm(x),
            cos_sin=cos_sin,
            mask=mask
        )

        h = x + attn_output
        h_norm = norm(h)
        return h + self.ffwd(h_norm)

    def init_weights(self, init_std=None, factor=1.0):
        self.attention.reset_parameters(init_std, factor)
        self.attention_norm.reset_parameters()

        self.feed_forward.reset_parameters(init_std, factor)
        self.ffn_norm.reset_parameters()

class BaseTransformer(nn.Module):
    def __init__(self,args:BaseTransformerArgs):
        super().__init__()
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.init_base_std = args.init_base_std
        self.init_std_factor = InitStdFactor(args.init_std_factor)
        self.max_seqlen = args.max_seqlen
        self.rotary_seq_len = 10* args.max_seqlen

        
        self.eos_id = args.eos_id

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(Block(args))
    
    def forward(self,h, mask:Union[torch.Tensor,bool]):
        cos_sin = self._precompute_rotary_embeddings(seqlen=self.max_seqlen,head_dim=self.head_dim)
        for i,layer in enumerate(self.layers):
            h = layer(h,cos_sin=cos_sin,mask=mask)
        return h
    
    def init_weights(self):
        self.rope_embeddings.reset_parameters()
        for depth, layer in enumerate(self.layers):
            factor = {
                InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.layers) + 1)) ** 0.5,
                InitStdFactor.DIM_RATIO: self.dim / 4096,
                InitStdFactor.DISABLED: 1.0,
            }[self.init_std_factor]

            layer.init_weights(self.init_base_std, factor)

    def _precompute_rotary_embeddings(self,seq_len,head_dim,base=100000, device='cuda'):
        #stride the channel
        channel_range = torch.arange(0,head_dim,2,dtype=torch.float32,device=device)
        inv_freqs = 1.0 / (base ** (channel_range / head_dim))
        #stride the time step
        t = torch.arange(seq_len,dtype=torch.float32,device=device)
        freqs = torch.outer(t,inv_freqs)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.bfloat16(), sin.bfloat16()   #Maybe this needs to be checked for gpu maynot support this
        cos, sin = cos[None,:, None, :], sin[None,:, None,:]
        return cos,sin
    




