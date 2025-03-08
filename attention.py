import torch
import torch.nn as nn
import torch.nn.functional as F
from rotary_embedding import RotaryEmbedding

class Attention(nn.Module):
    def __init__(self, dim, num_heads, causal=False, positional_encoding="rope"): 
        super().__init__()
        assert dim % num_heads == 0, 'dim must be divisible by num_heads'
        assert positional_encoding in ["absolute", "rope"]

        self.positional_encoding = positional_encoding
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q_proj = nn.Linear(dim, dim, bias=False) 
        self.k_proj = nn.Linear(dim, dim, bias=False) 
        self.v_proj = nn.Linear(dim, dim, bias=False) 
        self.out_proj = nn.Linear(dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(dim, dim)
        self.k_norm = nn.RMSNorm(dim, dim)

        self.scale = self.head_dim ** -0.5
        
        if positional_encoding == "rope":
            self.rotary_emb = RotaryEmbedding(self.head_dim)
        
        self.causal = causal

    def forward(self, x):
        B,T,C = x.shape

        queries = self.q_norm(self.q_proj(x)).reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        keys = self.k_norm(self.k_proj(x)).reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        values = self.v_proj(x).reshape(B, T, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        if self.positional_encoding == "rope":
            queries = self.rotary_emb.rotate_queries_or_keys(queries)
            keys = self.rotary_emb.rotate_queries_or_keys(keys)

        attn = F.scaled_dot_product_attention(queries, keys, values, is_causal=self.causal, scale=self.scale)
        out = self.out_proj(attn.permute(0, 2, 1, 3).reshape(B, T, -1))
        return out


# if __name__ == "__main__":
#     attn = Attention(512, 8)
#     x = torch.randn(1, 256, 512)
#     out = attn(x)
#     print(out.shape)
#     print(out)


        


