import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import Attention

def modulate(x, scale, shift):
    # when scale = 0 and shift is 0 the transformation is equivalent to identity
    return x * (1 + scale[:, None, :]) + shift[:, None, :]


class AdaLayerNormZero(nn.Module):
    def __init__(self, dim, final=False):
        """
        This block simply modulates and doesn't actually apply layernorm
        """
        super().__init__()
        self.silu = nn.SiLU()
        self.final = final
        if final:
            # if final adaln, we only need scale and shift
            self.condition_proj = nn.Linear(dim, dim * 2)
        else:
            self.condition_proj = nn.Linear(dim, dim * 6)

    def forward(self, c):
        # c is the concatenation of time embedding and class embedding
        c_emb = self.condition_proj(self.silu(c))
        if self.final:
            scale, shift = torch.chunk(c_emb, 2, dim=-1)
            return scale, shift

        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(c_emb, 6, dim=-1)
        return shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp

class FinalLayer(nn.Module):
    def __init__(self, dim, patch_size, out_dim):
        super().__init__()
        self.norm_final = nn.LayerNorm(dim, elementwise_affine=False)
        self.ada_ln = AdaLayerNormZero(dim, final=True)
        self.out_proj = nn.Linear(dim, patch_size * patch_size * out_dim)

    def forward(self, x, c):
        shift, scale = self.ada_ln(c)
        x = modulate(self.norm_final(x), scale, shift)
        x = self.out_proj(x)
        return x
        

class MLP(nn.Module):
    def __init__(self, dim, hidden_scale=4.0, act="swiglu"):
        super().__init__()
        self.act_ = act
        self.proj_size = int(dim * hidden_scale)
        if act == "swiglu":
            self.lin_up = nn.Linear(dim, self.proj_size * 2)
            self.lin_down = nn.Linear(self.proj_size, dim)
            self.act = F.silu
        elif act == "gelu":
            self.lin_up = nn.Linear(dim, self.proj_size)
            self.lin_down = nn.Linear(self.proj_size, dim)
            self.act = F.gelu

    def forward(self, x):
        if self.act_ == "swiglu":
            up_proj, gate = self.lin_up(x).split(self.proj_size, dim=-1)
            up_proj = up_proj * self.act(gate)
        elif self.act_ == "gelu":
            up_proj = self.lin_up(x)
            up_proj = self.act(up_proj)

        x = self.lin_down(up_proj)
        return x
    

class DiTBlock(nn.Module):
    def __init__(self, 
                 dim, 
                 hidden_scale=4.0, 
                 num_heads=8, 
                 causal=False,
                 positional_encoding="rope"
                 ):
        super().__init__()
        assert positional_encoding in ["absolute", "rope"]
        self.ada_ln = AdaLayerNormZero(dim) 
        self.ln1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(dim, elementwise_affine=False)
        self.attn = Attention(dim, num_heads=num_heads, causal=causal, positional_encoding=positional_encoding)
        self.mlp = MLP(dim, hidden_scale=hidden_scale)

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp  = self.ada_ln(c)
        # attention
        x = (self.attn(modulate(self.ln1(x), scale_msa, shift_msa)) \
                * gate_msa[:, None, :]) + x
        # MLP
        x = (self.mlp(modulate(self.ln2(x), scale_mlp, shift_mlp)) \
                * gate_mlp[:, None, :]) + x

        return x


if __name__ == "__main__":
    x = torch.randn(1, 256, 512)
    norm = DiTBlock(512, 512)
    c = torch.randn(1, 512)
    out= norm(x, c)
    print(out.shape)
