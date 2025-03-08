import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega) # outerproduct

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)
    emb = np.concatenate([emb_sin, emb_cos], axis=1) # (M, D)
    return emb

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    if embed_dim % 2 != 0:
        raise ValueError("embed_dim must be divisible by 2")

    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])

    emb = np.concatenate([emb_h, emb_w], axis=1)
    return emb

def get_2d_sincos_pos_embed(embed_dim, grid_size, base_size, interpolation_scale):
    if isinstance(grid_size, int):
        grid_size = (grid_size, grid_size)  
    
    grid_h = np.arange(grid_size[0], dtype=np.float32) / (grid_size[0] / base_size) / interpolation_scale
    grid_w = np.arange(grid_size[1], dtype=np.float32) / (grid_size[1] / base_size) / interpolation_scale

    grid = np.meshgrid(grid_w, grid_h)
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size[1], grid_size[0]])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    return pos_embed

class PatchEmbed(nn.Module):
    def __init__(self, 
                 img_size, 
                 patch_size, 
                 in_chans, 
                 embed_dim,
                 positional_encoding="absolute"
                 ):
        super().__init__()
        assert positional_encoding in ["absolute", "rope"]
        self.embed_dim = embed_dim
        self.img_size = img_size

        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        if positional_encoding == "absolute":
            # we are setting grid size here to be exactly the same
            # as input image but we can have a bigger grid size
            # in case of cropping 
            grid_size = img_size // patch_size
            base_size = img_size // patch_size
            pos_embed = get_2d_sincos_pos_embed(embed_dim, 
                                                grid_size, 
                                                base_size, 
                                                1) 
            persistent = True # since we use same grid size for all images
            self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=persistent)
        elif positional_encoding == "rope":
            self.pos_embed = None

        self.projection = nn.Conv2d(
            in_chans, 
            embed_dim, 
            kernel_size=patch_size, 
            stride=patch_size)
            

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size and W == self.img_size, \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size}*{self.img_size})."
        x = self.projection(x).flatten(2).transpose(1, 2)
        if self.pos_embed is None:
            return x
        return x + self.pos_embed


if __name__ == "__main__":
    p = PatchEmbed(32, 2, 3, 512)
    x = torch.randn(1, 3, 32, 32)
    out = p(x)
    print(p.num_patches)
    print(out.shape)
    pos_embed = get_2d_sincos_pos_embed(512, 16, 16, 1)
    print(pos_embed.shape)
    print(pos_embed)
    

