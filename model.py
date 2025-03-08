import torch
import torch.nn as nn
import torch.nn.functional as F

from dit_block import DiTBlock, FinalLayer
from patch_embed import PatchEmbed


class TimeStepEmbedding(nn.Module):
    def __init__(self, dim, freq_emb_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(freq_emb_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.freq_emb_dim = freq_emb_dim

    def pos_enc(self, t):
        denom = torch.tensor(10000) ** (
            (2 * torch.arange(self.freq_emb_dim)) / self.freq_emb_dim
        ).to(t.device)
        embeddings = t[:, None] * denom[None, :].to(t.device)
        embeddings[:, ::2] = embeddings[:, ::2].sin()
        embeddings[:, 1::2] = embeddings[:, 1::2].cos()
        return embeddings

    def forward(self, t):
        t_emb = self.pos_enc(t)
        t_emb = self.mlp(t_emb)
        return t_emb


class ClassEmbedding(nn.Module):
    def __init__(self, num_classes, dim):
        super().__init__()
        self.embed = nn.Embedding(num_classes, dim)
        self.num_classes = num_classes
        self.cond_mlp = nn.Linear(dim, dim)

    def forward(self, y):
        return self.cond_mlp(self.embed(y))


# input_size=32,
# patch_size=2,
# in_channels=3,
# dim=384,
# depth=12,
# num_heads=6,
# num_classes=10,
# learn_sigma=False,
# class_dropout_prob=0.1,


class DiT(nn.Module):
    def __init__(
        self,
        input_size,
        patch_size,
        in_channels,
        dim,
        depth,
        num_heads,
        hidden_scale,
        pos_encoding,
        num_classes,
        class_dropout_prob=0.1,
    ):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.class_dropout_prob = class_dropout_prob
        self.patch_size = patch_size
        self.input_size = input_size
        self.hidden_scale = hidden_scale
        self.pos_encoding = pos_encoding

        self.c_embedder = ClassEmbedding(num_classes, dim)
        self.t_embedder = TimeStepEmbedding(dim)
        self.x_embedder = PatchEmbed(
            input_size, patch_size, in_chans=in_channels, embed_dim=dim
        )
        self.in_channels = in_channels
        self.out_channels = in_channels

        self.blocks = nn.ModuleList(
            [
                DiTBlock(
                    dim=dim,
                    hidden_scale=hidden_scale,
                    num_heads=num_heads,
                    causal=False,
                    positional_encoding=pos_encoding,
                )
                for _ in range(depth)
            ]
        )

        self.final_layer = FinalLayer(dim=dim, patch_size=patch_size, out_dim=self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # initialize transformer layers
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # initialize patch_embed like nn.Linear
        w = self.x_embedder.projection.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.projection.bias, -1)

        # initialize label embedding table
        nn.init.normal_(self.c_embedder.embed.weight, std=0.02)

        # initialize timestep embedding MLP
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero out adaLn modulation blocks
        for block in self.blocks:
            nn.init.constant_(block.ada_ln.condition_proj.weight, 0)
            nn.init.constant_(block.ada_ln.condition_proj.bias, 0)

        # zero out output latyers
        nn.init.constant_(self.final_layer.ada_ln.condition_proj.weight, 0)
        nn.init.constant_(self.final_layer.ada_ln.condition_proj.bias, 0)
        nn.init.constant_(self.final_layer.out_proj.weight, 0)
        nn.init.constant_(self.final_layer.out_proj.bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size * patch_size * C)

        returns imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)  # h,w of each patch
        assert h * w == x.shape[1]
        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs 
    
    def noise_batch(self, x, t):
        """
        x: Batch of images of shape (N, C, H, W)
        t: Batch of t values of shape (N, 1)

        returns 
        x_t: noised batch of images (N, C, H, W)
        epsilon: noise added to the batch (N, C, H, W)
        """
        t = t[:, None, None, None]
        epsilon = torch.randn_like(x, device=x.device)

        # rectified flow
        x_t = (1 - t)*x + t*epsilon

        return x_t, epsilon

        
    def forward(self, x_t, t, c, drop_class=False):
        """
        x_t: Batch of images of shape (N, C, H, W) 
        t: Batch of t values of shape (N, )
        c: Batch of class labels of shape (N, )
        """
        x = self.x_embedder(x_t) # patch embedding along with positional
        t = self.t_embedder(t)
        c = self.c_embedder(c)
        if drop_class:
            c = torch.zeros_like(c)
        
        c = c + t
        for block in self.blocks:
            x = block(x, c)

        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x

        



if __name__ == "__main__":
    x = torch.randn(2, 3, 32, 32)
    t = torch.randn(2)
    c = torch.randint(0, 10, (2,))
    model = DiT(
        input_size=32,
        patch_size=2,
        in_channels=3,
        dim=384,
        depth=12,
        num_heads=6,
        num_classes=10,
        hidden_scale=4,
        pos_encoding="rope",
    )
    # print(x.shape, t.shape, c.shape)
    out = model(x, t, c)
    print(out.shape)
    # print(out)
    # print(model)
    # print("Number of parameters: ", sum(p.numel() for p in model.parameters()))

        

        
    