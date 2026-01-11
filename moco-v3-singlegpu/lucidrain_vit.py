import torch
from torch import nn, stack, tensor
import torch.nn.functional as F
from torch.nn import Module, ModuleList

from einops import rearrange, repeat, reduce, einsum, pack, unpack

# classes

class FeedForward(Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        normed = self.norm(x)
        return self.net(x), normed

class Attention(Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.norm = nn.LayerNorm(dim)
        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        normed = self.norm(x)

        qkv = self.to_qkv(normed).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out), normed

class Transformer(Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):

        normed_inputs = []

        for attn, ff in self.layers:
            attn_out, attn_normed_inp = attn(x)
            x = attn_out + x

            ff_out, ff_normed_inp = ff(x)
            x = ff_out + x

            normed_inputs.append(attn_normed_inp)
            normed_inputs.append(ff_normed_inp)

        return self.norm(x), stack(normed_inputs)

class NoMLPTransformer(Module):
    def __init__(self, dim, depth, heads, dim_head, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout))

    def forward(self, x):
        normed_inputs = []

        for attn in self.layers:
            attn_out, attn_normed_inp = attn(x)
            x = attn_out + x
            normed_inputs.append(attn_normed_inp)

        return self.norm(x), stack(normed_inputs)

class ViT(Module):
    def __init__(self, *, num_patches, num_classes, dim, depth, heads, mlp_dim,
                 pool: str = 'cls', dim_head: int = 64, dropout: float = 0.,
                 emb_dropout: float = 0.):
        """
        ViT that operates directly on pre-computed patch feature maps.

        Args:
            num_patches: number of patches per image (sequence length)
            num_classes: number of output classes
            dim: feature dimension of each patch (must match backbone output dim)
            depth, heads, dim_head, mlp_dim, dropout, emb_dropout: standard ViT hyperparams
            pool: 'cls' or 'mean'
        """
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # Feature maps are already embedded, so just normalize them
        self.feature_norm = nn.LayerNorm(dim)

        # Positional embedding + CLS token
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        # Transformer encoder
        # self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        self.transformer = NoMLPTransformer(dim, depth, heads, dim_head, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        # Final classifier
        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(
        self,
        feature_maps: torch.Tensor,  # [batch, num_patches, dim]
    ):
        # Normalize input feature maps
        x = self.feature_norm(feature_maps)
        b, n, _ = x.shape

        # Add CLS token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x, normed_layer_inputs = self.transformer(x)
        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)

