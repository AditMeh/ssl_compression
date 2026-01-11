"""
Vision Transformer (ViT) module for patch-based image classification.

Takes a pretrained backbone, extracts patch embeddings, adds positional
embeddings and a CLS token, processes through transformer layers, and
classifies using the CLS token output.
"""

import torch
import torch.nn as nn


class TransformerEncoderLayer(nn.Module):
    """
    Standard Transformer encoder layer with multi-head self-attention and FFN.
    """
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class ViTClassifier(nn.Module):
    """
    Vision Transformer classifier that:
    1. Splits input image into patches
    2. Embeds each patch using a frozen backbone
    3. Projects embeddings and adds positional embeddings + CLS token
    4. Processes through transformer encoder layers
    5. Classifies using the CLS token
    """
    def __init__(self, backbone, embed_dim=2048, num_classes=10, 
                 patch_size=32, input_size=224, attention_hidden_dim=512,
                 mlp_hidden_dim=512, projection_dim=512,
                 num_heads=8, num_layers=6, mlp_ratio=4.0, dropout=0.1,
                 use_learned_patch_embed=False, in_channels=3):
        """
        Args:
            backbone: Pretrained model with fc/head removed (outputs embed_dim features)
            embed_dim: Dimension of backbone output (2048 for ResNet50)
            num_classes: Number of output classes
            patch_size: Size of each patch (patches are patch_size x patch_size)
            input_size: Size to resize input images to (input_size x input_size)
            attention_hidden_dim: Unused, kept for backwards compatibility
            mlp_hidden_dim: Unused, kept for backwards compatibility
            projection_dim: Dimension of transformer embeddings (transformer_dim)
            num_heads: Number of attention heads
            num_layers: Number of transformer encoder layers
            mlp_ratio: Ratio of MLP hidden dim to embed dim in transformer FFN
            dropout: Dropout rate
            use_learned_patch_embed: If True, use standard ViT learned Conv2d projection 
                                     instead of frozen backbone (default: False)
            in_channels: Number of input channels (default: 3 for RGB)
        """
        super().__init__()
        
        # Use projection_dim as transformer_dim for backwards compatibility
        transformer_dim = projection_dim
        
        self.backbone = backbone
        self.embed_dim = embed_dim
        self.transformer_dim = transformer_dim
        self.patch_size = patch_size
        self.input_size = input_size
        self.use_learned_patch_embed = use_learned_patch_embed
        
        # Calculate number of patches (224/32 = 7, so 7x7 = 49 patches)
        assert input_size % patch_size == 0, \
            f"input_size ({input_size}) must be divisible by patch_size ({patch_size})"
        self.num_patches_per_side = input_size // patch_size
        self.num_patches = self.num_patches_per_side ** 2

        if use_learned_patch_embed:
            # Standard ViT patch embedding: Conv2d that projects patches directly
            # Conv2d with kernel_size=stride=patch_size acts as a linear projection per patch
            self.patch_embed = nn.Conv2d(
                in_channels, transformer_dim, 
                kernel_size=patch_size, stride=patch_size
            )
            self.patch_projector = None  # Not needed
        else:
            # Freeze backbone
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # Patch embedding projector (projects from backbone embed_dim to transformer_dim)
            self.patch_projector = nn.Sequential(
                nn.Linear(embed_dim, transformer_dim),
                nn.GELU(),
                nn.Linear(transformer_dim, transformer_dim)
            )
            self.patch_embed = None  # Not needed
        
        # CLS token (learnable class embedding)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, transformer_dim))
        
        # Positional embeddings (for num_patches + 1 CLS token)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, transformer_dim))
        
        # Transformer encoder layers
        self.transformer_layers = nn.ModuleList([
            TransformerEncoderLayer(transformer_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(transformer_dim)
        
        # Classification head (predicts class logits from CLS token)
        self.classifier = nn.Linear(transformer_dim, num_classes)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        # Initialize CLS token
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize positional embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        if self.use_learned_patch_embed:
            # Initialize patch embedding conv
            nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
            if self.patch_embed.bias is not None:
                nn.init.zeros_(self.patch_embed.bias)
        else:
            # Initialize projector
            for m in self.patch_projector.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, std=0.02)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)
        
        # Initialize classifier
        nn.init.trunc_normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def extract_patches(self, x):
        """
        Extract non-overlapping patches from input images.
        
        Args:
            x: (B, C, H, W) input images, assumed to be input_size x input_size
        Returns:
            patches: (B * num_patches, C, patch_size, patch_size)
        """
        B, C, H, W = x.shape
        
        # Use unfold to extract patches
        patches = x.unfold(2, self.patch_size, self.patch_size)  # (B, C, n_h, W, patch_size)
        patches = patches.unfold(3, self.patch_size, self.patch_size)  # (B, C, n_h, n_w, patch_size, patch_size)
        
        # Reshape to (B, num_patches, C, patch_size, patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(B, self.num_patches, C, self.patch_size, self.patch_size)
        
        # Reshape to (B * num_patches, C, patch_size, patch_size) for backbone
        patches = patches.view(B * self.num_patches, C, self.patch_size, self.patch_size)
        
        return patches
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: (B, C, H, W) input images
        Returns:
            logits: (B, num_classes) classification logits
        """
        B = x.shape[0]
        
        if self.use_learned_patch_embed:
            # Standard ViT: Conv2d directly embeds patches
            # (B, C, H, W) -> (B, transformer_dim, num_patches_h, num_patches_w)
            patch_embeddings = self.patch_embed(x)
            # Flatten spatial dims and transpose: (B, transformer_dim, num_patches) -> (B, num_patches, transformer_dim)
            patch_embeddings = patch_embeddings.flatten(2).transpose(1, 2)  # (B, num_patches, transformer_dim)
        else:
            # Use frozen backbone for patch embedding
            # Extract patches: (B * num_patches, C, patch_size, patch_size)
            patches = self.extract_patches(x)
            
            # Get patch embeddings from frozen backbone
            with torch.no_grad():
                self.backbone.eval()
                patch_embeddings = self.backbone(patches)  # (B * num_patches, embed_dim)
            
            # Reshape to (B, num_patches, embed_dim)
            patch_embeddings = patch_embeddings.view(B, self.num_patches, self.embed_dim)
            
            # Project patch embeddings to transformer dimension
            patch_embeddings = self.patch_projector(patch_embeddings)  # (B, num_patches, transformer_dim)
        
        # Expand CLS token for batch
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, transformer_dim)
        
        # Prepend CLS token to patch embeddings
        x = torch.cat([cls_tokens, patch_embeddings], dim=1)  # (B, num_patches + 1, transformer_dim)
        
        # Add positional embeddings
        x = x + self.pos_embed
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        # Apply final layer norm
        x = self.norm(x)
        
        # Extract CLS token output
        cls_output = x[:, 0]  # (B, transformer_dim)
        
        # Classification
        logits = self.classifier(cls_output)  # (B, num_classes)
        
        return logits
    
    def get_trainable_parameters(self):
        """Return only trainable parameters (patch embed/projector + transformer + classifier)."""
        params = []
        if self.use_learned_patch_embed:
            params.extend(self.patch_embed.parameters())
        else:
            params.extend(self.patch_projector.parameters())
        params.append(self.cls_token)
        params.append(self.pos_embed)
        for layer in self.transformer_layers:
            params.extend(layer.parameters())
        params.extend(self.norm.parameters())
        params.extend(self.classifier.parameters())
        return params


def create_backbone(model, arch='resnet50'):
    """
    Remove the final classification layer from a model to use as backbone.
    
    Args:
        model: Full model (e.g., ResNet50)
        arch: Architecture name to determine which layer to remove
    
    Returns:
        backbone: Model without final classification layer
        embed_dim: Dimension of backbone output
    """
    if arch.startswith('vit'):
        # For ViT, remove the head
        embed_dim = model.head.in_features
        model.head = nn.Identity()
    else:
        # For ResNet, replace fc with identity
        embed_dim = model.fc.in_features
        model.fc = nn.Identity()
    
    return model, embed_dim
