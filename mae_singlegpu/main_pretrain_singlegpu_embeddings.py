# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import Dataset, DataLoader
import wandb

import timm

# Version check removed - compatible with timm >= 0.3.2
import timm.optim.optim_factory as optim_factory

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler

import models_mae

from engine_pretrain import train_one_epoch


class EmbeddingDataset(Dataset):
    """Dataset that loads pre-computed embeddings, structured like ImageFolder.
    Folders are labels, files inside folders are samples (tensors loaded via torch.load).
    """
    def __init__(self, data_path):
        self.data_path = data_path
        self.samples = []
        self.classes = []
        self.class_to_idx = {}
        
        # Find all class folders
        if not os.path.isdir(data_path):
            raise ValueError(f"Data path {data_path} is not a directory")
        
        # Get all subdirectories (classes)
        for class_name in sorted(os.listdir(data_path)):
            class_path = os.path.join(data_path, class_name)
            if os.path.isdir(class_path):
                class_idx = len(self.classes)
                self.classes.append(class_name)
                self.class_to_idx[class_name] = class_idx
                
                # Get all tensor files in this class folder
                for file_name in sorted(os.listdir(class_path)):
                    if file_name.endswith('.pt'):
                        file_path = os.path.join(class_path, file_name)
                        self.samples.append((file_path, class_idx))
        
        print(f"Found {len(self.classes)} classes with {len(self.samples)} total samples")
        print(f"Classes: {self.classes}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        embedding = torch.load(file_path, map_location='cpu')
        
        # Ensure shape is (patches, dim) - if batch dimension exists, take first
        if len(embedding.shape) == 3:  # (B, patches, dim)
            embedding = embedding[0]
        elif len(embedding.shape) != 2:
            raise ValueError(f"Expected 2D (patches, dim) or 3D (B, patches, dim), got {embedding.shape}")
        
        return embedding.float(), label


class MaskedAutoencoderViT_NoPatchEmbed(models_mae.MaskedAutoencoderViT):
    """MAE ViT that skips patch embedding - expects pre-computed embeddings"""
    def __init__(self, embed_dim, num_patches, norm_pix_loss=False, **kwargs):
        from functools import partial
        
        # Hardcode mae_vit_base_patch16 defaults - pass directly like mae_vit_base_patch16_dec512d8b
        decoder_embed_dim = 512
        super().__init__(
            patch_size=16, embed_dim=embed_dim, depth=12, num_heads=16,
            decoder_embed_dim=decoder_embed_dim, decoder_depth=8, decoder_num_heads=16,
            mlp_ratio=4, norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            norm_pix_loss=norm_pix_loss, **kwargs
        )
        
        # Store num_patches
        self.num_patches = num_patches
        
        # Replace patch_embed with identity (won't be used in forward_encoder)
        self.patch_embed = torch.nn.Identity()
        
        # Replace position embeddings with correct num_patches
        from util.pos_embed import get_2d_sincos_pos_embed
        pos_embed = get_2d_sincos_pos_embed(embed_dim, int(num_patches**.5), cls_token=True)
        self.pos_embed = nn.Parameter(torch.from_numpy(pos_embed).float().unsqueeze(0), requires_grad=False)
        
        decoder_pos_embed = get_2d_sincos_pos_embed(decoder_embed_dim, int(num_patches**.5), cls_token=True)
        self.decoder_pos_embed = nn.Parameter(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0), requires_grad=False)
        
        # Change decoder to predict embeddings instead of pixels
        # decoder_pred currently: decoder_embed_dim -> patch_size**2 * in_chans
        # Change to: decoder_embed_dim -> embed_dim
        self.decoder_pred = torch.nn.Linear(decoder_embed_dim, embed_dim, bias=True)
    
    def forward_encoder(self, x, mask_ratio):
        # x is already (B, num_patches, embed_dim), skip patch embedding
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        
        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)
        
        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        
        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        
        return x, mask, ids_restore
    
    def forward_loss(self, embeddings, pred, mask):
        """
        embeddings: [N, L, D] - input embeddings
        pred: [N, L, D] - predicted embeddings
        mask: [N, L], 0 is keep, 1 is remove
        """
        target = embeddings  # Target is the input embeddings
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss
    
    def forward(self, embeddings, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(embeddings, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, embed_dim]
        loss = self.forward_loss(embeddings, pred, mask)
        return loss, pred, mask


def get_args_parser():
    parser = argparse.ArgumentParser('MAE pre-training (single GPU, embeddings)', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter)')
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Model parameters
    parser.add_argument('--embed_dim', default=768, type=int,
                        help='Embedding dimension (default: 768 for base)')
    parser.add_argument('--num_patches', default=196, type=int,
                        help='Number of patches (default: 196 for 224x224 image with patch_size=16)')

    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches).')

    parser.add_argument('--norm_pix_loss', action='store_true',
                        help='Use (per-patch) normalized pixels as targets for computing loss')
    parser.set_defaults(norm_pix_loss=False)

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=40, metavar='N',
                        help='epochs to warmup LR')

    # Dataset parameters
    parser.add_argument('--data_path', required=True, type=str,
                        help='path to directory containing embedding files (.pt or .npy)')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--wandb_project', default='mae-pretrain', type=str,
                        help='wandb project name')
    parser.add_argument('--wandb_name', default=None, type=str,
                        help='wandb run name')
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    return parser


def main(args):
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires a GPU.")

    # fix the seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cudnn.benchmark = True

    # Load embedding dataset
    dataset_train = EmbeddingDataset(args.data_path)
    print(f"Dataset size: {len(dataset_train)}")
    
    # Check first sample to verify dimensions
    if len(dataset_train) > 0:
        sample, _ = dataset_train[0]
        print(f"Sample shape: {sample.shape}")
        print(f"Expected shape: (num_patches={args.num_patches}, embed_dim={args.embed_dim})")
        if sample.shape[0] != args.num_patches or sample.shape[1] != args.embed_dim:
            print(f"Warning: Sample shape {sample.shape} doesn't match expected (num_patches={args.num_patches}, embed_dim={args.embed_dim})")

    # Use RandomSampler for single GPU
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    print("Sampler_train = %s" % str(sampler_train))

    # Initialize wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=vars(args)
    )

    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    
    # define the model - hardcoded to mae_vit_base_patch16 with custom embed_dim
    model = MaskedAutoencoderViT_NoPatchEmbed(
        embed_dim=args.embed_dim,
        num_patches=args.num_patches,
        norm_pix_loss=args.norm_pix_loss
    )

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    # Effective batch size for single GPU (no world_size multiplier)
    eff_batch_size = args.batch_size * args.accum_iter
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    # following timm: set wd as 0 for bias and norm layers
    param_groups = optim_factory.param_groups_weight_decay(model_without_ddp, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95))
    print(optimizer)
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=None,  # No tensorboard logging
            args=args
        )
        
        # Log to wandb
        wandb.log({
            'train_loss': train_stats['loss'],
            'train_lr': train_stats['lr'],
            'epoch': epoch
        })
        if args.output_dir and (epoch % 20 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        'epoch': epoch,}

        if args.output_dir:
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)

