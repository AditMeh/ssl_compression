python main_pretrain_singlegpu_embeddings.py \
    --epochs 100 \
    --embed_dim 512 \
    --data_path ../moco-v3-singlegpu/imagenette2_tensors_128_dim/train/ \
    --num_patches 49 \
    --wandb_name mae_pretrain_512_dim_imagenette