python main_pretrain_singlegpu_embeddings.py \
    --epochs 200 \
    --embed_dim 512 \
    --data_path ../moco-v3-singlegpu/embeddings/imagenette2_no_shared_initial_crop_512_dim/train/ \
    --num_patches 49 \
    --wandb_name mae_pretrain_512_dim_imagenette_no_shared_initial_crop \
    --output_dir ./output_dir/mae_pretrain_512_dim_imagenette_no_shared_initial_crop

python main_pretrain_singlegpu_embeddings.py \
    --epochs 200 \
    --embed_dim 512 \
    --data_path ../moco-v3-singlegpu/embeddings/imagenette2_shared_initial_crop_512_dim/train/ \
    --num_patches 49 \
    --wandb_name mae_pretrain_512_dim_imagenette_shared_initial_crop \
    --output_dir ./output_dir/mae_pretrain_512_dim_imagenette_shared_initial_crop