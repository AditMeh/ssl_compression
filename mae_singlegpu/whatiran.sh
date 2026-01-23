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

python extract_features.py \
    output_dir/mae_pretrain_512_dim_imagenette_no_shared_initial_crop/checkpoint-199.pth \
    ../moco-v3-singlegpu/embeddings/imagenette2_no_shared_initial_crop_512_dim/ \
    --output-dir pooled_embeddings_512_dim_no_shared_initial_crop \
    --embed-dim 512 \
    --num-patches 49

python extract_features.py \
    output_dir/mae_pretrain_512_dim_imagenette_shared_initial_crop/checkpoint-199.pth \
    ../moco-v3-singlegpu/embeddings/imagenette2_shared_initial_crop_512_dim/ \
    --output-dir pooled_embeddings_512_dim_shared_initial_crop \
    --embed-dim 512 \
    --num-patches 49

python linprobe.py \
    --data-path pooled_embeddings_512_dim_no_shared_initial_crop \
    --wandb-project fixed-size-compression-ssl \
    --wandb-name probe_mae_no_shared_initial_crop_512_dim 

python linprobe.py \
    --data-path pooled_embeddings_512_dim_shared_initial_crop \
    --wandb-project fixed-size-compression-ssl \
    --wandb-name probe_mae_shared_initial_crop_512_dim 
