# python main_moco_single_gpu.py /datasets/imagenet/ \
#     -a resnet50 \
#     -b 256 \
#     --epochs 100 \
#     --workers 8 \
#     --crop-size 32 \
#     --wandb-project mocov3_singlegpu \
#     --wandb-name no_shared_initial_crop_moco \
#     --checkpoint-dir ./checkpoints_no_shared_initial_crop

# python main_moco_single_gpu.py ./imagenette2 \
#     -a resnet50 \
#     -b 256 \
#     --epochs 100 \
#     --workers 8 \
#     --crop-size 32 \
#     --wandb-project mocov3_singlegpu \
#     --wandb-name no_shared_initial_crop_moco \
#     --checkpoint-dir ./checkpoints_no_shared_initial_crop

python main_moco_single_gpu.py ../datasets/imagenette2/ \
    -a resnet50 \
    -b 256 \
    --epochs 200 \
    --workers 8 \
    --crop-size 32 \
    --final-layer-planes 128 \
    --wandb-project fixed-size-compression-ssl \
    --wandb-name shared_initial_crop_moco_512_dim_imagenette2 \
    --checkpoint-dir ./checkpoints_shared_initial_crop_512_dim_imagenette2 \
    --use-shared-initial-crop \
    --seed 42

python main_moco_single_gpu.py ../datasets/imagenette2/ \
    -a resnet50 \
    -b 256 \
    --epochs 200 \
    --workers 8 \
    --crop-size 32 \
    --final-layer-planes 128 \
    --wandb-project fixed-size-compression-ssl \
    --wandb-name no_shared_initial_crop_moco_512_dim_imagenette2 \
    --checkpoint-dir ./checkpoints_no_shared_initial_crop_512_dim_imagenette2 \
    --seed 42