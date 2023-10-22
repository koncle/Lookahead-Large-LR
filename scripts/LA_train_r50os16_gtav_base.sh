#!/usr/bin/env bash
    # Example on Cityscapes
    # CUDA_VISIBLE_DEVICES=4,5,6,7
     python -m torch.distributed.launch --nproc_per_node=4 train.py \
        --dataset gtav \
        --covstat_val_dataset gtav \
        --val_dataset bdd100k  \
        --arch network.deepv3.DeepR50V3PlusD \
        --city_mode 'train' \
        --lr_schedule poly \
        --poly_exp 0.9 \
        --max_cu_epoch 10000 \
        --class_uniform_pct 0.5 \
        --class_uniform_tile 1024 \
        --crop_size 768 \
        --scale_min 0.5 \
        --scale_max 2.0 \
        --rrotate 0 \
        --max_iter 40000 \
        --bs_mult 4 \
        --gblur \
        --color_aug 0.5 \
        --wt_reg_weight 0.0 \
        --relax_denom 0.0 \
        --cov_stat_epoch 0 \
        --wt_layer 0 0 0 0 0 0 0 \
        --date 0101 \
        --ckpt ./logs/ \
        --tb_path ./logs/ \
        --exp LA_r50os16_gtav_base_lr-0.05_steps-15_alpha-0.05_Longer_Valid_reg0.01 \
        --alpha 0.05 \
        --lr 0.05 \
        --LA \
        --avgLA \
        --reg_strength 0.01
        # --snapshot /data/zj/PycharmProjects/RobustNet/logs/0101/LA_r50os16_gtav_base_lr-0.05_steps-15_alpha-0.05_Longer_Valid/08_24_11/last_gtav_epoch_14_mean-iu_0.66740-pre.pth \
        # --restore_optimizer
