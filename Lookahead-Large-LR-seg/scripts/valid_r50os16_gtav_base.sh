#!/usr/bin/env bash
echo "Running inference on" ${1}

     python -m torch.distributed.launch --nproc_per_node=4 valid.py \
        --val_dataset bdd100k cityscapes mapillary \
        --arch network.deepv3.DeepR50V3PlusD \
        --wt_layer 0 0 0 0 0 0 0 \
        --date 0101 \
        --exp r50os16_gtav_base \
        --snapshot ${1}
