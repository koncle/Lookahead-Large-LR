#!/usr/bin/env bash
# --val_dataset synthia bdd100k cityscapes mapillary \
echo "Running inference on" ${1}

     python -m torch.distributed.launch --nproc_per_node=4 valid.py \
        --val_dataset cityscapes bdd100k  mapillary synthia \
        --arch network.deepv3.DeepR50V3PlusD \
        --wt_layer 0 0 0 0 0 0 0 \
        --date 0101 \
        --exp test \
        --snapshot ${1}

