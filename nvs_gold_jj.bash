#!/bin/bash

# List of dataset names
datasets=(
    jumpingjacks150_v3_tex_statictimestep75
)

# Loop through each dataset and call the training script
for d in "${datasets[@]}"; do
  for DATA_SUBDIR in \
    golden_bay_4k_32x16_rot330 \
    aaaa
  do
    export DATA_SUBDIR
    echo "Processing DATASET: $d with DATA_SUBDIR: $DATA_SUBDIR"

    python render.py \
    -m outputs/"$d"/$DATA_SUBDIR \
    -s data/"$d" \
    --checkpoint outputs/$DATA_SUBDIR/chkpnt35000.pth \
    --eval \
    --skip_train \
    --pbr \
    --gamma \
    --indirect
  done
done
