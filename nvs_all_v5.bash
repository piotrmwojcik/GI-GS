#!/bin/bash

# List of dataset names
datasets=(
    hook150_v5_spec32_statictimestep1
    jumpingjacks150_v5_spec32_statictimestep75
    mouse150_v5_spec32_statictimestep1
    spheres_v5_spec32_statictimestep1
    standup150_v5_spec32_statictimestep75
)

# Loop through each dataset and call the training script
for d in "${datasets[@]}"; do
  for DATA_SUBDIR in \
    chapel_day_4k_32x16_rot0 \
    dam_wall_4k_32x16_rot90 \
    golden_bay_4k_32x16_rot330
  do
    export DATA_SUBDIR
    echo "Processing DATASET: $d with DATA_SUBDIR: $DATA_SUBDIR"

    python render.py \
    -m outputs_specular_JULY/"$d"/$DATA_SUBDIR \
    -s data_specular_new/datasets_v5_specular32/"$d" \
    --checkpoint outputs_specular_JULY/"$d"/$DATA_SUBDIR/chkpnt35000.pth \
    --eval \
    --skip_train \
    --pbr \
    --gamma \
    --indirect
  done
done
