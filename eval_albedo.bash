#!/bin/bash

# List of dataset names
datasets=(
    hook150_v3_transl_statictimestep1
    jumpingjacks150_v3_tex_statictimestep75
    mouse150_v2_transl_statictimestep1
    spheres_cube_dataset_v5_statictimestep1
    standup150_v3_statictimestep75
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

    python render.py -m outputs/hook150_v2_statictimestep1/ -s data/hook150_v2_statictimestep1 --checkpoint outputs/hook150_v2_statictimestep1/chkpnt35000.pth --eval --skip_train --brdf_eval

  done
done
