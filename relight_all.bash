#!/bin/bash

# List of dataset names
datasets=(
    hook150_v3_transl_statictimestep1
    jumpingjacks150_v3_tex_statictimestep75
    mouse150_v2_transl_statictimestep1
    spheres_cube_dataset_v7_statictimestep1
    standup150_v3_statictimestep75
)

# Loop through each dataset and call the training script
for d in "${datasets[@]}"; do
  for pair in "${pairs[@]}"; do
    pairs=(
      "chapel_day_4k_32x16_rot0 golden_bay_4k_32x16_rot330"
      "dam_wall_4k_32x16_rot90 small_harbour_sunset_4k_32x16_rot270"
      "golden_bay_4k_32x16_rot330 dam_wall_4k_32x16_rot90"
    )
    read DATA_SUBDIR MAP_NAME <<< "$pair"
    export DATA_SUBDIR
    echo "Processing DATASET: $d with DATA_SUBDIR: $DATA_SUBDIR and MAP_NAME $MAP_NAME"

    #python relight.py \
    #  -m outputs/bicycle \
    #  -s datasets/nerf_real_360/bicycle/ \
    #  --checkpoint outputs/bicycle/chkpnt40000.pth \
    #  --hdri datasets/TensoIR/Environment_Maps/high_res_envmaps_2k/bridge.hdr \
    #  --eval \
    #  --gamma
    done
  done
done
