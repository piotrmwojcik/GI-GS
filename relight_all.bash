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
  pairs=(
    "chapel_day_4k_32x16_rot0 golden_bay_4k_32x16_rot330"
    #"chapel_day_4k_32x16_rot0 dam_wall_4k_32x16_rot90"
    "dam_wall_4k_32x16_rot90 small_harbour_sunset_4k_32x16_rot270"
    #"dam_wall_4k_32x16_rot90 golden_bay_4k_32x16_rot330"
    "golden_bay_4k_32x16_rot330 dam_wall_4k_32x16_rot90"
    #"golden_bay_4k_32x16_rot330 chapel_day_4k_32x16_rot0"
  )

  for pair in "${pairs[@]}"; do
    read DATA_SUBDIR MAP_NAME <<< "$pair"
    export DATA_SUBDIR
    export MAP_NAME
    DATASET=$d
    export DATASET
    echo "Processing DATASET: $d with DATA_SUBDIR: $DATA_SUBDIR and MAP_NAME: $MAP_NAME"

    # Uncomment below to execute your actual command
    #python relight.py \
    #  -m outputs_diffuse/$d/$DATA_SUBDIR/ \
    #  -s data/$d/ \
    #  --checkpoint outputs_diffuse/$d/$DATA_SUBDIR/chkpnt35000.pth \
    #  --hdri data/$d/$MAP_NAME.hdr \
    # --eval \
    #  --gamma


    python relight_eval.py \
      --output_dir outputs_diffuse_appendix/$d/$DATA_SUBDIR/test/ours_None/relight/ \
      --gt_dir data/$d/
  done
done
