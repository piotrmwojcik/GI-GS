#!/bin/bash

# List of dataset names
datasets=(
  #hook150_v2_statictimestep1
  jumpingjacks200_statictimestep100
  mouse150_statictimestep1
  spheres_cube_dataset_v4_2_onetimestep90
  standup150_v2_statictimestep75
)

# Loop through each dataset and call the training script
for d in "${datasets[@]}"; do
  echo "Training on $d"
  python train.py -m outputs/"$d" -s data/"$d" --iterations 35000 --eval --gamma --radius 0.8 --bias 0.01 --thick 0.05 --delta 0.0625 --step 16 --start 64 --indirect
done