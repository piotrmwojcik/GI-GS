import os
import json
from pathlib import Path

# List of datasets and map names
datasets = (
    "hook150_v3_transl_statictimestep1",
    "jumpingjacks150_v3_tex_statictimestep75",
    "mouse150_v2_transl_statictimestep1",
    "spheres_cube_dataset_v5_statictimestep1",
    "standup150_v3_statictimestep75",
)

map_names = (
    "chapel_day_4k_32x16_rot0",
    "dam_wall_4k_32x16_rot90",
    "golden_bay_4k_32x16_rot330",
)

# Base path (replace '~' with absolute path)
base_path = Path("~/GI-GS/outputs_diffuse").expanduser()

# Iterate and read each JSON file
for dataset in datasets:
    for map_name in map_names:
        json_path = base_path / dataset / map_name / "test" / "ours_None" / "pbr" / "r_0150_NVS.json"

        if json_path.is_file():
            with open(json_path, 'r') as f:
                data = json.load(f)
                print(f"Read JSON from: {json_path}")
                # Process the JSON as needed
        else:
            print(f"Missing: {json_path}")
