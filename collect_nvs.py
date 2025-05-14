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

# Base path
base_path = Path("~/GI-GS/outputs_diffuse").expanduser()

# Iterate through datasets and map names
for dataset in datasets:
    for map_name in map_names:
        folder = base_path / dataset / map_name / "test" / "ours_None" / "pbr"

        if not folder.is_dir():
            print(f"Folder not found: {folder}")
            continue

        # Look for any .json file containing "NVS" in the filename
        json_files = list(folder.glob("*NVS*.json"))

        if not json_files:
            print(f"No NVS JSON files in: {folder}")
            continue

        psnr_vals, ssim_vals, lpips_vals = [], [], []

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    psnr_vals.append(data.get("psnr_avg"))
                    ssim_vals.append(data.get("ssim_avg"))
                    lpips_vals.append(data.get("lpips_avg"))
            except Exception as e:
                print(f"Error reading {json_file}: {e}")

        # Compute and print averages
        if psnr_vals and ssim_vals and lpips_vals:
            avg_psnr = sum(psnr_vals) / len(psnr_vals)
            avg_ssim = sum(ssim_vals) / len(ssim_vals)
            avg_lpips = sum(lpips_vals) / len(lpips_vals)

            print(f"\nResults for {dataset} | {map_name}:")
            print(f"  Average PSNR  : {avg_psnr:.3f}")
            print(f"  Average SSIM  : {avg_ssim:.3f}")
            print(f"  Average LPIPS : {avg_lpips:.3f}")
        else:
            print(f"Missing metrics in JSONs for: {folder}")
