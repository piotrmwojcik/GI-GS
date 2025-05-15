import os
import json
import math
from pathlib import Path

# Datasets and map names
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

# Global value lists
global_psnr, global_ssim, global_lpips = [], [], []

# Function to compute mean and std
def compute_mean_std(values):
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))
    return mean, std

# Iterate through all dataset/map combinations
for dataset in datasets:
    for map_name in map_names:
        folder = base_path / dataset / map_name / "test" / "ours_None" / "pbr"

        if not folder.is_dir():
            print(f"Folder not found: {folder}")
            continue

        json_files = list(folder.glob("*albedo_metrics*"))
        if not json_files:
            print(f"No albedo JSON files in: {folder}")
            continue

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if data.get("albedo_psnr") is not None:
                        global_psnr.append(data["albedo_psnr"])
                    if data.get("albedo_ssim") is not None:
                        global_ssim.append(data["albedo_ssim"])
                    if data.get("albedo_lpips") is not None:
                        global_lpips.append(data["albedo_lpips"])
            except Exception as e:
                print(f"Error reading {json_file}: {e}")

# Compute and print results
if global_psnr and global_ssim and global_lpips:
    psnr_mean, psnr_std = compute_mean_std(global_psnr)
    ssim_mean, ssim_std = compute_mean_std(global_ssim)
    lpips_mean, lpips_std = compute_mean_std(global_lpips)

    print("\n🎯 Global Averages and Standard Deviations Across All Albedo JSONs:")
    print(f"  PSNR   : {psnr_mean:.3f} ± {psnr_std:.3f}")
    print(f"  SSIM   : {ssim_mean:.3f} ± {ssim_std:.3f}")
    print(f"  LPIPS  : {lpips_mean:.3f} ± {lpips_std:.3f}")
else:
    print("\n⚠️ Not enough data to compute global statistics.")
