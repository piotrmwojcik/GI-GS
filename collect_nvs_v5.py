import os
import json
from pathlib import Path

# Datasets and map names
datasets = (
    "hook150_v5_spec32_statictimestep1",
    "jumpingjacks150_v5_spec32_statictimestep75",
    "mouse150_v5_spec32_statictimestep1",
    "spheres_v5_spec32_statictimestep1",
    "standup150_v5_spec32_statictimestep75"
)

map_names = (
    "chapel_day_4k_32x16_rot0",
    "dam_wall_4k_32x16_rot90",
    "golden_bay_4k_32x16_rot330",
)

# Base path
base_path = Path("~/GI-GS/outputs_specular_JULY").expanduser()

# Global accumulators
global_psnr, global_ssim, global_lpips = [], [], []

# Iterate through all dataset/map combinations
for dataset in datasets:
    for map_name in map_names:
        folder = base_path / dataset / map_name / "test" / "ours_None" / "pbr"

        if not folder.is_dir():
            print(f"Folder not found: {folder}")
            continue

        json_files = list(folder.glob("*NVS*.json"))
        if not json_files:
            print(f"No NVS JSON files in: {folder}")
            continue

        for json_file in json_files:
            try:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                    if data.get("psnr_avg") is not None:
                        global_psnr.append(data["psnr_avg"])
                    if data.get("ssim_avg") is not None:
                        global_ssim.append(data["ssim_avg"])
                    if data.get("lpips_avg") is not None:
                        global_lpips.append(data["lpips_avg"])
            except Exception as e:
                print(f"Error reading {json_file}: {e}")

# Compute global averages
if global_psnr and global_ssim and global_lpips:
    avg_psnr = sum(global_psnr) / len(global_psnr)
    avg_ssim = sum(global_ssim) / len(global_ssim)
    avg_lpips = sum(global_lpips) / len(global_lpips)

    print("\n🎯 Global Averages Across All NVS JSONs:")
    print(f"  PSNR   (avg): {avg_psnr:.3f}")
    print(f"  SSIM   (avg): {avg_ssim:.4f}")
    print(f"  LPIPS  (avg): {avg_lpips:.4f}")
else:
    print("\n⚠️ Not enough data to compute global averages.")
