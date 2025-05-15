import os
import json
import math

# Datasets
datasets = [
    "hook150_v3_transl_statictimestep1",
    "jumpingjacks150_v3_tex_statictimestep75",
    "mouse150_v2_transl_statictimestep1",
    "spheres_cube_dataset_v5_statictimestep1",
    "standup150_v3_statictimestep75",
]

# Environment name pairs (source -> target)
pairs = [
    # ("chapel_day_4k_32x16_rot0", "golden_bay_4k_32x16_rot330"),
    ("dam_wall_4k_32x16_rot90", "small_harbour_sunset_4k_32x16_rot270"),
    # ("golden_bay_4k_32x16_rot330", "dam_wall_4k_32x16_rot90"),
]

# Base directory for JSON files
base_dir = "relight"

# Metric value lists
psnr_values = []
ssim_values = []
lpips_values = []

# Iterate over datasets and pairs
for dataset in datasets:
    for from_env, to_env in pairs:
        json_path = os.path.join(
            base_dir,
            dataset,
            f"relight_FROM_{from_env}",
            f"relight_TO_{to_env}",
            f"{to_env}.json"
        )

        if not os.path.isfile(json_path):
            print(f"Missing: {json_path}")
            continue

        try:
            with open(json_path, "r") as f:
                data = json.load(f)
                psnr_values.append(data.get("psnr_avg", 0.0))
                ssim_values.append(data.get("ssim_avg", 0.0))
                lpips_values.append(data.get("lpips_avg", 0.0))
        except Exception as e:
            print(f"Error reading {json_path}: {e}")

# Function to compute mean and std
def compute_mean_std(values):
    if not values:
        return 0.0, 0.0
    mean = sum(values) / len(values)
    std = math.sqrt(sum((x - mean) ** 2 for x in values) / len(values))
    return mean, std

# Compute and display results
if psnr_values and ssim_values and lpips_values:
    psnr_mean, psnr_std = compute_mean_std(psnr_values)
    ssim_mean, ssim_std = compute_mean_std(ssim_values)
    lpips_mean, lpips_std = compute_mean_std(lpips_values)

    print("\n✅ Global Averages and Standard Deviations Across All JSONs:")
    print(f"psnr_avg:  {psnr_mean:.3f} ± {psnr_std:.3f}")
    print(f"ssim_avg:  {ssim_mean:.4f} ± {ssim_std:.4f}")
    print(f"lpips_avg: {lpips_mean:.4f} ± {lpips_std:.4f}")
else:
    print("❌ No valid JSON files found.")
