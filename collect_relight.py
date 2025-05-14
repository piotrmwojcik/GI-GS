import os
import json

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



# Iterate over datasets and pairs
for dataset in datasets:
    # Accumulators
    psnr_total = 0.0
    ssim_total = 0.0
    lpips_total = 0.0
    count = 0
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
                psnr_total += data.get("psnr_avg", 0.0)
                ssim_total += data.get("ssim_avg", 0.0)
                lpips_total += data.get("lpips_avg", 0.0)
                count += 1
        except Exception as e:
            print(f"Error reading {json_path}: {e}")

    # Compute global averages
    if count > 0:
        print("\n✅ Global Averages Across All JSONs:")
        print(f"psnr_avg:  {psnr_total / count:.4f}")
        print(f"ssim_avg:  {ssim_total / count:.4f}")
        print(f"lpips_avg: {lpips_total / count:.4f}")
    else:
        print("❌ No valid JSON files found.")
