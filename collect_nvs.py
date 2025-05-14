import os
import json
from PIL import Image

# Define input-output folder name suffixes (base path is shared)
base_input_dir = "/home/pwojcik/IRGS/outputs/irgs_images_710_780_single_ts"
base_output_dir = "/home/pwojcik/IRGS/outputs/irgs_images_710_780_single_ts"

# Folder name suffixes to process
folder_suffixes = [
    "test_lg0_rli_chapel_day_4k_32x16_rot0_light",
    "test_lg0_rli_test_env_j6_i24_light",
    "test_lg0_rli_golden_bay_4k_32x16_rot330_light",
]

# File extensions
image_extensions = ('.png',)
json_extension = ".json"

# Process each folder
for suffix in folder_suffixes:
    input_dir = os.path.join(base_input_dir, suffix)
    output_dir = os.path.join(base_output_dir, f"{suffix}_for_paper")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nProcessing folder: {input_dir}")

    # Initialize metric accumulators
    psnr_vals, ssim_vals, lpips_vals = [], [], []

    for root, _, files in os.walk(input_dir):
        for filename in files:
            file_path = os.path.join(root, filename)

            # --- Image Processing ---
            if filename.lower().endswith(image_extensions):
                try:
                    with Image.open(file_path) as img:
                        if img.width > 500 or img.height > 300:
                            new_size = (img.width // 4, img.height // 4)
                            img = img.resize(new_size, Image.LANCZOS)

                        left, top = 115, 25
                        right = img.width - 85
                        bottom = img.height - 5

                        if right > left and bottom > top:
                            img = img.crop((left, top, right, bottom))
                        else:
                            print(f"Skipping crop: {file_path} — crop area too small.")
                            continue

                        rel_path = os.path.relpath(file_path, input_dir)
                        flat_name = rel_path.replace(os.sep, "_")
                        save_path = os.path.join(output_dir, flat_name)
                        img.save(save_path)
                        print(f"Saved image: {save_path}")
                except Exception as e:
                    print(f"Error processing image {file_path}: {e}")

            # --- JSON Metric Accumulation ---
            elif filename.lower().endswith(json_extension):
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        psnr_vals.append(data.get('psnr_avg'))
                        ssim_vals.append(data.get('ssim_avg'))
                        lpips_vals.append(data.get('lpips_avg'))
                except Exception as e:
                    print(f"Error reading JSON {file_path}: {e}")

    # Compute and report averages if any JSONs were found
    if psnr_vals and ssim_vals and lpips_vals:
        avg_psnr = sum(psnr_vals) / len(psnr_vals)
        avg_ssim = sum(ssim_vals) / len(ssim_vals)
        avg_lpips = sum(lpips_vals) / len(lpips_vals)
        print(f"\nMetrics for {suffix}:")
        print(f"  Average PSNR  : {avg_psnr:.3f}")
        print(f"  Average SSIM  : {avg_ssim:.4f}")
        print(f"  Average LPIPS : {avg_lpips:.4f}")
    else:
        print(f"\nNo JSON metrics found in {input_dir}.")

print("\nAll folders processed.")
