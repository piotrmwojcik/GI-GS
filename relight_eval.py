import os
from argparse import ArgumentParser

import imageio.v2 as imageio
import numpy as np
import torch
import os
from tqdm import tqdm, trange
from PIL import Image

from utils.image_utils import psnr as get_psnr
from utils.loss_utils import ssim as get_ssim
from lpips import LPIPS


lpips_fn = LPIPS(net="vgg").cuda()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    parser.add_argument("--output_dir", type=str, help="The path to the output directory that stores the relighting results.")
    parser.add_argument("--gt_dir", type=str, help="The path to the output directory that stores the relighting ground truth.")
    args = parser.parse_args()



    psnr_all = 0.0
    ssim_all = 0.0
    lpips_all = 0.0

    data_subdir = os.environ.get("DATA_SUBDIR", "")
    map_name = os.environ.get("MAP_NAME", "")
    dataset = os.environ.get("DATASET", "")

    light_name_list = [map_name]

    for light_name in light_name_list:
        print(f"evaluation {light_name}")
        num_test = 200
        psnr_avg = 0.0
        ssim_avg = 0.0
        lpips_avg = 0.0
        for idx in trange(num_test):
            with torch.no_grad():
                prediction = np.array(Image.open(os.path.join(args.output_dir, f"r_{(10*(idx+1)):04}_{light_name}.png")))[..., :3]  # [H, W, 3]
                prediction = torch.from_numpy(prediction).cuda().permute(2, 0, 1) / 255.0  # [3, H, W]
                print('!!!! ', 'done')
                gt_img = np.array(Image.open(os.path.join(f"/home/pwojcik/GI-GS/data{dataset}/{map_name}/r_{(10*(idx+1)):04}.png")))[..., :3]  # [H, W, 3]
                gt_img = torch.from_numpy(gt_img).cuda().permute(2, 0, 1) / 255.0  # [3, H, W]
                psnr_avg += get_psnr(gt_img, prediction).mean().double()
                ssim_avg += get_ssim(gt_img, prediction).mean().double()
                lpips_avg += lpips_fn(gt_img, prediction).mean().double()

        print(f"{light_name} psnr_avg: {psnr_avg / num_test}")
        print(f"{light_name} ssim_avg: {ssim_avg / num_test}")
        print(f"{light_name} lpips_avg: {lpips_avg / num_test}")

        psnr_all += psnr_avg / num_test
        ssim_all += ssim_avg / num_test
        lpips_all += lpips_avg / num_test
    
    print(f"psnr_avg: {psnr_all / len(light_name_list)}")
    print(f"ssim_avg: {ssim_all / len(light_name_list)}")
    print(f"lpips_avg: {lpips_all / len(light_name_list)}")
