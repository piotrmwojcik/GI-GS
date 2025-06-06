import os
from argparse import ArgumentParser
from os import makedirs
from typing import Dict, List, Tuple

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import kornia
from PIL import Image
from tqdm import tqdm
from typing import Dict, Optional, Union
from diff_gaussian_rasterization import Gaussian_SSR
import math
import json

from arguments import GroupParams, ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel, render
from pbr import CubemapLight, get_brdf_lut, pbr_shading
from scene import Scene
from utils.general_utils import safe_state
from utils.image_utils import viridis_cmap
from lpips import LPIPS
import pyexr
import nvdiffrast.torch as dr



def read_hdr(path: str) -> np.ndarray:
    """Reads an HDR map from disk.

    Args:
        path (str): Path to the .hdr file.

    Returns:
        numpy.ndarray: Loaded (float) HDR map with RGB channels in order.
    """
    with open(path, "rb") as h:
        buffer_ = np.frombuffer(h.read(), np.uint8)
    bgr = cv2.imdecode(buffer_, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def linear_to_srgb(linear: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(linear, torch.Tensor):
        """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        eps = torch.finfo(torch.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * torch.clamp(linear, min=eps) ** (5 / 12) - 11) / 200
        return torch.where(linear <= 0.0031308, srgb0, srgb1)
    elif isinstance(linear, np.ndarray):
        eps = np.finfo(np.float32).eps
        srgb0 = 323 / 25 * linear
        srgb1 = (211 * np.maximum(eps, linear) ** (5 / 12) - 11) / 200
        return np.where(linear <= 0.0031308, srgb0, srgb1)
    else:
        raise NotImplementedError
    
def srgb_to_linear(srgb: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(srgb, torch.Tensor):
        """Assumes `linear` is in [0, 1], see https://en.wikipedia.org/wiki/SRGB."""
        linear0 = 25 / 323 * srgb
        linear1 = ((srgb + 0.055) / 1.055)**2.4
        return torch.where(srgb <= 0.04045, linear0, linear1)
    elif isinstance(srgb, np.ndarray):
        linear0 = 25 / 323 * srgb
        linear1 = ((srgb + 0.055) / 1.055)**2.4
        return np.where(srgb <= 0.04045, linear0, linear1)
    else:
        raise NotImplementedError


def cube_to_dir(s: int, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if s == 0:
        rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1:
        rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2:
        rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3:
        rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4:
        rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5:
        rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)


def latlong_to_cubemap(latlong_map: torch.Tensor, res: List[int]) -> torch.Tensor:
    cubemap = torch.zeros(
        6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device="cuda"
    )
    for s in range(6):
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device="cuda"),
            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device="cuda"),
            indexing="ij",
        )
        v = F.normalize(cube_to_dir(s, gx, gy), p=2, dim=-1)

        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)

        cubemap[s, ...] = dr.texture(
            latlong_map[None, ...], texcoord[None, ...], filter_mode="linear"
        )[0]
    return cubemap


def render_set(
    model_path: str,
    name: str,
    light_name: str,
    scene: Scene,
    hdri: torch.Tensor,
    light: CubemapLight,
    pipeline: GroupParams,
    metallic: bool = False,
    tone: bool = False,
    gamma: bool = False,
    radius: float = 0.8,
    bias: float = 0.01,
    thick: float = 0.05,
    delta: float = 0.0625,
    step: int = 16,
    start: int = 8
) -> None:
    iteration = scene.loaded_iter
    if name == "train":
        views = scene.getTrainCameras()
    elif name == "test":
        views = scene.getTestCameras()
    else:
        raise ValueError

    # build mip for environment light
    light.build_mips()
    envmap = light.export_envmap(return_img=True).permute(2, 0, 1).clamp(min=0.0, max=1.0)
    os.makedirs(os.path.join(model_path, name), exist_ok=True)
    envmap_path = os.path.join(model_path, name, "envmap_relight.png")
    torchvision.utils.save_image(envmap, envmap_path)

    relight_path = os.path.join(model_path, name, f"ours_{iteration}", "relight")
    makedirs(relight_path, exist_ok=True)

    brdf_lut = get_brdf_lut().cuda()
    canonical_rays = scene.get_canonical_rays()

    background = torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        background[...] = 0.0  # NOTE: set zero
        rendering_result = render(
            viewpoint_camera=view,
            pc=scene.gaussians,
            pipe=pipeline,
            bg_color=background,
            inference=True,
            pad_normal=False,
            derive_normal=True,
            radius=radius,
            bias=bias,
            thick=thick,
            delta=delta,
            step=step,
            start=start
        )

        tanfovx = math.tan(view.FoVx * 0.5)
        tanfovy = math.tan(view.FoVy * 0.5)
        image_height=int(view.image_height)
        image_width=int(view.image_width)

        depth_map = rendering_result["depth_map"]
        depth_img = viridis_cmap(depth_map.squeeze().cpu().numpy())
        depth_img = (depth_img * 255).astype(np.uint8)
        normal_map = rendering_result["normal_map"]
        normal_mask = rendering_result["normal_mask"]
        occlusion = rendering_result["occlusion_map"]
        #print('!!! ', occlusion.shape)
        torchvision.utils.save_image(
            occlusion, os.path.join(relight_path, f"{view.image_name}_{light_name}_occlusion.png")
        )

        # normal from point cloud
        H, W = view.image_height, view.image_width
        c2w = torch.inverse(view.world_view_transform.T)  # [4, 4]
        view_dirs = -(
            (F.normalize(canonical_rays[:, None, :], p=2, dim=-1) * c2w[None, :3, :3])  # [HW, 3, 3]
            .sum(dim=-1)
            .reshape(H, W, 3)
        )  # [H, W, 3]
        alpha_mask = view.gt_alpha_mask.cuda()
        print('!!! alpha_mask ', alpha_mask.shape, (alpha_mask > 0).sum())

        albedo_map = rendering_result["albedo_map"]  # [3, H, W]
        roughness_map = rendering_result["roughness_map"]  # [1, H, W]
        metallic_map = rendering_result["metallic_map"]  # [1, H, W]
        #occlusion = torch.ones_like(roughness_map).permute(1, 2, 0)
        pbr_dir = os.path.join(model_path, 'test', f"ours_{iteration}", "pbr")
        json_path = os.path.join(pbr_dir, "albedo_ratio.json")

        with open(json_path, 'r') as f:
            data_3ch = json.load(f)

        # Convert list back to tensor
        three_channel_ratio = torch.tensor(data_3ch["three_channel_ratio"]).cuda()
        print(three_channel_ratio, three_channel_ratio.shape, albedo_map.shape)



        pbr_result = pbr_shading(
            light=light,
            normals=normal_map.permute(1, 2, 0),  # [H, W, 3]
            view_dirs=view_dirs,
            mask=normal_mask.permute(1, 2, 0),  # [H, W, 1]
            albedo=(albedo_map*three_channel_ratio[:, None, None]).permute(1, 2, 0),  # [H, W, 3]
            roughness=roughness_map.permute(1, 2, 0),  # [H, W, 1]
            metallic=metallic_map.permute(1, 2, 0) if metallic else None,  # [H, W, 1]
            tone=tone,
            occlusion=occlusion.permute(1, 2, 0),
            gamma=gamma,
            brdf_lut=brdf_lut
        )
        render_direct = pbr_result["render_rgb"].permute(2, 0, 1)  # [3, H, W]
        render_direct = torch.where(
            normal_mask,
            render_direct,
            background[:, None, None])
        out_normal_view = rendering_result["out_normal_view"]
        depth_pos = rendering_result["depth_pos"]
        SSR = Gaussian_SSR(tanfovx, tanfovy, image_width, image_height, radius, bias, thick, delta, step, start)
        if metallic:
            F0 = torch.ones_like(albedo_map) * 0.04  # [1, H, W, 3]
            metallic_map = torch.zeros_like(roughness_map)
        else:
            F0 = (1.0 - metallic) * 0.04 + albedo_map * metallic_map
        linear_rgb = srgb_to_linear(render_direct)    

        (IRR, _) = SSR(out_normal_view, depth_pos, linear_rgb, albedo_map, roughness_map, metallic_map, F0)
        IRR = linear_to_srgb(IRR)
        IRR = kornia.filters.median_blur(IRR[None, ...], (3, 3))[0]
        render_rgb = render_direct + IRR
        render_rgb = render_rgb * alpha_mask

        torchvision.utils.save_image(
            render_rgb, os.path.join(relight_path, f"{view.image_name}_{light_name}.png")
        )


@torch.no_grad()
def launch(
    model_path: str,
    checkpoint: str,
    hdri_path: str,
    dataset: GroupParams,
    pipeline: GroupParams,
    skip_train: bool,
    skip_test: bool,
    metallic: bool = False,
    tone: bool = False,
    gamma: bool = False,
    radius: float = 0.8,
    bias: float = 0.01,
    thick: float = 0.05,
    delta: float = 0.0625,
    step: int = 16,
    start: int = 8
) -> None:
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, shuffle=False)

    # load hdri
    print(f"read hdri from {hdri_path}")
    hdri = read_hdr(hdri_path)
    hdri = torch.from_numpy(hdri).cuda()
    res = 256

    cubemap = CubemapLight(base_res=res, path=hdri_path).cuda()
    cubemap.base.data = latlong_to_cubemap(hdri, [res, res])
    cubemap.eval()

    light_name = os.path.basename(hdri_path).split(".")[0]

    checkpoint = torch.load(checkpoint)
    if isinstance(checkpoint, Tuple):
        model_params = checkpoint[0]
    elif isinstance(checkpoint, Dict):
        model_params = checkpoint["gaussians"]
    else:
        raise TypeError
    gaussians.restore(model_params)

    if not skip_train:
        render_set(
            model_path=model_path,
            name="train",
            light_name=light_name,
            scene=scene,
            hdri=hdri,
            light=cubemap,
            metallic=metallic,
            tone=tone,
            gamma=gamma,
            pipeline=pipeline,
            radius=radius,
            bias=bias,
            thick=thick,
            delta=delta,
            step=step,
            start=start
        )
    if not skip_test:
        render_set(
            model_path=model_path,
            name="test",
            light_name=light_name,
            scene=scene,
            hdri=hdri,
            light=cubemap,
            metallic=metallic,
            tone=tone,
            gamma=gamma,
            pipeline=pipeline,
            radius=radius,
            bias=bias,
            thick=thick,
            delta=delta,
            step=step,
            start=start
        )


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--hdri", type=str, default=None, help="The path to the hdri for relighting.")
    parser.add_argument("--checkpoint", type=str, default=None, help="The path to the checkpoint to load.")
    parser.add_argument("--tone", action="store_true", help="Enable aces film tone mapping.")
    parser.add_argument("--gamma", action="store_true", help="Enable linear_to_sRGB for gamma correction.")
    parser.add_argument("--radius", default=0.8, type=float, help="Path tracing range")
    parser.add_argument("--bias", default=0.01, type=float, help="ensure hit the surface")
    parser.add_argument("--thick", default=0.05, type=float, help="thickness of the surface")
    parser.add_argument("--delta", default=0.0625, type=float, help="angle interval to control the num-sample")
    parser.add_argument("--step", default=16, type=int, help="Path tracing steps")
    parser.add_argument("--start", default=8, type=int, help="Path tracing starting point")
    parser.add_argument("--metallic", action="store_true", help="Enable metallic material reconstruction.")
    args = get_combined_args(parser)

    model_path = os.path.dirname(args.checkpoint)
    print("Rendering " + model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    launch(
        model_path=model_path,
        checkpoint=args.checkpoint,
        hdri_path=args.hdri,
        dataset=model.extract(args),
        pipeline=pipeline.extract(args),
        skip_train=args.skip_train,
        skip_test=args.skip_test,
        metallic=args.metallic,
        tone=args.tone,
        gamma=args.gamma,
        radius=args.radius,
        bias=args.bias,
        thick=args.thick,
        delta=args.delta,
        step=args.step,
        start=args.start
    )


