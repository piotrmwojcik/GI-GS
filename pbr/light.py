from typing import List, Optional

import cv2
import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn as nn
import torch.nn.functional as F

from .renderutils import diffuse_cubemap, specular_cubemap


def resizeImage(img, width, height, interpolation=cv2.INTER_CUBIC):
    if img.shape[1] < width:  # up res
        if interpolation == 'max_pooling':
            return cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
        else:
            return cv2.resize(img, (width, height), interpolation=interpolation)
    if interpolation == 'max_pooling':  # down res, max pooling
        try:
            import skimage.measure
            scaleFactor = int(float(img.shape[1]) / width)
            factoredWidth = width * scaleFactor
            img = cv2.resize(img, (factoredWidth, int(factoredWidth / 2)), interpolation=cv2.INTER_CUBIC)
            blockSize = scaleFactor
            r = skimage.measure.block_reduce(img[:, :, 0], (blockSize, blockSize), np.max)
            g = skimage.measure.block_reduce(img[:, :, 1], (blockSize, blockSize), np.max)
            b = skimage.measure.block_reduce(img[:, :, 2], (blockSize, blockSize), np.max)
            img = np.dstack((np.dstack((r, g)), b)).astype(np.float32)
            return img
        except:
            print("Failed to do max_pooling, using default")
            return cv2.resize(img, (width, height), interpolation=cv2.INTER_CUBIC)
    else:  # down res, using interpolation
        return cv2.resize(img, (width, height), interpolation=interpolation)


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


class cubemap_mip(torch.autograd.Function):
    @staticmethod
    def forward(ctx, cubemap: torch.Tensor) -> torch.Tensor:
        # avg_pool_nhwc
        y = cubemap.permute(0, 3, 1, 2)  # NHWC -> NCHW
        y = torch.nn.functional.avg_pool2d(y, (2, 2))
        return y.permute(0, 2, 3, 1).contiguous()  # NCHW -> NHWC

    @staticmethod
    def backward(ctx, dout: torch.Tensor) -> torch.Tensor:
        res = dout.shape[1] * 2
        out = torch.zeros(6, res, res, dout.shape[-1], dtype=torch.float32, device="cuda")
        for s in range(6):
            gy, gx = torch.meshgrid(
                torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
                torch.linspace(-1.0 + 1.0 / res, 1.0 - 1.0 / res, res, device="cuda"),
                indexing="ij",
            )
            v = F.normalize(cube_to_dir(s, gx, gy), p=2, dim=-1)
            out[s, ...] = dr.texture(
                dout[None, ...] * 0.25,
                v[None, ...].contiguous(),
                filter_mode="linear",
                boundary_mode="cube",
            )
        return out




class CubemapLight(nn.Module):
    # for nvdiffrec
    LIGHT_MIN_RES = 16

    MIN_ROUGHNESS = 0.08
    MAX_ROUGHNESS = 0.5

    def __init__(
        self,
        base_res: int = 16,
        scale: float = 0.5,
        bias: float = 0.25,
        path = ''
    ) -> None:
        super(CubemapLight, self).__init__()
        self.mtx = None
        self.path = path
        self.gt_envmap = self.load_envmap(path, 1024)
        base = (
            torch.rand(6, base_res, base_res, 3, dtype=torch.float32, device="cuda") * scale + bias
        )
        # print(torch.isnan(base).any())
        self.base = nn.Parameter(base)
        self.register_parameter("env_base", self.base)

    def load_envmap(self, path, resizeWidth=None):
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if resizeWidth:
            image = resizeImage(image, resizeWidth, int(resizeWidth/2), cv2.INTER_AREA)
        env_map_torch = torch.tensor(image, dtype=torch.float32, device='cuda', requires_grad=False)
        env_map_torch = env_map_torch.permute(2, 0, 1).unsqueeze(0) # [1, 3, H, W]
        return env_map_torch

    def split_envmap_loss(self):
        envmap_train = self.export_envmap(return_img=True)

        H = envmap_train.shape[2]  # Get height, [1, C, H, W]
        H_half = H // 2  # Midpoint for splitting

        upper_loss = F.mse_loss(envmap_train[:, :H_half, :], self.gt_envmap[:, :H_half, :]).item()
        lower_loss = F.mse_loss(envmap_train[:, H_half:, :], self.gt_envmap[:, :H_half, :]).item()

        return upper_loss, lower_loss

    def xfm(self, mtx) -> None:
        self.mtx = mtx

    def clamp_(self, min: Optional[float]=None, max: Optional[float]=None) -> None:
        self.base.clamp_(min, max)

    def get_mip(self, roughness: torch.Tensor) -> torch.Tensor:
        return torch.where(
            roughness < self.MAX_ROUGHNESS,
            (torch.clamp(roughness, self.MIN_ROUGHNESS, self.MAX_ROUGHNESS) - self.MIN_ROUGHNESS)
            / (self.MAX_ROUGHNESS - self.MIN_ROUGHNESS)
            * (len(self.specular) - 2),
            (torch.clamp(roughness, self.MAX_ROUGHNESS, 1.0) - self.MAX_ROUGHNESS)
            / (1.0 - self.MAX_ROUGHNESS)
            + len(self.specular)
            - 2,
        )

    def build_mips(self, cutoff: float = 0.99) -> None:
        self.specular = [self.base]
        while self.specular[-1].shape[1] > self.LIGHT_MIN_RES:
            #print('add specular ', self.specular[-1])
            self.specular += [cubemap_mip.apply(self.specular[-1])]

        #print('self specular: ', len(self.specular), self.specular)

        self.diffuse = diffuse_cubemap(self.specular[-1])

        for idx in range(len(self.specular) - 1):
            roughness = (idx / (len(self.specular) - 2)) * (
                self.MAX_ROUGHNESS - self.MIN_ROUGHNESS
            ) + self.MIN_ROUGHNESS
            self.specular[idx] = specular_cubemap(self.specular[idx], roughness, cutoff)
        self.specular[-1] = specular_cubemap(self.specular[-1], 1.0, cutoff)

    def export_envmap(
        self,
        filename: Optional[str] = None,
        res: List[int] = [512, 1024],
        return_img: bool = False,
    ) -> Optional[torch.Tensor]:
        # cubemap_to_latlong
        #gy, gx = torch.meshgrid(
        #    torch.linspace(0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device="cuda"),
        #    torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device="cuda"),
        #    indexing="ij",
        #)
        # cubemap_to_latlong
        gy, gx = torch.meshgrid(
            torch.linspace(0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device="cuda"),
            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device="cuda"),
            indexing="ij",
        )

        sintheta, costheta = torch.sin(gy * np.pi), torch.cos(gy * np.pi)
        sinphi, cosphi = torch.sin(gx * np.pi), torch.cos(gx * np.pi)

        reflvec = torch.stack(
            (sintheta * sinphi, costheta, -sintheta * cosphi), dim=-1
        )  # [H, W, 3]
        color = dr.texture(
            self.base[None, ...],
            reflvec[None, ...].contiguous(),
            filter_mode="linear",
            boundary_mode="cube",
        )[
            0
        ]  # [H, W, 3]
        if return_img:
            return color
        else:
            cv2.imwrite(filename, color.clamp(min=0.0).cpu().numpy()[..., ::-1])