import numpy as np
import rlkit.torch.pytorch_util as ptu
from kornia.geometry.transform import (warp_affine, warp_perspective, get_rotation_matrix2d, get_perspective_transform)

class WarpPerspective:
    def __init__(self, size, warp_pixels=6, num_warps=1000):
        self.size = size
        self.warp_pixels = warp_pixels
        self.num_warps = num_warps
        src = np.array([[[0, 0], [0, size], [size, 0], [size, size]]])
        
        self.warps = []
        for i in range(num_warps):
            dst_jitter = np.random.uniform(-warp_pixels, warp_pixels, size=(1, 4, 2))
            dst = np.clip(src + dst_jitter, 0, 64)
            self.warps.append(get_perspective_transform(ptu.from_numpy(src), ptu.from_numpy(dst)).detach().cpu().numpy()[0])
        self.warps = np.array(self.warps)

    def __call__(self, tensor):
        b, *_ = tensor.size()
        warp_matrix = ptu.from_numpy(self.warps[np.random.randint(0, self.num_warps, size=b)])
        return warp_perspective(tensor, warp_matrix, dsize=(self.size, self.size))



