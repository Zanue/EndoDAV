from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Tuple, cast

import imageio.v3 as iio
import numpy as np
import numpy as onp
import numpy.typing as onpt
import skimage.transform
from scipy.spatial.transform import Rotation


class HamlynLoader:
    """Helper for loading frames for Record3D captures."""

    # NOTE(hangg): Consider moving this module into
    # `examples/7_record3d_visualizer.py` since it is usecase-specific.

    def __init__(self, data_dir: Path, depth_dir: Path = None):
        # metadata_path = data_dir / "metadata"

        # # Read metadata.
        # metadata = json.loads(metadata_path.read_text())

        # K: onp.ndarray = np.array(metadata["K"], np.float32).reshape(3, 3).T
        # fps = metadata["fps"]
        K: onp.ndarray = np.array([
            [525., 0., 319.5],
            [0., 525., 239.5],
            [0., 0., 1.]
        ], np.float32)
        fps = 30

        # T_world_cameras: onp.ndarray = np.array(metadata["poses"], np.float32)
        # T_world_cameras = np.concatenate(
        #     [
        #         Rotation.from_quat(T_world_cameras[:, :4]).as_matrix(),
        #         T_world_cameras[:, 4:, None],
        #     ],
        #     -1,
        # )
        # T_world_cameras = (T_world_cameras @ np.diag([1, -1, -1, 1])).astype(np.float32)
        T_world_cameras = np.array([
            [1., 0., 0., 0.],
            [0., -1., 0., 0.],
            [0., 0., -1., 0.],
        ]).astype(np.float32)

        self.K = K
        self.fps = fps
        self.T_world_cameras = T_world_cameras

        self.is_pred = (depth_dir is not None)
        rgb_dir = data_dir / "image01"
        if depth_dir is None:
            depth_dir = data_dir / "depth01"
        self.rgb_paths = sorted(rgb_dir.glob("*.jpg"), key=lambda p: int(p.stem))
        self.depth_paths = sorted(depth_dir.glob("*.png"), key=lambda p: int(p.stem))
        # print('depth len: ', len(self.depth_paths))
        # self.conf_paths = [rgb_path.with_suffix(".conf") for rgb_path in self.rgb_paths]

    def num_frames(self) -> int:
        return len(self.rgb_paths)

    def get_frame(self, index: int):
        # Read conf.
        # conf: onp.ndarray = np.frombuffer(
        #     liblzfse.decompress(self.conf_paths[index].read_bytes()), dtype=np.uint8
        # )
        # if conf.shape[0] == 640 * 480:
        #     conf = conf.reshape((640, 480))  # For a FaceID camera 3D Video
        # elif conf.shape[0] == 256 * 192:
        #     conf = conf.reshape((256, 192))  # For a LiDAR 3D Video
        # else:
        #     assert False, f"Unexpected conf shape {conf.shape}"

        # Read depth.
        # depth: onp.ndarray = np.frombuffer(
        #     liblzfse.decompress(self.depth_paths[index].read_bytes()), dtype=np.float32
        # ).copy()
        # if depth.shape[0] == 640 * 480:
        #     depth = depth.reshape((640, 480))  # For a FaceID camera 3D Video
        # elif depth.shape[0] == 256 * 192:
        #     depth = depth.reshape((256, 192))  # For a LiDAR 3D Video
        # else:
        #     assert False, f"Unexpected depth shape {depth.shape}"
        depth = iio.imread(self.depth_paths[index]).astype(np.float32)
        if self.is_pred:
            depth = depth / 200.
        else:
            depth = depth / 50.

        # Read RGB.
        rgb = iio.imread(self.rgb_paths[index])
        return HamlynFrame(
            K=self.K,
            rgb=rgb,
            depth=depth,
            mask=np.ones(rgb.shape[:2]).astype(bool),
            T_world_camera=self.T_world_cameras,
        )


@dataclasses.dataclass
class HamlynFrame:
    """A single frame from a Record3D capture."""

    K: onpt.NDArray[onp.float32]
    rgb: onpt.NDArray[onp.uint8]
    depth: onpt.NDArray[onp.float32]
    mask: onpt.NDArray[onp.bool_]
    T_world_camera: onpt.NDArray[onp.float32]

    def get_point_cloud(
        self, downsample_factor: int = 1
    ) -> Tuple[onpt.NDArray[onp.float32], onpt.NDArray[onp.uint8]]:
        rgb = self.rgb[::downsample_factor, ::downsample_factor]
        depth = skimage.transform.resize(self.depth, rgb.shape[:2], order=0)
        mask = cast(
            onpt.NDArray[onp.bool_],
            skimage.transform.resize(self.mask, rgb.shape[:2], order=0),
        ).astype(bool)
        assert depth.shape == rgb.shape[:2]

        K = self.K
        T_world_camera = self.T_world_camera

        img_wh = rgb.shape[:2][::-1]

        grid = (
            np.stack(np.meshgrid(np.arange(img_wh[0]), np.arange(img_wh[1])), 2) + 0.5
        )
        grid = grid * downsample_factor

        homo_grid = np.pad(grid[mask], np.array([[0, 0], [0, 1]]), constant_values=1)
        local_dirs = np.einsum("ij,bj->bi", np.linalg.inv(K), homo_grid)
        dirs = np.einsum("ij,bj->bi", T_world_camera[:3, :3], local_dirs)
        points = (T_world_camera[:, -1] + dirs * depth[mask, None]).astype(np.float32)
        point_colors = rgb[mask]

        return points, point_colors
