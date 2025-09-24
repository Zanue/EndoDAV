"""Record3D visualizer

Parse and stream record3d captures. To get the demo data, see `./assets/download_record3d_dance.sh`.
"""

import time
from pathlib import Path
import argparse

import numpy as onp
import tyro
from tqdm.auto import tqdm

import viser
# import viser.extras
import viser.transforms as tf
from utils import HamlynLoader, SCAREDLoader, EndoNeRFLoader


def main(
    data_path: Path = Path(__file__).parent,
    data_type: str = "scared",
    depth_path: Path = None,
    downsample_factor: int = 8,
    max_frames: int = 100,
    share: bool = False,
) -> None:
    server = viser.ViserServer()
    if share:
        server.request_share_url()

    print("Loading frames!")
    if data_type == "scared":
        loader = SCAREDLoader(data_path, depth_path)
    elif data_type == "hamlyn":
        loader = HamlynLoader(data_path, depth_path)
    elif data_type == "endonerf":
        loader = EndoNeRFLoader(data_path, depth_path)
    num_frames = min(max_frames, loader.num_frames())

    # Add playback UI.
    with server.gui.add_folder("Playback"):
        gui_timestep = server.gui.add_slider(
            "Timestep",
            min=0,
            max=num_frames - 1,
            step=1,
            initial_value=0,
            disabled=True,
        )
        gui_next_frame = server.gui.add_button("Next Frame", disabled=True)
        gui_prev_frame = server.gui.add_button("Prev Frame", disabled=True)
        gui_playing = server.gui.add_checkbox("Playing", True)
        gui_framerate = server.gui.add_slider(
            "FPS", min=1, max=60, step=0.1, initial_value=loader.fps
        )
        gui_framerate_options = server.gui.add_button_group(
            "FPS options", ("10", "20", "30", "60")
        )

    # Frame step buttons.
    @gui_next_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value + 1) % num_frames

    @gui_prev_frame.on_click
    def _(_) -> None:
        gui_timestep.value = (gui_timestep.value - 1) % num_frames

    # Disable frame controls when we're playing.
    @gui_playing.on_update
    def _(_) -> None:
        gui_timestep.disabled = gui_playing.value
        gui_next_frame.disabled = gui_playing.value
        gui_prev_frame.disabled = gui_playing.value

    # Set the framerate when we click one of the options.
    @gui_framerate_options.on_click
    def _(_) -> None:
        gui_framerate.value = int(gui_framerate_options.value)

    prev_timestep = gui_timestep.value

    # Toggle frame visibility when the timestep slider changes.
    @gui_timestep.on_update
    def _(_) -> None:
        nonlocal prev_timestep
        current_timestep = gui_timestep.value
        with server.atomic():
            frame_nodes[current_timestep].visible = True
            frame_nodes[prev_timestep].visible = False
        prev_timestep = current_timestep
        server.flush()  # Optional!

    # Load in frames.
    server.scene.add_frame(
        "/frames",
        wxyz=tf.SO3.exp(onp.array([onp.pi / 2.0, 0.0, 0.0])).wxyz,
        position=(0, 0, 0),
        show_axes=False,
    )
    frame_nodes: list[viser.FrameHandle] = []
    for i in tqdm(range(num_frames)):
        frame = loader.get_frame(i)
        position, color = frame.get_point_cloud(downsample_factor)

        # Add base frame.
        frame_nodes.append(server.scene.add_frame(f"/frames/t{i}", show_axes=False))

        # Place the point cloud in the frame.
        server.scene.add_point_cloud(
            name=f"/frames/t{i}/point_cloud",
            points=position,
            colors=color,
            point_size=0.01,
            point_shape="rounded",
        )

        # Place the frustum.
        fov = 2 * onp.arctan2(frame.rgb.shape[0] / 2, frame.K[0, 0])
        aspect = frame.rgb.shape[1] / frame.rgb.shape[0]
        server.scene.add_camera_frustum(
            f"/frames/t{i}/frustum",
            fov=fov,
            aspect=aspect,
            scale=0.15,
            image=frame.rgb[::downsample_factor, ::downsample_factor],
            wxyz=tf.SO3.from_matrix(frame.T_world_camera[:3, :3]).wxyz,
            position=frame.T_world_camera[:3, 3],
        )

        # Add some axes.
        server.scene.add_frame(
            f"/frames/t{i}/frustum/axes",
            axes_length=0.05,
            axes_radius=0.005,
        )

    # Hide all but the current frame.
    for i, frame_node in enumerate(frame_nodes):
        frame_node.visible = i == gui_timestep.value

    # Playback update loop.
    prev_timestep = gui_timestep.value
    while True:
        if gui_playing.value:
            gui_timestep.value = (gui_timestep.value + 1) % num_frames

        time.sleep(1.0 / gui_framerate.value)


if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser(description="Process input arguments.")
    hamlyn_path = '/data_hdd2/users/zhouzanwei/data/Medical/hamlyn-EDM/hamlyn/rectified01'
    SCARED_path = '/data_hdd2/users/zhouzanwei/data/Medical/SCARED/scared/train/dataset3/keyframe2'
    EndoNeRF_path = '/data_hdd2/users/zhouzanwei/data/Medical/endonerf/cutting_tissues_twice'

    # Define arguments
    parser.add_argument(
        "--data_path",
        type=Path,
        nargs="?",
        default=Path(EndoNeRF_path),
        help="Path to the data"
    )
    parser.add_argument(
        "--data_type",
        type=str,
        choices=["hamlyn", "scared", "endonerf"],
        default="scared",
        help="type of dataset"
    )
    parser.add_argument(
        "--depth_path",
        type=Path,
        nargs="?",
        default=None,
        help="Path to the data"
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=100,
        help="Max frames to visualize"
    )

    # Parse arguments
    args = parser.parse_args()

    tyro.cli(main(
        data_path=args.data_path,
        data_type=args.data_type,
        depth_path=args.depth_path,
        max_frames=args.max_frames))
