"""
Render a custom camera trajectory using a trained NeRF model.

Usage:
    python render_custom.py --config configs/lego.txt
"""
import os
import sys
import imageio
import numpy as np
import torch

from model_helpers import *
from data_loader.dataset import Data
from data_loader.load_blender import pose_spherical
from model import Model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def custom_render_poses():
    """
    Define your custom camera trajectory here.
    Modify this function to create any camera path you want.
    
    pose_spherical(theta, phi, radius):
        theta  - azimuth angle (horizontal rotation, degrees)
        phi    - elevation angle (degrees, negative = looking down)
        radius - distance from origin
    """
    N = 120

    # === Option 1: Top-down sweep (fixed azimuth, vary elevation) ===
    # render_poses = torch.stack([
    #     pose_spherical(0, phi, 4.0)
    #     for phi in np.linspace(-30, -90, N)
    # ], 0)

    # === Option 2: Zoom-in spiral ===
    # render_poses = torch.stack([
    #     pose_spherical(angle, -30.0, radius)
    #     for angle, radius in zip(np.linspace(-180, 180, N), np.linspace(5.0, 2.5, N))
    # ], 0)

    # === Option 3: Full spiral (orbit + elevation oscillation + zoom) ===
    render_poses = torch.stack([
        pose_spherical(
            angle,
            -30 + 20 * np.sin(2 * np.pi * i / N),
            4.0 - 1.0 * np.sin(2 * np.pi * i / N)
        )
        for i, angle in enumerate(np.linspace(-180, 180, N + 1)[:-1])
    ], 0)

    return render_poses


@torch.no_grad()
def main():
    # Parse args from config (reuse existing config parser)
    sys.argv += ['--render_only']  # needed so checkpoint loads properly
    from main import config_parser
    parser = config_parser()
    args = parser.parse_args()

    # Load dataset (for hwf, K, near, far)
    dataset = Data(args)

    basedir = args.basedir
    expname = args.expname

    # Load trained model
    model = Model(args, device=DEVICE)

    # Get custom camera poses
    render_poses = custom_render_poses().to(DEVICE)
    print(f'Custom render poses shape: {render_poses.shape}')

    # Output directory
    savedir = os.path.join(basedir, expname, f'custom_render_{model.start:06d}')
    os.makedirs(savedir, exist_ok=True)

    # Render
    model.renderer.eval()
    rgbs, disps = model.renderer.render_path(
        render_poses, dataset.hwf, dataset.K, model.chunk,
        model.nerf, model.nerf_fine,
        near=dataset.near, far=dataset.far,
        savedir=savedir, render_factor=args.render_factor
    )
    model.renderer.train()

    # Save video
    print(f'Done rendering, saving to {savedir}')
    to8b = lambda x: (255 * np.clip(x, 0, 1)).astype(np.uint8)
    imageio.mimwrite(os.path.join(savedir, 'custom_rgb.mp4'), to8b(rgbs), fps=30, quality=8)
    imageio.mimwrite(os.path.join(savedir, 'custom_disp.mp4'), to8b(disps / np.max(disps)), fps=30, quality=8)
    print(f'Videos saved to {savedir}/custom_rgb.mp4')


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    main()
