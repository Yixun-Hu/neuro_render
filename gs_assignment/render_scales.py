"""
Render scenes with reduced Gaussian scale to visualize Gaussian distribution patterns.
"""
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from scene import Scene
from gaussian_renderer import render, GaussianModel
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, get_combined_args
import torchvision

if __name__ == "__main__":
    parser = ArgumentParser(description="Reduced scale rendering")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--scale_factors", nargs="+", type=float, default=[1.0, 0.5, 0.2, 0.05])
    parser.add_argument("--view_indices", nargs="+", type=int, default=[0, 50, 100])
    args = get_combined_args(parser)

    dataset = model.extract(args)
    pipe = pipeline.extract(args)
    
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=args.iteration, shuffle=False)
    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    
    train_cameras = scene.getTrainCameras()
    
    out_dir = os.path.join(args.model_path, 'scale_analysis', "ours_{}".format(scene.loaded_iter))
    os.makedirs(out_dir, exist_ok=True)
    
    for scale_factor in args.scale_factors:
        for vi in args.view_indices:
            cam = train_cameras[vi]
            render_pkg = render(cam, gaussians, pipe, background, scale_factor)
            image = torch.clamp(render_pkg["render"], 0.0, 1.0)
            fname = os.path.join(out_dir, f"view{vi:03d}_scale{scale_factor:.2f}.png")
            torchvision.utils.save_image(image, fname)
            print(f"Saved {fname}")
    
    print("Done!")
