"""Generate all images needed for the NLS assignment report.
Uses the same rendering pipeline as the notebook widget (model.inference)."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utils.utils as utils
from train import *

OUT = "report_images"
os.makedirs(OUT, exist_ok=True)

scan_folder = "processed_2025_03_06_15_45_13-temp4"
model = PanoModel.load_from_checkpoint(
    f"checkpoints/{scan_folder}/last.ckpt", device="cuda",
    cached_data=f"checkpoints/{scan_folder}/data.pkl"
)
model = model.to("cuda").eval()
model.load_volume()

B = 1.1  # brightness


def render(height=540, width=720, time=0.55, fov_scale=1.0,
           offset_x=0.0, offset_y=0.0, offset_z=0.0,
           offset_qw=0.0, offset_qx=0.0, offset_qy=0.0, offset_qz=0.0):
    """Render matching the notebook widget's render_image() exactly."""
    offset_translation = torch.tensor([offset_x, offset_y, offset_z], device="cuda", dtype=torch.float32)
    offset_quaternion = torch.tensor([offset_qw, offset_qx, offset_qy, offset_qz], device="cuda", dtype=torch.float32)

    t_tensor = torch.full((width * height,), time, device="cuda", dtype=torch.float32)

    frame_index = int(time * model.args.num_frames - 1)
    frame_index = max(0, min(frame_index, model.args.num_frames - 1))

    intrinsics_inv = model.data.intrinsics_inv[frame_index].clone()
    intrinsics_inv[0] *= fov_scale
    intrinsics_inv = intrinsics_inv.unsqueeze(0).repeat(width * height, 1, 1).to("cuda")

    quaternion_camera_to_world = model.data.quaternion_camera_to_world[frame_index].to("cuda")
    quaternion_camera_to_world = quaternion_camera_to_world + offset_quaternion
    quaternion_camera_to_world = quaternion_camera_to_world / quaternion_camera_to_world.norm(dim=-1, keepdim=True)

    camera_to_world = model.model_rotation(quaternion_camera_to_world, t_tensor).to("cuda")
    ray_origins = model.model_translation(t_tensor, 1.0) + offset_translation

    uv = utils.make_grid(height, width, [0, 1], [0, 1]).to("cuda")
    adjusted_uv = uv * fov_scale + 0.5 * (1 - fov_scale)

    ray_directions = model.generate_ray_directions(adjusted_uv, camera_to_world, intrinsics_inv)
    with torch.no_grad():
        rgb_transmission = model.inference(t_tensor, adjusted_uv.clamp(0, 1), ray_origins, ray_directions, 1.0)

    rgb_image = model.color_and_tone(rgb_transmission, height, width).permute(1, 2, 0).detach().cpu()
    rgb_image = (rgb_image * B).clamp(0, 1)
    return rgb_image


# --- 1. Different time values (fov_scale=1.0, default offsets) ---
print("=== Time sweep ===")
fig, axes = plt.subplots(1, 5, figsize=(25, 5))
for i, t in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
    axes[i].imshow(render(time=t, fov_scale=1.0))
    axes[i].set_title(f"time={t:.2f}", fontsize=14)
    axes[i].axis('off')
plt.tight_layout()
plt.savefig(f"{OUT}/time_sweep.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT}/time_sweep.png")

# --- 2. Different fov_scale (time=0.0, default offsets) ---
print("=== FOV sweep ===")
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, fov in enumerate([0.5, 1.0, 2.0]):
    axes[i].imshow(render(time=0.0, fov_scale=fov))
    axes[i].set_title(f"fov_scale={fov:.1f}", fontsize=14)
    axes[i].axis('off')
plt.tight_layout()
plt.savefig(f"{OUT}/fov_sweep.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT}/fov_sweep.png")

# --- 3. Different offsets (time=0.5, fov_scale=1.0) ---
print("=== Offset sweep ===")
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
offsets = [
    (0, 0, 0, "No offset"),
    (0.3, 0, 0, "X=+0.3"),
    (0, 0.3, 0, "Y=+0.3"),
    (0, 0, 0.3, "Z=+0.3"),
]
for i, (ox, oy, oz, label) in enumerate(offsets):
    axes[i].imshow(render(time=0.5, fov_scale=1.0, offset_x=ox, offset_y=oy, offset_z=oz))
    axes[i].set_title(label, fontsize=14)
    axes[i].axis('off')
plt.tight_layout()
plt.savefig(f"{OUT}/offset_sweep.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT}/offset_sweep.png")

# --- Shared ablation settings for all three toggles ---
abl_time = 0.65
abl_fov = 1.8
abl_ox, abl_oy, abl_oz = 0.25, -0.05, 0.20

# --- 4. Toggle: ray_offset ---
print("=== Ray offset toggle ===")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, (flag, label) in enumerate([(False, "ray_offset ON"), (True, "ray_offset OFF")]):
    model.args.no_offset = flag
    axes[i].imshow(render(time=abl_time, fov_scale=abl_fov, offset_x=abl_ox, offset_y=abl_oy, offset_z=abl_oz))
    axes[i].set_title(label, fontsize=14)
    axes[i].axis('off')
model.args.no_offset = False
plt.tight_layout()
plt.savefig(f"{OUT}/toggle_ray_offset.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT}/toggle_ray_offset.png")

# --- 5. Toggle: view_color ---
print("=== View color toggle ===")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, (flag, label) in enumerate([(False, "view_color ON"), (True, "view_color OFF")]):
    model.args.no_view_color = flag
    axes[i].imshow(render(time=abl_time, fov_scale=abl_fov, offset_x=abl_ox, offset_y=abl_oy, offset_z=abl_oz))
    axes[i].set_title(label, fontsize=14)
    axes[i].axis('off')
model.args.no_view_color = False
plt.tight_layout()
plt.savefig(f"{OUT}/toggle_view_color.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT}/toggle_view_color.png")

# --- 6. Toggle: lens_distortion ---
print("=== Lens distortion toggle ===")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, (flag, label) in enumerate([(False, "lens_distortion ON"), (True, "lens_distortion OFF")]):
    model.args.no_lens_distortion = flag
    axes[i].imshow(render(time=abl_time, fov_scale=abl_fov, offset_x=abl_ox, offset_y=abl_oy, offset_z=abl_oz))
    axes[i].set_title(label, fontsize=14)
    axes[i].axis('off')
model.args.no_lens_distortion = False
plt.tight_layout()
plt.savefig(f"{OUT}/toggle_lens_distortion.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT}/toggle_lens_distortion.png")

# --- 7. Breakdown examples ---
print("=== Breakdown examples ===")
breakdown_cases = [
    dict(fov_scale=3.5, time=0.5, label="Extreme FOV (3.5)"),
    dict(fov_scale=1.0, time=0.5, offset_x=0.7, offset_z=0.7, label="Large offset (0.7,0,0.7)"),
]
fig, axes = plt.subplots(1, len(breakdown_cases), figsize=(12, 5))
for i, case in enumerate(breakdown_cases):
    label = case.pop("label")
    axes[i].imshow(render(**case))
    axes[i].set_title(label, fontsize=14)
    axes[i].axis('off')
plt.tight_layout()
plt.savefig(f"{OUT}/breakdown.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT}/breakdown.png")

print("\nAll report images generated!")
