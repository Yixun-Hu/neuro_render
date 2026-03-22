"""Generate all images needed for the NLS assignment report."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import utils.utils as utils
from train import *

OUT = "report_images"
os.makedirs(OUT, exist_ok=True)

# Use the currently available trained checkpoint by default.
scan_folder = "processed_2025_03_06_15_45_13-temp4"
model = PanoModel.load_from_checkpoint(
    f"checkpoints/{scan_folder}/last.ckpt", device="cuda",
    cached_data=f"checkpoints/{scan_folder}/data.pkl"
)
model = model.to("cuda").eval()
model.load_volume()

B = 1.1  # brightness

def save_rgb(rgb, path):
    img = (rgb * B).clamp(0, 1).permute(1, 2, 0).cpu().numpy()
    plt.imsave(path, img)
    print(f"Saved: {path}")

# --- 1. Different time values ---
print("=== Time sweep ===")
fig, axes = plt.subplots(1, 5, figsize=(25, 5))
for i, t in enumerate([0.0, 0.25, 0.5, 0.75, 1.0]):
    _, rgb, _, _ = model.generate_outputs(height=540, width=720, time=t, fov_scale=1.5)
    axes[i].imshow((rgb * B).clamp(0, 1).permute(1, 2, 0).cpu())
    axes[i].set_title(f"time={t:.2f}", fontsize=14)
    axes[i].axis('off')
plt.tight_layout()
plt.savefig(f"{OUT}/time_sweep.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT}/time_sweep.png")

# --- 2. Different fov_scale ---
print("=== FOV sweep ===")
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for i, fov in enumerate([1.0, 1.5, 2.0, 2.5]):
    _, rgb, _, _ = model.generate_outputs(height=540, width=720, time=0.5, fov_scale=fov)
    axes[i].imshow((rgb * B).clamp(0, 1).permute(1, 2, 0).cpu())
    axes[i].set_title(f"fov_scale={fov:.1f}", fontsize=14)
    axes[i].axis('off')
plt.tight_layout()
plt.savefig(f"{OUT}/fov_sweep.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT}/fov_sweep.png")

# --- 3. Different offsets ---
print("=== Offset sweep ===")
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
offsets = [
    (0, 0, 0, "No offset"),
    (0.3, 0, 0, "X=+0.3"),
    (0, 0.3, 0, "Y=+0.3"),
    (0, 0, 0.3, "Z=+0.3"),
]
for i, (ox, oy, oz, label) in enumerate(offsets):
    translation = torch.tensor([ox, oy, oz], device="cuda", dtype=torch.float32)
    _, rgb, _, _ = model.generate_outputs(height=540, width=720, time=0.5, fov_scale=1.5, translation=translation)
    axes[i].imshow((rgb * B).clamp(0, 1).permute(1, 2, 0).cpu())
    axes[i].set_title(label, fontsize=14)
    axes[i].axis('off')
plt.tight_layout()
plt.savefig(f"{OUT}/offset_sweep.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT}/offset_sweep.png")

# --- Shared ablation settings for all three toggles ---
ablation_time = 0.65
ablation_fov = 1.8
ablation_translation = torch.tensor([0.25, -0.05, 0.20], device="cuda", dtype=torch.float32)

# --- 4. Toggle: ray_offset ---
print("=== Ray offset toggle ===")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for i, (flag, label) in enumerate([(False, "ray_offset ON"), (True, "ray_offset OFF")]):
    model.args.no_offset = flag
    _, rgb, _, _ = model.generate_outputs(
        height=540, width=720, time=ablation_time, fov_scale=ablation_fov, translation=ablation_translation
    )
    axes[i].imshow((rgb * B).clamp(0, 1).permute(1, 2, 0).cpu())
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
    _, rgb, _, _ = model.generate_outputs(
        height=540, width=720, time=ablation_time, fov_scale=ablation_fov, translation=ablation_translation
    )
    axes[i].imshow((rgb * B).clamp(0, 1).permute(1, 2, 0).cpu())
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
    _, rgb, _, _ = model.generate_outputs(
        height=540, width=720, time=ablation_time, fov_scale=ablation_fov, translation=ablation_translation
    )
    axes[i].imshow((rgb * B).clamp(0, 1).permute(1, 2, 0).cpu())
    axes[i].set_title(label, fontsize=14)
    axes[i].axis('off')
model.args.no_lens_distortion = False
plt.tight_layout()
plt.savefig(f"{OUT}/toggle_lens_distortion.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT}/toggle_lens_distortion.png")

# --- 7. Breakdown examples (extreme FOV / edge of sphere / large offset) ---
print("=== Breakdown examples ===")
fig, axes = plt.subplots(1, 4, figsize=(24, 5))
breakdown_cases = [
    (3.5, 0.5, None, "Extreme FOV (3.5)"),
    (1.5, 0.0, None, "Edge: t=0.0"),
    (1.5, 1.0, None, "Edge: t=1.0"),
    (1.5, 0.5, torch.tensor([0.7, 0.0, 0.7], device="cuda", dtype=torch.float32), "Large offset (0.7,0,0.7)"),
]
for i, (fov, t, trans, label) in enumerate(breakdown_cases):
    _, rgb, _, _ = model.generate_outputs(height=540, width=720, time=t, fov_scale=fov, translation=trans)
    axes[i].imshow((rgb * B).clamp(0, 1).permute(1, 2, 0).cpu())
    axes[i].set_title(label, fontsize=14)
    axes[i].axis('off')
plt.tight_layout()
plt.savefig(f"{OUT}/breakdown.png", dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {OUT}/breakdown.png")

print("\nAll report images generated!")
