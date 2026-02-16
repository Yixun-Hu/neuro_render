# 2D Gaussian Splatting Assignment Report

**Name:** Yixun Hu  
**Net ID:** yh4742


## 1. Implementation Description

### `forward.cu` — Gaussian to Pixel Coordinate Transforms

The `compute_transmat` function constructs three matrices to map 2D Gaussian splat coordinates to pixel space:

- **`splat2world`** (transposed 4×3 matrix): Maps local splat coordinates `[u, v, 1]^T` to homogeneous world coordinates. The first two columns use the orientation vectors `L[0]` and `L[1]` (from `R * S`), and the third column contains the Gaussian center `(p_orig.x, p_orig.y, p_orig.z, 1)`.

- **`world2ndc`** (transposed 4×4 matrix): The projection matrix transposed from row-major `projmatrix[]` into GLM's column-major format by swapping rows and columns.

- **`ndc2pix`** (3×4 matrix): Converts NDC to pixel coordinates using viewport scaling `W/2` and `H/2` with offsets `(W-1)/2` and `(H-1)/2`.

**Implementation note:** The correct matrix values were cross-referenced with the backward pass in `backward.cu::compute_transmat_aabb`, which contains the complete reference for all three matrices. GLM uses column-major ordering, so care was taken to ensure the transpose semantics matched.


### `gaussian_model.py` — Adaptive Density Control

Three functions were implemented for the densification and pruning pipeline:

- **`densify_and_split`**: Selects Gaussians with high positional gradient (`≥ grad_threshold`) **and** large scale (`max(scale) > percent_dense * scene_extent`). These represent over-reconstructed regions where large Gaussians need to be subdivided into `N` smaller ones.

- **`densify_and_clone`**: Selects Gaussians with high positional gradient (`≥ grad_threshold`) **and** small scale (`max(scale) ≤ percent_dense * scene_extent`). These represent under-reconstructed regions where small Gaussians are duplicated to increase coverage.

- **`densify_and_prune`**: Removes Gaussians that are:
  - Nearly transparent: `opacity < min_opacity`
  - Too large in view space: `max_radii2D > max_screen_size`
  - Too large in world space: `max(scale) > 0.1 * extent`

**Key design insight:** The split vs. clone distinction is based on scale relative to the scene — large Gaussians that need refinement get split (replaced by N smaller ones), while small Gaussians in under-reconstructed areas get cloned (duplicated at the same position to allow the optimizer to spread them).


## 2. Training Progression

The model was trained on the Lego scene for 30,000 iterations. Below are representative renderings at iterations 7,000 and 30,000 from different viewpoints.

### Front View (View 0)

| Iteration 7,000 | Iteration 30,000 | Ground Truth |
|:---:|:---:|:---:|
| ![iter7k_v0](output/lego/train/ours_7000/renders/00000.png) | ![iter30k_v0](output/lego/train/ours_30000/renders/00000.png) | ![gt_v0](output/lego/train/ours_30000/gt/00000.png) |

### Side View (View 50)

| Iteration 7,000 | Iteration 30,000 | Ground Truth |
|:---:|:---:|:---:|
| ![iter7k_v50](output/lego/train/ours_7000/renders/00050.png) | ![iter30k_v50](output/lego/train/ours_30000/renders/00050.png) | ![gt_v50](output/lego/train/ours_30000/gt/00050.png) |

### Another Angle (View 100)

| Iteration 7,000 | Iteration 30,000 | Ground Truth |
|:---:|:---:|:---:|
| ![iter7k_v100](output/lego/train/ours_7000/renders/00100.png) | ![iter30k_v100](output/lego/train/ours_30000/renders/00100.png) | ![gt_v100](output/lego/train/ours_30000/gt/00100.png) |

**Observations on training progression:**

- **Iteration 7,000**: The overall structure and colors of the lego bulldozer are already well-captured. Major geometric features (body, treads, arm, baseplate) are recognizable. However, fine details like small lego studs, thin mechanical parts, and edge sharpness show some softness.

- **Iteration 30,000**: Substantial improvement in fine detail fidelity. Lego studs are crisper, mechanical joints are more defined, and surface textures (e.g., the gray baseplate pattern) are sharper. The rendering is nearly indistinguishable from the ground truth at this stage.

- **Key quality improvements**: The most notable improvements between 7k and 30k iterations occur in (1) edge sharpness and anti-aliasing, (2) fine geometric detail like lego studs and thin rods, and (3) color accuracy in shadow regions. The normal and distortion regularization losses (activated at iterations 3k and 7k respectively) contribute to the improved surface quality in later iterations.


## 3. Reduced Scale Gaussian Distribution

By rendering with reduced Gaussian scales, we can visualize the spatial distribution of the learned Gaussians. The scale modifier multiplies each Gaussian's covariance, making them smaller and revealing individual splat positions.

### View 0 — Scale Comparisons

| Scale = 1.0 (Full) | Scale = 0.5 | Scale = 0.2 | Scale = 0.05 |
|:---:|:---:|:---:|:---:|
| ![s1.0](output/lego/scale_analysis/ours_30000/view000_scale1.00.png) | ![s0.5](output/lego/scale_analysis/ours_30000/view000_scale0.50.png) | ![s0.2](output/lego/scale_analysis/ours_30000/view000_scale0.20.png) | ![s0.05](output/lego/scale_analysis/ours_30000/view000_scale0.05.png) |

### View 50 — Scale Comparisons

| Scale = 1.0 (Full) | Scale = 0.5 | Scale = 0.2 | Scale = 0.05 |
|:---:|:---:|:---:|:---:|
| ![s1.0](output/lego/scale_analysis/ours_30000/view050_scale1.00.png) | ![s0.5](output/lego/scale_analysis/ours_30000/view050_scale0.50.png) | ![s0.2](output/lego/scale_analysis/ours_30000/view050_scale0.20.png) | ![s0.05](output/lego/scale_analysis/ours_30000/view050_scale0.05.png) |

### View 100 — Scale Comparisons

| Scale = 1.0 (Full) | Scale = 0.5 | Scale = 0.2 | Scale = 0.05 |
|:---:|:---:|:---:|:---:|
| ![s1.0](output/lego/scale_analysis/ours_30000/view100_scale1.00.png) | ![s0.5](output/lego/scale_analysis/ours_30000/view100_scale0.50.png) | ![s0.2](output/lego/scale_analysis/ours_30000/view100_scale0.20.png) | ![s0.05](output/lego/scale_analysis/ours_30000/view100_scale0.05.png) |

**Patterns observed in Gaussian distribution:**

1. **Surface-aligned distribution**: Gaussians are densely concentrated along visible surfaces (baseplate, body panels, treads). This is consistent with 2D Gaussian Splatting's use of flat (disk-like) primitives that align with surfaces.

2. **Higher density at geometric complexity**: Areas with fine detail (lego studs, mechanical joints, thin rods) have a much denser concentration of Gaussians compared to flat surfaces. The adaptive density control (split/clone) creates more Gaussians where the reconstruction error gradient is highest.

3. **Edge emphasis**: At very small scales (0.05), Gaussians can be seen lining the edges and boundaries of the object. This is because edges require many small Gaussians to accurately represent the sharp transitions.

4. **Sparse in simple regions**: Large flat areas like the baseplate have relatively fewer, larger Gaussians — the pruning mechanism removes redundant Gaussians while the split mechanism avoids over-subdividing areas that are already well-represented.

5. **No background Gaussians**: Since the scene uses a black background, the pruning step effectively removes any Gaussians that might float in empty space (low opacity → pruned).


## 4. Bonus: Novel View Animation

A customized camera trajectory animation was generated by modifying `generate_path` in `utils/render_utils.py` to add vertical height variation (`z_variation=1.0`), creating a **spiral path** instead of the default flat ellipse:

```python
# utils/render_utils.py, line 165
new_poses = generate_ellipse_path(poses=pose_recenter, n_frames=n_frames, z_variation=1.0)
```

This was rendered using:

```
python render.py -s data/nerf_synthetic/lego/ -m output/lego --iteration 30000 \
    --skip_mesh --skip_test --skip_train --render_path
```

The resulting video is saved at:  
[output/lego/traj/ours_30000/render_traj_color.mp4](output/lego/traj/ours_30000/render_traj_color.mp4). And this video is submitted as attached.

With `z_variation=1.0`, the camera spirals around the lego scene with vertical oscillation, providing a more dynamic view that reveals the object from both above and below. This 240-frame animation demonstrates that the trained 2DGS model produces temporally consistent, high-quality renders from novel viewpoints not seen during training.
