import numpy as np
import matplotlib.pyplot as plt
import utils.utils as utils
from train import *

# Load model and data
scan_folder = "processed_2025_03_06_15_45_13-temp4"
model = PanoModel.load_from_checkpoint(f"checkpoints/{scan_folder}/last.ckpt", device="cuda", cached_data=f"checkpoints/{scan_folder}/data.pkl")
model = model.to("cuda")
model = model.eval()
model.load_volume()

# Generate outputs (reference image, rendered image)
rgb_reference, rgb, _, _ = model.generate_outputs(height=1080, width=1440, u_lims=[0, 1], v_lims=[0, 1], time=0.55, fov_scale=2.4)
brightness = 1.1

# Apply brightness for visualization
rgb = (rgb * brightness).clamp(0,1)
rgb_reference = (rgb_reference * brightness).clamp(0,1)

# Plot images
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].imshow(rgb_reference.permute(1, 2, 0).cpu())
ax[0].set_aspect(1.6)
ax[0].set_title("Reference")
ax[1].imshow(rgb.permute(1, 2, 0).cpu() )
ax[1].set_title("Wide FOV Re-Render")

for a in ax:
    a.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

plt.subplots_adjust(wspace=0.0)
# plt.show()
plt.savefig('visualization.png')
