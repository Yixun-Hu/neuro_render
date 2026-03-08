"""Bonus Task: Apply DDPM to EMNIST-Letters and generate images from noise.

EMNIST-Letters contains 28x28 grayscale images of handwritten English letters (A-Z),
so we can reuse SimpleUNet directly. This script:
  1) Trains DDPM on EMNIST-Letters
  2) Generates images from pure noise
  3) Visualizes the reverse denoising trajectory
  4) Compares with-noise vs no-noise sampling
  5) Saves a final grid of generated samples
"""
import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

TASK_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.join(TASK_DIR, "..")
sys.path.insert(0, os.path.join(ROOT_DIR, "src"))

from ddpm import ddpm_sample, ddpm_sample_no_noise, ddpm_train_step, make_schedule
from util import SimpleUNet, train_ddpm
from visualization import (
    animate_noise_comparison_image,
    plot_image_denoising,
    plot_training_loss,
    save_image_grid,
)


def make_emnist_dataloader(batch_size=128):
    """EMNIST-Letters 28x28 grayscale, normalized to [-1, 1].
    EMNIST images are transposed relative to MNIST, so we rotate them back."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.transpose(1, 2).flip(2)),  # fix EMNIST orientation
        transforms.Lambda(lambda x: x * 2 - 1),
    ])
    dataset = datasets.EMNIST(
        root="./data", split="letters", train=True, download=True, transform=transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    output_dir = os.path.join(ROOT_DIR, "outputs")
    checkpoint_dir = os.path.join(ROOT_DIR, "checkpoints")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    schedule = make_schedule(T=1000, device=device)

    # --- Data and Model ---
    dataloader = make_emnist_dataloader(batch_size=128)
    model = SimpleUNet(in_channels=1).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"SimpleUNet parameters: {total_params:,}")

    # --- Train ---
    epochs = 15
    print(f"\n=== Training on EMNIST-Letters ({epochs} epochs) ===")
    losses = train_ddpm(
        model,
        dataloader,
        schedule,
        ddpm_train_step,
        epochs=epochs,
        lr=2e-4,
        device=device,
        log_interval=1,
    )
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "model_emnist.pth"))
    torch.save(losses, os.path.join(checkpoint_dir, "losses_emnist.pt"))

    # --- Loss curve ---
    plot_training_loss(losses, save_path=os.path.join(output_dir, "5_bonus_loss_emnist.png"))

    # --- Generate images from noise ---
    print("\n=== Generating EMNIST letter images from noise ===")
    samples, traj = ddpm_sample(
        model, (16, 1, 28, 28), schedule, n_snapshot_steps=20, device=device
    )

    # Denoising trajectory GIF
    plot_image_denoising(
        traj, save_path=os.path.join(output_dir, "5_bonus_sampling_emnist.gif")
    )

    # Final generated samples grid
    save_image_grid(
        samples, save_path=os.path.join(output_dir, "5_bonus_samples_emnist.png"), ncols=4
    )

    # --- With-noise vs no-noise comparison ---
    print("\n=== Noise comparison ===")
    _, traj_w = ddpm_sample(
        model, (8, 1, 28, 28), schedule, n_snapshot_steps=20, device=device
    )
    _, traj_wo = ddpm_sample_no_noise(
        model, (8, 1, 28, 28), schedule, n_snapshot_steps=20, device=device
    )
    animate_noise_comparison_image(
        traj_w, traj_wo,
        save_path=os.path.join(output_dir, "5_bonus_noise_comparison_emnist.gif"),
    )

    print("\nBonus image generation task complete!")
    print("Outputs:")
    print(f"  - outputs/5_bonus_loss_emnist.png")
    print(f"  - outputs/5_bonus_sampling_emnist.gif")
    print(f"  - outputs/5_bonus_samples_emnist.png")
    print(f"  - outputs/5_bonus_noise_comparison_emnist.gif")


if __name__ == "__main__":
    main()
