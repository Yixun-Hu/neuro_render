# DDPM Assignment Written Answers (Q1-Q3)

## Q1 - Forward Process
As timestep `t` increases, the signal-to-noise ratio (SNR) decreases monotonically because the clean signal is scaled by `sqrt(alpha_bar_t)` while noise grows with `sqrt(1 - alpha_bar_t)`. Since `alpha_bar_t = prod_s alpha_s` keeps shrinking over time, the retained signal energy goes to near zero and the sample becomes dominated by Gaussian noise. At large `t`, `x_t` is therefore approximately distributed as `N(0, I)`, which is why samples look like pure random noise.

## Q2 - DDPM Training
Across training epochs, one-step denoising improves because the model learns a better estimate of the true injected noise `epsilon`. Early epochs remove only coarse corruption; later epochs recover clearer structure and sharper details. Across timesteps, denoising is easier at small `t` (weak corruption) and harder at large `t` (heavy corruption with little signal left), so quality typically degrades as `t` increases.

One-step denoising is not enough for high-quality generation because DDPM generation is inherently a multi-step process: each reverse step removes only part of the noise and progressively refines the sample distribution. High-quality final samples require composing many small denoising updates from `t = T-1` to `t = 0`.

## Q3 - Stochasticity in the Reverse Process
We add random noise for `t > 0` to sample from the learned reverse conditional distribution instead of collapsing to a single deterministic path. This stochasticity preserves diversity and better matches the probabilistic model of the data distribution.

At `t = 0`, no extra noise is added because we want a clean final sample estimate `x_0`. If noise is removed at all steps (`sigma_t = 0`), sampling becomes deterministic for a fixed initial `x_T`; this usually reduces diversity and can produce less realistic or mode-collapsed outputs, even if trajectories look smoother.

## Bonus Task: DDPM Image Generation on EMNIST-Letters

I applied DDPM to **EMNIST-Letters** (handwritten English letter images, A-Z, 28x28 grayscale) — a dataset distinct from Fashion-MNIST and MNIST, as implemented in `scripts/5_bonus_emnist.py`.

### Outputs
- `outputs/5_bonus_loss_emnist.png` — training loss curve
- `outputs/5_bonus_samples_emnist.png` — 4x4 grid of generated letter images from pure noise
- `outputs/5_bonus_sampling_emnist.gif` — reverse denoising trajectory from noise to letters
- `outputs/5_bonus_noise_comparison_emnist.gif` — with-noise vs no-noise sampling comparison

### Observations
- The model successfully generates recognizable handwritten English letters from pure Gaussian noise after only 15 epochs of training.
- The denoising trajectory shows a clear progression: coarse letter shapes emerge early in the reverse process, with fine stroke details refined in the final steps.
- The noise comparison confirms the same pattern seen in Fashion-MNIST: stochastic sampling produces more diverse letter styles, while deterministic sampling (sigma=0) yields less varied outputs.
