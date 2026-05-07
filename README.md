# TPU Super-Resolution

JAX/Flax super-resolution training scaffold for Cloud TPU. The current main
model is an ATD v2-style restoration transformer: local shifted-window
attention, adaptive dictionary refinement, token-dictionary cross attention,
category-context feed-forward fusion, and x4 pixel-shuffle upsampling.

## Activate

```bash
source ~/venvs/tpu-jax/bin/activate
cd ~/dev/sr-tpu
```

## Data Layout

The prepared local dataset is here:

```text
~/datasets/sr/prepared/sr_x4_v1/
```

Supported extensions: jpg, jpeg, png, webp, bmp.

The loader creates supervised pairs automatically:

1. Random crop from a high-resolution image.
2. Downsample by `--scale`, optionally with realistic blur/noise/JPEG degradation.
3. Feed the LR crop to the HAT/ATD model.
4. Model predicts the HR crop.

Prepared folders:

```text
train_balanced_hr  # recommended first baseline: clean data, DIV2K repeated for sampling balance
train_clean_hr     # clean data without repeats
train_mixed_hr     # clean data plus optional OST JPEG images
val_x4_bicubic/    # fixed x4 LR/HR validation pairs
```

Rebuild the prepared dataset if the raw data changes:

```bash
python prepare_dataset.py --overwrite
```

## Smoke Test

```bash
python train.py --model-preset hat_atd_tiny --out checkpoints/smoke_hat_atd_tiny --crop-size 128 --batch-size 1 --steps 2 --log-every 1 --eval-every 0 --save-every 2
```

## Train

Recommended ATD v2-style real-world run:

```bash
python train.py \
  --model-preset atd_v2_xlarge \
  --data ~/datasets/sr/prepared/sr_x4_v1/train_mixed_hr \
  --val-data ~/datasets/sr/prepared/sr_x4_v1/val_hr \
  --out checkpoints/atd_v2_xlarge_guarded_x4 \
  --scale 4 \
  --crop-size 256 \
  --batch-size 2 \
  --steps 300000 \
  --steps-per-epoch 2000 \
  --lr 3e-5 \
  --warmup-steps 10000 \
  --grad-clip-norm 0.3 \
  --skip-grad-norm 20 \
  --skip-loss-threshold 0.8 \
  --degradation mixed-sharp \
  --edge-loss-weight 0.03 \
  --detail-loss-weight 0.12 \
  --spectrum-loss-weight 0.03 \
  --base-divergence-weight 0.01 \
  --base-divergence-floor 0.0039215686 \
  --net-perceptual-metrics all \
  --log-every 25 \
  --eval-every 500 \
  --save-every 1000 \
  --wandb \
  --wandb-run-name atd_v2_xlarge_guarded_x4
```

Heavier 98M-parameter run:

```bash
python train.py \
  --model-preset atd_v2_xxlarge \
  --data ~/datasets/sr/prepared/sr_x4_v1/train_mixed_hr \
  --val-data ~/datasets/sr/prepared/sr_x4_v1/val_hr \
  --out checkpoints/atd_v2_xxlarge_guarded_x4 \
  --scale 4 \
  --crop-size 256 \
  --batch-size 2 \
  --steps 300000 \
  --steps-per-epoch 2000 \
  --lr 2e-5 \
  --warmup-steps 12000 \
  --grad-clip-norm 0.25 \
  --skip-grad-norm 15 \
  --skip-loss-threshold 0.8 \
  --degradation mixed-sharp \
  --edge-loss-weight 0.025 \
  --detail-loss-weight 0.10 \
  --spectrum-loss-weight 0.025 \
  --base-divergence-weight 0.01 \
  --base-divergence-floor 0.0039215686 \
  --net-perceptual-metrics all \
  --log-every 25 \
  --eval-every 500 \
  --save-every 1000 \
  --wandb \
  --wandb-run-name atd_v2_xxlarge_guarded_x4
```

Preset sizes:

```text
hat_atd_tiny:   0.34M params
hat_atd_base:   2.59M params
hat_atd_large: 15.30M params
hat_atd_xlarge: 42.11M params
hat_atd_xxlarge: 73.95M params
atd_v2_tiny:    0.43M params
atd_v2_base:   13.02M params
atd_v2_large:  29.90M params
atd_v2_xlarge: 57.36M params
atd_v2_xxlarge:97.96M params
```

The ATD v2 presets default to `global_residual=False`, so the model cannot
collapse into simply returning the x4 cubic skip path. Start with
`atd_v2_xlarge`; move to `atd_v2_xxlarge` if throughput and memory look healthy.

`crop-size / scale` must be divisible by `window-size`. The defaults are
`crop-size=256`, `scale=4`, `window-size=8`, so LR patches are `64x64`.

## W&B And Samples

Install is already done in `~/venvs/tpu-jax`. Log in once for online runs:

```bash
wandb login
```

If no W&B login is found, `--wandb` automatically falls back to offline mode.
Sync later with:

```bash
wandb sync wandb/offline-run-...
```

Training logs:

```text
train/loss, train/pixel_loss, train/edge_loss, train/psnr, train/lr, train/grad_norm, train/steps_per_sec
train/base_mae, train/collapse_loss, train/skip_update
eval/loss, eval/pixel_loss, eval/edge_loss, eval/psnr, eval/bicubic_psnr, eval/psnr_gain_vs_bicubic
eval/detail_loss, eval/spectrum_loss, eval/base_mae, eval/collapse_loss
eval/ssim, eval/ms_ssim, eval/color_delta_e
eval/bicubic_ssim, eval/bicubic_ms_ssim, eval/bicubic_color_delta_e
eval/ssim_gain_vs_bicubic, eval/ms_ssim_gain_vs_bicubic, eval/color_delta_e_improvement_vs_bicubic
eval/images: one fixed validation contact sheet, input | pred | target | absdiff x4
samples/usr_samples: one contact sheet, bicubic | pred | abs(pred-bicubic) x16 | x64
samples/<name>_pred_minus_bicubic_mae/p99/max
```

`eval/images` uses the same fixed validation crops for the whole run, so W&B
image history is directly comparable across epochs.
`samples/usr_samples` is logged as a single compact contact sheet and also saved
locally as `_contact_sheet.jpg` in each sample epoch folder.

Optional neural perceptual metrics can be logged on those fixed validation crops:

```bash
--net-perceptual-metrics lpips  # requires torch, torchvision, lpips
--net-perceptual-metrics dists  # requires torch, torchvision, piq
--net-perceptual-metrics all
```

They log `eval/lpips_fixed` and/or `eval/dists_fixed`; lower is better.

`usr_samples` are inferred every epoch by default and saved locally under:

```text
checkpoints/<run>/usr_samples/epoch_<epoch>_step_<step>/
```

Useful knobs:

```bash
--data-workers 4
--preload-data
--prefetch-batches 16
--image-log-count 4
--image-log-every-epochs 1
--sample-dir usr_samples
--sample-max-images 8
--sample-max-side 512
--sample-log-every-epochs 1
--steps-per-epoch 0
--progress auto
--wandb-quiet
```

The default training loader decodes unique images into host RAM, then uses four
background worker threads to crop/augment/degrade batches into a prefetch queue.
For `train_balanced_hr`, that is 12,184 sampling entries backed by 9,784 unique
images, roughly 13 GiB of raw RGB host memory.

For the cleanest terminal output, keep the default progress bar. Metrics update
in the progress postfix, while eval/checkpoint/sample events are printed as
separate clean lines. To write plain metric lines instead, use `--progress off`.

## Real-World Restoration Data

The bicubic-style dataset is not enough for noisy phone photos. For the next
run, use SIDD noisy/GT pairs plus a smaller amount of clean HR self-pairs.

Download SIDD Small sRGB:

```bash
python download_real_datasets.py \
  --dataset sidd-small-srgb \
  --root ~/datasets/sr/real
```

Prepare pair manifests. The clean HR root is optional, but it gives the model
more texture/general detail while SIDD teaches real phone noise:

```bash
python prepare_real_dataset.py \
  --sidd-root ~/datasets/sr/real/sidd-small-srgb \
  --clean-hr-root ~/datasets/sr/prepared/sr_x4_v1/train_mixed_hr \
  --out ~/datasets/sr/prepared/sr_real_x4_v1 \
  --min-side 256 \
  --sidd-repeat 16 \
  --clean-limit 3000 \
  --overwrite
```

The loader auto-detects `pairs.jsonl`. For SIDD rows it crops aligned
noisy/clean images, downsamples the noisy crop to LR, and trains toward the
clean HR crop. For clean self-pairs, it synthesizes phone-like LR degradation.

Recommended first real-denoise run:

```bash
python train.py \
  --model-preset atd_v2_xlarge \
  --data ~/datasets/sr/prepared/sr_real_x4_v1/train_pairs \
  --val-data ~/datasets/sr/prepared/sr_real_x4_v1/val_pairs \
  --out checkpoints/atd_v2_xlarge_sidd_real_x4 \
  --scale 4 \
  --crop-size 256 \
  --batch-size 2 \
  --steps 200000 \
  --steps-per-epoch 2000 \
  --lr 2e-5 \
  --warmup-steps 8000 \
  --grad-clip-norm 0.25 \
  --skip-grad-norm 15 \
  --skip-loss-threshold 0.8 \
  --degradation mixed-denoise \
  --edge-loss-weight 0.02 \
  --detail-loss-weight 0.08 \
  --spectrum-loss-weight 0.02 \
  --base-divergence-weight 0.0 \
  --net-perceptual-metrics all \
  --log-every 25 \
  --eval-every 500 \
  --save-every 1000 \
  --wandb \
  --wandb-run-name atd_v2_xlarge_sidd_real_x4
```

## Inference

```bash
python infer.py \
  --checkpoint checkpoints/atd_v2_xlarge_guarded_x4 \
  --input usr_samples/seo.jpg \
  --output samples/seo_x4.png
```

Compare the latest restored checkpoint on a folder:

```bash
python infer.py \
  --checkpoint checkpoints/atd_v2_xlarge_guarded_x4 \
  --input usr_samples \
  --output samples/atd_v2_latest \
  --save-bicubic \
  --compare
```

Compare specific checkpoint steps if the `checkpoint_<step>` directories still
exist:

```bash
python infer.py \
  --checkpoint checkpoints/atd_v2_xlarge_guarded_x4 \
  --checkpoint-step 279000 280000 281000 \
  --input usr_samples \
  --output samples/atd_v2_candidates \
  --save-bicubic \
  --compare
```

While training is still using the TPU, force CPU inference:

```bash
python infer.py \
  --platform cpu \
  --checkpoint checkpoints/atd_v2_xlarge_guarded_x4/checkpoint_281000 \
  --input usr_samples \
  --output samples/atd_v2_cpu_check \
  --max-side 256 \
  --compare
```

For large photos, tile the LR input to reduce memory pressure:

```bash
python infer.py \
  --checkpoint checkpoints/atd_v2_xlarge_guarded_x4 \
  --input ~/Pictures/to_upscale \
  --output samples/final_x4 \
  --tile-size 128 \
  --tile-overlap 16
```

`infer.py` writes `metrics.csv` for folder/multi-checkpoint runs with
`mae_vs_bicubic`, `p99_vs_bicubic`, and output paths. Comparison sheets are saved
under `output/compare/`.

## Notes

- `mixed-real` adds random Gaussian/motion blur, mixed resize kernels, JPEG,
  grayscale RGB noise, chroma noise, color jitter, and light post-resize blur.
- `mixed-balanced` and `mixed-sharp` mix clean bicubic, light degradation, and
  real degradation so the model sees enough sharp examples instead of learning
  only safe smoothing.
- `phone-real` is the stronger phone-photo degradation path: heavier
  blur/resampling, JPEG/WebP, RGB/chroma noise, banding, and occasional
  over-sharpening artifacts.
- `mixed-denoise` mostly samples `phone-real`, with a smaller amount of
  `mixed-real` and clean bicubic so the model does not learn only smoothing.
- `atd_v2_*` presets predict the full HR image by default instead of hiding
  behind a cubic skip path.
- Global residual uses a cubic x4 base; the model predicts a residual on top.
- `--detail-loss-weight` and `--spectrum-loss-weight` add direct pressure on
  high-frequency reconstruction. Watch `samples/*pred_minus_bicubic*` to ensure
  the model is not collapsing back to a near-bicubic output.
- `--skip-grad-norm` and `--skip-loss-threshold` skip outlier updates before a
  single bad batch can push the model into a collapsed solution.
- `--base-divergence-weight` is a light guardrail against outputs becoming
  indistinguishable from the cubic base. Keep it small.
- For real photos, degradation design matters as much as model size.
- TPU compile happens on the first step, so step 1 is slower than later steps.
