from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
logging.getLogger("absl").setLevel(logging.ERROR)

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import checkpoints
from PIL import Image, ImageDraw, ImageOps
from tqdm import tqdm

from sr_tpu.data import IMAGE_EXTENSIONS, batch_iterator, count_training_items, is_pair_dataset
from sr_tpu.model import ModelConfig, apply_preset, create_model
from sr_tpu.train_state import TrainState


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TPU-friendly super-resolution models on JAX/Flax.")
    parser.add_argument(
        "--data",
        default="~/datasets/sr/prepared/sr_x4_v1/train_balanced_hr",
        help="Directory containing high-resolution training images.",
    )
    parser.add_argument(
        "--val-data",
        default="~/datasets/sr/prepared/sr_x4_v1/val_hr",
        help="Directory containing high-resolution validation images. Set empty to disable eval.",
    )
    parser.add_argument("--out", default="checkpoints/hat_atd_base_x4", help="Checkpoint output directory.")
    parser.add_argument(
        "--model-preset",
        choices=[
            "custom",
            "edsr_lite",
            "hat_atd_tiny",
            "hat_atd_base",
            "hat_atd_large",
            "hat_atd_xlarge",
            "hat_atd_xxlarge",
            "atd_v2_tiny",
            "atd_v2_base",
            "atd_v2_large",
            "atd_v2_xlarge",
            "atd_v2_xxlarge",
        ],
        default="hat_atd_base",
    )
    parser.add_argument("--model", choices=["edsr_lite", "hat_atd", "atd_v2"], default="hat_atd")
    parser.add_argument("--scale", type=int, default=4, help="Super-resolution scale factor.")
    parser.add_argument("--crop-size", type=int, default=256, help="High-resolution crop size.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--steps", type=int, default=100000)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--skip-grad-norm",
        type=float,
        default=0.0,
        help="Skip an optimizer update when raw grad norm exceeds this value. 0 disables.",
    )
    parser.add_argument(
        "--skip-loss-threshold",
        type=float,
        default=0.0,
        help="Skip an optimizer update when raw loss exceeds this value. 0 disables.",
    )
    parser.add_argument("--features", type=int, default=96)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--token-count", type=int, default=64)
    parser.add_argument("--channel-reduction", type=int, default=4)
    parser.add_argument("--residual-scale", type=float, default=0.15)
    parser.add_argument("--no-global-residual", action="store_true")
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument(
        "--input-mode",
        choices=["auto", "lr", "bicubic_up"],
        default="auto",
        help="Use lr for pixel-shuffle transformer models; bicubic_up preserves the old EDSR baseline.",
    )
    parser.add_argument(
        "--degradation",
        choices=[
            "bicubic",
            "mixed-light",
            "mixed-balanced",
            "mixed-sharp",
            "mixed-real",
            "phone-real",
            "mixed-denoise",
        ],
        default="bicubic",
        help="CPU-side synthetic LR degradation applied after HR crops.",
    )
    parser.add_argument(
        "--edge-loss-weight",
        type=float,
        default=0.0,
        help="Add Charbonnier loss on image gradients. Useful for sharper real-world SR runs.",
    )
    parser.add_argument(
        "--detail-loss-weight",
        type=float,
        default=0.0,
        help="Add Laplacian high-frequency reconstruction loss. Helps avoid bicubic-looking outputs.",
    )
    parser.add_argument(
        "--spectrum-loss-weight",
        type=float,
        default=0.0,
        help="Add multi-scale high-pass reconstruction loss for texture/detail pressure.",
    )
    parser.add_argument(
        "--base-divergence-weight",
        type=float,
        default=0.0,
        help="Penalize collapse to the x4 cubic base when pred/base MAE falls below the floor.",
    )
    parser.add_argument(
        "--base-divergence-floor",
        type=float,
        default=1.0 / 255.0,
        help="Target minimum pred/base MAE in 0..1 units for --base-divergence-weight.",
    )
    parser.add_argument(
        "--net-perceptual-metrics",
        choices=["off", "lpips", "dists", "all"],
        default="off",
        help="Optional PyTorch perceptual metrics on fixed eval images. Requires torch plus lpips and/or piq.",
    )
    parser.add_argument(
        "--net-perceptual-every-epochs",
        type=int,
        default=1,
        help="Run optional LPIPS/DISTS every N epochs on fixed eval images. 0 disables.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-workers", type=int, default=4)
    parser.add_argument("--eval-data-workers", type=int, default=0)
    parser.add_argument("--prefetch-batches", type=int, default=16)
    parser.add_argument(
        "--preload-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Decode unique dataset images into host RAM before training.",
    )
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-at-start", action="store_true")
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=0,
        help="Epoch length for sample logging. 0 means ceil(num_train_images / batch_size).",
    )
    parser.add_argument("--image-log-count", type=int, default=4, help="Validation crop comparisons to log to W&B.")
    parser.add_argument(
        "--image-log-every-epochs",
        type=int,
        default=1,
        help="Log validation crop images every N epochs. 0 disables image logging.",
    )
    parser.add_argument("--sample-dir", default="usr_samples", help="User LR images to upscale during training.")
    parser.add_argument("--sample-max-images", type=int, default=8)
    parser.add_argument(
        "--sample-max-side",
        type=int,
        default=512,
        help="Resize user samples so their longest LR side is at most this value before inference. 0 disables resizing.",
    )
    parser.add_argument(
        "--sample-log-every-epochs",
        type=int,
        default=1,
        help="Run and log usr_samples every N epochs. 0 disables sample inference.",
    )
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--keep-checkpoints", type=int, default=3)
    parser.add_argument("--resume", default="", help="Checkpoint dir to restore before training.")
    parser.add_argument(
        "--reset-optimizer-on-resume",
        action="store_true",
        help="Restore model weights from --resume, but restart optimizer state and step count for fine-tuning.",
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="sr-tpu")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-mode", choices=["auto", "online", "offline", "disabled"], default="auto")
    parser.add_argument("--wandb-tags", default="hat-atd,tpu,sr-x4")
    parser.add_argument("--progress", choices=["auto", "on", "off"], default="auto")
    parser.add_argument(
        "--wandb-quiet",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reduce W&B console chatter. Use --no-wandb-quiet to show W&B startup logs.",
    )
    return parser.parse_args()


def psnr(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    mse = jnp.mean((jnp.clip(pred, 0.0, 1.0) - target) ** 2)
    return -10.0 * jnp.log10(jnp.maximum(mse, 1e-10))


def charbonnier(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.sqrt((pred - target) ** 2 + 1e-6))


def image_gradients(image: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    dx = image[:, :, 1:, :] - image[:, :, :-1, :]
    dy = image[:, 1:, :, :] - image[:, :-1, :, :]
    return dx, dy


def edge_charbonnier(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    pred_dx, pred_dy = image_gradients(pred)
    target_dx, target_dy = image_gradients(target)
    return 0.5 * (charbonnier(pred_dx, target_dx) + charbonnier(pred_dy, target_dy))


def _depthwise_filter(image: jnp.ndarray, kernel_2d: jnp.ndarray) -> jnp.ndarray:
    channels = image.shape[-1]
    kernel = jnp.tile(kernel_2d[:, :, None, None], (1, 1, 1, channels))
    return jax.lax.conv_general_dilated(
        image,
        kernel,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        feature_group_count=channels,
    )


def laplacian(image: jnp.ndarray) -> jnp.ndarray:
    kernel = jnp.asarray(
        [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=jnp.float32,
    )
    return _depthwise_filter(image.astype(jnp.float32), kernel)


def detail_charbonnier(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    return charbonnier(laplacian(pred), laplacian(target))


def spectrum_loss(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    pred = jnp.clip(pred, 0.0, 1.0).astype(jnp.float32)
    target = jnp.clip(target, 0.0, 1.0).astype(jnp.float32)
    total = jnp.asarray(0.0, dtype=jnp.float32)
    weight = jnp.asarray(1.0, dtype=jnp.float32)
    for _ in range(3):
        pred_low = _filter_image(pred)
        target_low = _filter_image(target)
        total = total + weight * charbonnier(pred - pred_low, target - target_low)
        pred = avg_pool_2x(pred_low)
        target = avg_pool_2x(target_low)
        weight = weight * 0.5
    return total


def cubic_base_batch(x: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    if x.shape[1:] == target.shape[1:]:
        return x.astype(jnp.float32)
    return jax.image.resize(
        x.astype(jnp.float32),
        (
            x.shape[0],
            target.shape[1],
            target.shape[2],
            x.shape[3],
        ),
        method="cubic",
    )


def base_mae(pred: jnp.ndarray, base: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean(jnp.abs(jnp.clip(pred, 0.0, 1.0) - jnp.clip(base, 0.0, 1.0)))


def base_divergence_loss(
    pred: jnp.ndarray,
    base: jnp.ndarray,
    floor: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    mae = base_mae(pred, base)
    return jnp.square(jnp.maximum(floor - mae, 0.0)), mae


def _gaussian_kernel(channels: int, size: int = 11, sigma: float = 1.5) -> jnp.ndarray:
    coords = jnp.arange(size, dtype=jnp.float32) - size // 2
    kernel_1d = jnp.exp(-(coords**2) / (2.0 * sigma**2))
    kernel_1d = kernel_1d / jnp.sum(kernel_1d)
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    return jnp.tile(kernel_2d[:, :, None, None], (1, 1, 1, channels))


def _filter_image(image: jnp.ndarray) -> jnp.ndarray:
    channels = image.shape[-1]
    kernel = _gaussian_kernel(channels)
    return jax.lax.conv_general_dilated(
        image,
        kernel,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        feature_group_count=channels,
    )


def ssim_and_cs(pred: jnp.ndarray, target: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    pred = jnp.clip(pred, 0.0, 1.0).astype(jnp.float32)
    target = jnp.clip(target, 0.0, 1.0).astype(jnp.float32)
    c1 = 0.01**2
    c2 = 0.03**2

    mu_x = _filter_image(pred)
    mu_y = _filter_image(target)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = jnp.maximum(_filter_image(pred * pred) - mu_x2, 0.0)
    sigma_y2 = jnp.maximum(_filter_image(target * target) - mu_y2, 0.0)
    sigma_xy = _filter_image(pred * target) - mu_xy

    luminance = (2.0 * mu_xy + c1) / (mu_x2 + mu_y2 + c1)
    contrast_structure = (2.0 * sigma_xy + c2) / (sigma_x2 + sigma_y2 + c2)
    ssim_map = luminance * contrast_structure
    return jnp.mean(ssim_map), jnp.mean(contrast_structure)


def avg_pool_2x(image: jnp.ndarray) -> jnp.ndarray:
    pooled = jax.lax.reduce_window(
        image,
        init_value=0.0,
        computation=jax.lax.add,
        window_dimensions=(1, 2, 2, 1),
        window_strides=(1, 2, 2, 1),
        padding="VALID",
    )
    return pooled * 0.25


def ms_ssim(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    weights = jnp.asarray([0.0448, 0.2856, 0.3001, 0.2363, 0.1333], dtype=jnp.float32)
    pred = jnp.clip(pred, 0.0, 1.0).astype(jnp.float32)
    target = jnp.clip(target, 0.0, 1.0).astype(jnp.float32)
    cs_values = []
    for _ in range(4):
        _, cs = ssim_and_cs(pred, target)
        cs_values.append(jnp.clip(cs, 1e-6, 1.0))
        pred = avg_pool_2x(pred)
        target = avg_pool_2x(target)
    final_ssim, _ = ssim_and_cs(pred, target)
    values = jnp.stack(cs_values + [jnp.clip(final_ssim, 1e-6, 1.0)])
    return jnp.prod(values**weights)


def rgb_to_lab(rgb: jnp.ndarray) -> jnp.ndarray:
    rgb = jnp.clip(rgb, 0.0, 1.0).astype(jnp.float32)
    linear = jnp.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    matrix = jnp.asarray(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=jnp.float32,
    )
    xyz = jnp.einsum("...c,dc->...d", linear, matrix)
    white = jnp.asarray([0.95047, 1.0, 1.08883], dtype=jnp.float32)
    xyz = xyz / white

    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    f = jnp.where(xyz > epsilon, xyz ** (1.0 / 3.0), (kappa * xyz + 16.0) / 116.0)
    l = 116.0 * f[..., 1] - 16.0
    a = 500.0 * (f[..., 0] - f[..., 1])
    b = 200.0 * (f[..., 1] - f[..., 2])
    return jnp.stack([l, a, b], axis=-1)


def color_delta_e(pred: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    pred_lab = rgb_to_lab(pred)
    target_lab = rgb_to_lab(target)
    return jnp.mean(jnp.sqrt(jnp.sum((pred_lab - target_lab) ** 2, axis=-1) + 1e-6))


def quality_metrics(pred: jnp.ndarray, target: jnp.ndarray) -> dict[str, jnp.ndarray]:
    ssim_value, _ = ssim_and_cs(pred, target)
    return {
        "ssim": ssim_value,
        "ms_ssim": ms_ssim(pred, target),
        "color_delta_e": color_delta_e(pred, target),
    }


def resolve_input_mode(args: argparse.Namespace, config: ModelConfig) -> str:
    if args.input_mode != "auto":
        return args.input_mode
    return "bicubic_up" if config.model == "edsr_lite" else "lr"


def build_model_config(args: argparse.Namespace) -> ModelConfig:
    config = ModelConfig(
        model=args.model,
        scale=args.scale,
        features=args.features,
        blocks=args.blocks,
        groups=args.groups,
        heads=args.heads,
        window_size=args.window_size,
        mlp_ratio=args.mlp_ratio,
        token_count=args.token_count,
        channel_reduction=args.channel_reduction,
        residual_scale=args.residual_scale,
        dtype=args.dtype,
        global_residual=not args.no_global_residual,
    )
    return apply_preset(config, args.model_preset)


def validate_args(args: argparse.Namespace, config: ModelConfig, input_mode: str) -> None:
    if args.crop_size % args.scale:
        raise ValueError(f"crop-size must be divisible by scale, got {args.crop_size=} {args.scale=}")
    if args.edge_loss_weight < 0:
        raise ValueError(f"edge-loss-weight must be >= 0, got {args.edge_loss_weight}")
    if args.detail_loss_weight < 0:
        raise ValueError(f"detail-loss-weight must be >= 0, got {args.detail_loss_weight}")
    if args.spectrum_loss_weight < 0:
        raise ValueError(f"spectrum-loss-weight must be >= 0, got {args.spectrum_loss_weight}")
    if args.base_divergence_weight < 0:
        raise ValueError(
            f"base-divergence-weight must be >= 0, got {args.base_divergence_weight}"
        )
    if args.base_divergence_floor < 0:
        raise ValueError(f"base-divergence-floor must be >= 0, got {args.base_divergence_floor}")
    if args.skip_grad_norm < 0:
        raise ValueError(f"skip-grad-norm must be >= 0, got {args.skip_grad_norm}")
    if args.skip_loss_threshold < 0:
        raise ValueError(f"skip-loss-threshold must be >= 0, got {args.skip_loss_threshold}")
    if args.net_perceptual_every_epochs < 0:
        raise ValueError(
            "net-perceptual-every-epochs must be >= 0, "
            f"got {args.net_perceptual_every_epochs}"
        )
    if input_mode == "lr":
        lr_size = args.crop_size // args.scale
        if config.model == "hat_atd" and lr_size % config.window_size:
            raise ValueError(
                "For HAT/ATD training, crop_size / scale must be divisible by window_size. "
                f"got crop_size={args.crop_size} scale={args.scale} lr_size={lr_size} "
                f"window_size={config.window_size}"
            )
    if config.features % config.heads:
        raise ValueError(f"features must be divisible by heads, got {config.features=} {config.heads=}")


def make_optimizer(args: argparse.Namespace):
    warmup_steps = min(args.warmup_steps, max(args.steps - 1, 0))
    schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=args.lr,
        warmup_steps=warmup_steps,
        decay_steps=max(args.steps, warmup_steps + 1),
        end_value=args.lr * args.min_lr_ratio,
    )
    transforms = []
    if args.grad_clip_norm > 0:
        transforms.append(optax.clip_by_global_norm(args.grad_clip_norm))
    transforms.append(optax.adamw(schedule, weight_decay=args.weight_decay))
    return optax.chain(*transforms), schedule


def create_state(
    args: argparse.Namespace,
    model,
    *,
    input_shape: tuple[int, int, int, int] | None = None,
    tx=None,
) -> TrainState:
    key = jax.random.PRNGKey(args.seed)
    if input_shape is None:
        input_mode = getattr(args, "input_mode", "bicubic_up")
        input_size = args.crop_size if input_mode == "bicubic_up" else args.crop_size // args.scale
        input_shape = (args.batch_size, input_size, input_size, 3)
    variables = model.init(key, jnp.ones(input_shape, jnp.float32))
    if tx is None:
        tx, _ = make_optimizer(args)
    return TrainState.create(apply_fn=model.apply, params=variables["params"], tx=tx)


def count_params(params: Any) -> int:
    return int(sum(np.prod(leaf.shape) for leaf in jax.tree_util.tree_leaves(params)))


@jax.jit
def train_step(
    state: TrainState,
    batch: tuple[jnp.ndarray, jnp.ndarray],
    edge_loss_weight: jnp.ndarray,
    detail_loss_weight: jnp.ndarray,
    spectrum_loss_weight: jnp.ndarray,
    base_divergence_weight: jnp.ndarray,
    base_divergence_floor: jnp.ndarray,
    skip_grad_norm: jnp.ndarray,
    skip_loss_threshold: jnp.ndarray,
):
    x, y = batch
    base = jax.lax.stop_gradient(cubic_base_batch(x, y))

    def loss_fn(params):
        pred = state.apply_fn({"params": params}, x)
        pixel_loss = charbonnier(pred, y)
        edge_loss = jax.lax.cond(
            edge_loss_weight > 0,
            lambda _: edge_charbonnier(pred, y),
            lambda _: jnp.asarray(0.0, dtype=pred.dtype),
            operand=None,
        )
        detail_loss = jax.lax.cond(
            detail_loss_weight > 0,
            lambda _: detail_charbonnier(pred, y),
            lambda _: jnp.asarray(0.0, dtype=pred.dtype),
            operand=None,
        )
        spectral_loss = jax.lax.cond(
            spectrum_loss_weight > 0,
            lambda _: spectrum_loss(pred, y),
            lambda _: jnp.asarray(0.0, dtype=pred.dtype),
            operand=None,
        )
        collapse_loss, pred_base_mae = base_divergence_loss(pred, base, base_divergence_floor)
        loss = (
            pixel_loss
            + edge_loss_weight * edge_loss
            + detail_loss_weight * detail_loss
            + spectrum_loss_weight * spectral_loss
            + base_divergence_weight * collapse_loss
        )
        return loss, (
            pred,
            pixel_loss,
            edge_loss,
            detail_loss,
            spectral_loss,
            collapse_loss,
            pred_base_mae,
        )

    (
        loss,
        (pred, pixel_loss, edge_loss, detail_loss, spectral_loss, collapse_loss, pred_base_mae),
    ), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    grad_norm = optax.global_norm(grads)
    skip_for_grad = jnp.logical_and(skip_grad_norm > 0, grad_norm > skip_grad_norm)
    skip_for_loss = jnp.logical_and(skip_loss_threshold > 0, loss > skip_loss_threshold)
    skip_update = jnp.logical_or(
        jnp.logical_or(skip_for_grad, skip_for_loss),
        jnp.logical_not(jnp.logical_and(jnp.isfinite(loss), jnp.isfinite(grad_norm))),
    )
    updated_state = state.apply_gradients(grads=grads)
    state = jax.lax.cond(skip_update, lambda _: state, lambda _: updated_state, operand=None)
    return state, {
        "loss": loss,
        "pixel_loss": pixel_loss,
        "edge_loss": edge_loss,
        "detail_loss": detail_loss,
        "spectrum_loss": spectral_loss,
        "collapse_loss": collapse_loss,
        "base_mae": pred_base_mae,
        "psnr": psnr(pred, y),
        "grad_norm": grad_norm,
        "skip_update": skip_update.astype(jnp.float32),
        "param_norm": optax.global_norm(state.params),
    }


@jax.jit
def eval_step(
    state: TrainState,
    batch: tuple[jnp.ndarray, jnp.ndarray],
    edge_loss_weight: jnp.ndarray,
    detail_loss_weight: jnp.ndarray,
    spectrum_loss_weight: jnp.ndarray,
    base_divergence_weight: jnp.ndarray,
    base_divergence_floor: jnp.ndarray,
):
    x, y = batch
    base = cubic_base_batch(x, y)
    pred = state.apply_fn({"params": state.params}, x)
    pixel_loss = charbonnier(pred, y)
    edge_loss = jax.lax.cond(
        edge_loss_weight > 0,
        lambda _: edge_charbonnier(pred, y),
        lambda _: jnp.asarray(0.0, dtype=pred.dtype),
        operand=None,
    )
    detail_loss = jax.lax.cond(
        detail_loss_weight > 0,
        lambda _: detail_charbonnier(pred, y),
        lambda _: jnp.asarray(0.0, dtype=pred.dtype),
        operand=None,
    )
    spectral_loss = jax.lax.cond(
        spectrum_loss_weight > 0,
        lambda _: spectrum_loss(pred, y),
        lambda _: jnp.asarray(0.0, dtype=pred.dtype),
        operand=None,
    )
    collapse_loss, pred_base_mae = base_divergence_loss(pred, base, base_divergence_floor)
    return {
        "loss": (
            pixel_loss
            + edge_loss_weight * edge_loss
            + detail_loss_weight * detail_loss
            + spectrum_loss_weight * spectral_loss
            + base_divergence_weight * collapse_loss
        ),
        "pixel_loss": pixel_loss,
        "edge_loss": edge_loss,
        "detail_loss": detail_loss,
        "spectrum_loss": spectral_loss,
        "collapse_loss": collapse_loss,
        "base_mae": pred_base_mae,
        "psnr": psnr(pred, y),
        **quality_metrics(pred, y),
    }


@jax.jit
def predict_step(state: TrainState, x: jnp.ndarray):
    return jnp.clip(state.apply_fn({"params": state.params}, x), 0.0, 1.0)


@jax.jit
def reference_quality_step(pred: jnp.ndarray, target: jnp.ndarray):
    return quality_metrics(pred, target)


def to_uint8(image: np.ndarray) -> np.ndarray:
    return np.asarray(np.clip(image, 0.0, 1.0) * 255.0 + 0.5, dtype=np.uint8)


def resize_array(image: np.ndarray, size: tuple[int, int], resample=Image.Resampling.BICUBIC) -> np.ndarray:
    return np.asarray(Image.fromarray(to_uint8(image)).resize(size, resample), dtype=np.uint8)


def psnr_np(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    mse = np.mean((np.clip(pred, 0.0, 1.0) - target) ** 2, axis=(1, 2, 3))
    return -10.0 * np.log10(np.maximum(mse, 1e-10))


def bicubic_baseline_batch(
    x_np: np.ndarray,
    y_np: np.ndarray,
    *,
    input_mode: str,
) -> np.ndarray:
    if input_mode == "bicubic_up":
        return np.clip(x_np, 0.0, 1.0)

    upscaled = []
    for index in range(x_np.shape[0]):
        target = y_np[index]
        upscaled.append(
            resize_array(
                x_np[index],
                (target.shape[1], target.shape[0]),
                Image.Resampling.BICUBIC,
            ).astype(np.float32)
            / 255.0
        )
    return np.stack(upscaled)


def comparison_strip(images: list[np.ndarray]) -> np.ndarray:
    heights = [image.shape[0] for image in images]
    max_height = max(heights)
    normalized = []
    for image in images:
        if image.shape[0] == max_height:
            normalized.append(image)
            continue
        width = round(image.shape[1] * max_height / image.shape[0])
        normalized.append(np.asarray(Image.fromarray(image).resize((width, max_height), Image.Resampling.BICUBIC)))
    return np.concatenate(normalized, axis=1)


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    if width <= 0 or image.shape[1] == width:
        return image
    height = max(1, round(image.shape[0] * width / image.shape[1]))
    return np.asarray(Image.fromarray(image).resize((width, height), Image.Resampling.BICUBIC))


def labeled_cell(image: np.ndarray, label: str, *, width: int, header: int = 26) -> np.ndarray:
    image = resize_to_width(image, width)
    canvas = np.full((image.shape[0] + header, image.shape[1], 3), 248, dtype=np.uint8)
    canvas[header:] = image
    pil = Image.fromarray(canvas)
    draw = ImageDraw.Draw(pil)
    draw.text((6, 6), label, fill=(25, 25, 25))
    return np.asarray(pil)


def make_contact_sheet(
    rows: list[tuple[str, list[np.ndarray]]],
    *,
    column_labels: list[str],
    cell_width: int,
    gutter: int = 8,
) -> np.ndarray:
    if not rows:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    rendered_rows = []
    for row_label, images in rows:
        cells = []
        for index, image in enumerate(images):
            column = column_labels[index] if index < len(column_labels) else f"col {index + 1}"
            label = f"{row_label} - {column}" if index == 0 else column
            cells.append(labeled_cell(image, label, width=cell_width))

        row_height = max(cell.shape[0] for cell in cells)
        padded_cells = []
        for cell in cells:
            padded = np.full((row_height, cell.shape[1], 3), 255, dtype=np.uint8)
            padded[: cell.shape[0], : cell.shape[1]] = cell
            padded_cells.append(padded)
            padded_cells.append(np.full((row_height, gutter, 3), 255, dtype=np.uint8))
        rendered_rows.append(np.concatenate(padded_cells[:-1], axis=1))

    sheet_width = max(row.shape[1] for row in rendered_rows)
    padded_rows = []
    for row in rendered_rows:
        padded = np.full((row.shape[0], sheet_width, 3), 255, dtype=np.uint8)
        padded[: row.shape[0], : row.shape[1]] = row
        padded_rows.append(padded)
        padded_rows.append(np.full((gutter, sheet_width, 3), 255, dtype=np.uint8))
    return np.concatenate(padded_rows[:-1], axis=0)


def diff_heatmap(
    a: np.ndarray,
    b: np.ndarray,
    *,
    gain: float,
) -> np.ndarray:
    diff = np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)), axis=-1)
    heat = np.clip(diff * gain, 0.0, 255.0).astype(np.uint8)
    return np.stack([heat, np.zeros_like(heat), 255 - heat], axis=-1)


def pad_images_to_common_canvas(images: list[np.ndarray]) -> list[np.ndarray]:
    if not images:
        return []
    max_height = max(image.shape[0] for image in images)
    max_width = max(image.shape[1] for image in images)
    padded = []
    for image in images:
        canvas = np.full((max_height, max_width, 3), 255, dtype=np.uint8)
        canvas[: image.shape[0], : image.shape[1]] = image
        padded.append(canvas)
    return padded


def make_eval_image_logs(
    state: TrainState,
    eval_image_batches: list[tuple[np.ndarray, np.ndarray]],
    *,
    count: int,
    scale: int,
    input_mode: str,
) -> Any | None:
    if count <= 0 or not eval_image_batches:
        return None
    import wandb

    rows = []
    remaining = count
    image_index = 1
    for x_np, y_np in eval_image_batches:
        if remaining <= 0:
            break
        pred_np = np.asarray(predict_step(state, jnp.asarray(x_np)))
        batch = min(remaining, x_np.shape[0])
        for index in range(batch):
            target = to_uint8(y_np[index])
            pred = to_uint8(pred_np[index])
            if input_mode == "lr":
                lr_up = resize_array(
                    x_np[index],
                    (target.shape[1], target.shape[0]),
                    Image.Resampling.BICUBIC,
                )
            else:
                lr_up = to_uint8(x_np[index])
            diff = to_uint8(np.abs(pred_np[index] - y_np[index]) * 4.0)
            rows.append((f"val {image_index:02d}", [lr_up, pred, target, diff]))
            image_index += 1
        remaining -= batch
    sheet = make_contact_sheet(
        rows,
        column_labels=["input", "pred", "target", "absdiff x4"],
        cell_width=256,
    )
    return wandb.Image(sheet, caption="fixed validation: input | pred | target | absdiff x4")


def make_fixed_eval_image_batches(
    val_data: Path,
    *,
    batch_size: int,
    crop_size: int,
    scale: int,
    seed: int,
    input_mode: str,
    count: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    if count <= 0:
        return []
    batches_needed = int(np.ceil(count / batch_size))
    fixed_batches = batch_iterator(
        val_data,
        batch_size=batch_size,
        crop_size=crop_size,
        scale=scale,
        seed=seed,
        input_mode=input_mode,
        degradation="bicubic",
        augment=False,
        num_workers=0,
        prefetch_batches=1,
        preload=False,
        preload_progress=False,
    )
    return [next(fixed_batches) for _ in range(batches_needed)]


class TorchPerceptualEvaluator:
    def __init__(self, mode: str):
        self.mode = mode
        self.torch = None
        self.device = None
        self.lpips = None
        self.dists = None
        if mode == "off":
            return

        try:
            import torch

            self.torch = torch
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        except Exception as exc:
            print(
                "WARNING: --net-perceptual-metrics requested but torch is unavailable: "
                f"{exc}. Skipping LPIPS/DISTS.",
                flush=True,
            )
            return

        if mode in {"lpips", "all"}:
            try:
                import lpips

                self.lpips = lpips.LPIPS(net="alex").to(self.device).eval()
            except Exception as exc:
                print(f"WARNING: LPIPS unavailable, skipping it: {exc}", flush=True)

        if mode in {"dists", "all"}:
            try:
                from piq import DISTS

                self.dists = DISTS(reduction="mean").to(self.device).eval()
            except Exception as exc:
                print(f"WARNING: DISTS unavailable, skipping it: {exc}", flush=True)

    @property
    def available(self) -> bool:
        return self.lpips is not None or self.dists is not None

    def _to_tensor(self, images: np.ndarray):
        assert self.torch is not None
        tensor = self.torch.from_numpy(np.asarray(images, dtype=np.float32))
        tensor = tensor.permute(0, 3, 1, 2).contiguous()
        return tensor.to(self.device)

    def compute(self, pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
        if not self.available or self.torch is None:
            return {}
        pred_tensor = self._to_tensor(np.clip(pred, 0.0, 1.0))
        target_tensor = self._to_tensor(np.clip(target, 0.0, 1.0))
        logs: dict[str, float] = {}
        with self.torch.no_grad():
            if self.lpips is not None:
                value = self.lpips(pred_tensor * 2.0 - 1.0, target_tensor * 2.0 - 1.0)
                logs["eval/lpips_fixed"] = float(value.mean().detach().cpu())
            if self.dists is not None:
                value = self.dists(pred_tensor, target_tensor)
                logs["eval/dists_fixed"] = float(value.mean().detach().cpu())
        return logs


def make_net_perceptual_logs(
    state: TrainState,
    eval_image_batches: list[tuple[np.ndarray, np.ndarray]],
    evaluator: TorchPerceptualEvaluator,
) -> dict[str, float]:
    if not evaluator.available or not eval_image_batches:
        return {}
    preds = []
    targets = []
    for x_np, y_np in eval_image_batches:
        preds.append(np.asarray(predict_step(state, jnp.asarray(x_np))))
        targets.append(y_np)
    return evaluator.compute(np.concatenate(preds, axis=0), np.concatenate(targets, axis=0))


def list_sample_images(sample_dir: str | Path, limit: int) -> list[Path]:
    root = Path(sample_dir).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    if not root.exists() or limit <= 0:
        return []
    images = [
        path
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return images[:limit]


def resize_sample(image: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return image
    width, height = image.size
    longest = max(width, height)
    if longest <= max_side:
        return image
    scale = max_side / longest
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(size, Image.Resampling.LANCZOS)


def pad_to_multiple(array: np.ndarray, multiple: int) -> tuple[np.ndarray, tuple[int, int]]:
    height, width = array.shape[:2]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return array, (height, width)
    padded = np.pad(array, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
    return padded, (height, width)


def prepare_sample_input(
    image: Image.Image,
    *,
    scale: int,
    input_mode: str,
    window_size: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    array = np.asarray(image, dtype=np.float32) / 255.0
    if input_mode == "lr":
        array, original_hw = pad_to_multiple(array, window_size)
        return array[None, ...], original_hw

    width, height = image.size
    upscaled = image.resize((width * scale, height * scale), Image.Resampling.BICUBIC)
    array = np.asarray(upscaled, dtype=np.float32) / 255.0
    return array[None, ...], (height * scale, width * scale)


def save_and_log_user_samples(
    state: TrainState,
    sample_paths: list[Path],
    *,
    out_dir: Path,
    step: int,
    epoch: int,
    scale: int,
    input_mode: str,
    window_size: int,
    max_side: int,
    run,
) -> None:
    if not sample_paths:
        return
    wandb = None
    if run is not None:
        import wandb as wandb_module

        wandb = wandb_module

    output_dir = out_dir / "usr_samples" / f"epoch_{epoch:06d}_step_{step:08d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_rows = []
    sample_metrics: dict[str, float] = {}
    for path in sample_paths:
        image = resize_sample(ImageOps.exif_transpose(Image.open(path)).convert("RGB"), max_side)
        x_np, original_hw = prepare_sample_input(
            image,
            scale=scale,
            input_mode=input_mode,
            window_size=window_size,
        )
        pred = np.asarray(predict_step(state, jnp.asarray(x_np)))[0]
        if input_mode == "lr":
            height, width = original_hw
            pred = pred[: height * scale, : width * scale]
            bicubic = image.resize((width * scale, height * scale), Image.Resampling.BICUBIC)
        else:
            height, width = original_hw
            pred = pred[:height, :width]
            bicubic = image.resize((width, height), Image.Resampling.BICUBIC)

        pred_u8 = to_uint8(pred)
        output_path = output_dir / f"{path.stem}_x{scale}.png"
        Image.fromarray(pred_u8).save(output_path)

        bicubic_u8 = np.asarray(bicubic)
        residual_x16 = diff_heatmap(pred_u8, bicubic_u8, gain=16.0)
        residual_x64 = diff_heatmap(pred_u8, bicubic_u8, gain=64.0)
        sample_rows.append((path.stem, [bicubic_u8, pred_u8, residual_x16, residual_x64]))
        if wandb is not None:
            residual = np.abs(pred_u8.astype(np.float32) - bicubic_u8.astype(np.float32))
            sample_metrics.update(
                {
                    f"samples/{path.stem}_pred_minus_bicubic_mae": float(residual.mean()),
                    f"samples/{path.stem}_pred_minus_bicubic_p99": float(
                        np.percentile(residual, 99)
                    ),
                    f"samples/{path.stem}_pred_minus_bicubic_max": float(residual.max()),
                }
            )

    if sample_rows:
        sheet = make_contact_sheet(
            sample_rows,
            column_labels=["bicubic", "pred", "delta x16", "delta x64"],
            cell_width=360,
        )
        Image.fromarray(sheet).save(output_dir / "_contact_sheet.jpg", quality=92)

    if run is not None and sample_rows:
        sample_metrics.update(
            {
                "train/step": step,
                "samples/usr_samples": wandb.Image(
                    sheet,
                    caption=(
                        f"usr_samples epoch {epoch} step {step}: "
                        "bicubic | pred | abs(pred-bicubic) x16 | x64"
                    ),
                )
            }
        )
        run.log(sample_metrics, step=step)


def _has_wandb_login() -> bool:
    if os.environ.get("WANDB_API_KEY"):
        return True
    netrc = Path("~/.netrc").expanduser()
    if not netrc.exists():
        return False
    try:
        return "api.wandb.ai" in netrc.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False


def init_wandb(args: argparse.Namespace, metadata: dict[str, Any]):
    if not args.wandb or args.wandb_mode == "disabled":
        return None
    if args.wandb_quiet:
        os.environ.setdefault("WANDB_SILENT", "true")
        os.environ.setdefault("WANDB_CONSOLE", "off")
    import wandb

    mode = args.wandb_mode
    if mode == "auto":
        mode = "online" if _has_wandb_login() else "offline"
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name or None,
        config=metadata,
        mode=mode,
        tags=[tag.strip() for tag in args.wandb_tags.split(",") if tag.strip()],
    )
    wandb.define_metric("train/step")
    wandb.define_metric("train/*", step_metric="train/step")
    wandb.define_metric("eval/*", step_metric="train/step")
    wandb.define_metric("samples/*", step_metric="train/step")
    return run


def use_progress_bar(args: argparse.Namespace) -> bool:
    if args.progress == "on":
        return True
    if args.progress == "off":
        return False
    return os.isatty(1)


def log_console(message: str, progress=None) -> None:
    if progress is not None:
        progress.write(message)
    else:
        print(message, flush=True)


def format_metrics(step: int, logs: dict[str, float]) -> str:
    return (
        f"step={step} loss={logs['train/loss']:.5f} "
        f"edge={logs['train/edge_loss']:.5f} "
        f"detail={logs['train/detail_loss']:.5f} "
        f"base_mae={logs['train/base_mae'] * 255.0:.2f}/255 "
        f"psnr={logs['train/psnr']:.2f} lr={logs['train/lr']:.2e} "
        f"grad={logs['train/grad_norm']:.2f} skip={logs['train/skip_update']:.0f} "
        f"steps_per_sec={logs['train/steps_per_sec']:.2f}"
    )


def maybe_eval(
    state: TrainState,
    eval_batches,
    *,
    eval_batch_count: int,
    edge_loss_weight: jnp.ndarray,
    detail_loss_weight: jnp.ndarray,
    spectrum_loss_weight: jnp.ndarray,
    base_divergence_weight: jnp.ndarray,
    base_divergence_floor: jnp.ndarray,
    input_mode: str,
) -> dict[str, float]:
    losses = []
    pixel_losses = []
    edge_losses = []
    detail_losses = []
    spectrum_losses = []
    collapse_losses = []
    base_maes = []
    psnrs = []
    ssims = []
    ms_ssims = []
    color_delta_es = []
    bicubic_psnrs = []
    bicubic_losses = []
    bicubic_ssims = []
    bicubic_ms_ssims = []
    bicubic_color_delta_es = []
    for _ in range(eval_batch_count):
        x_np, y_np = next(eval_batches)
        metrics = eval_step(
            state,
            (jnp.asarray(x_np), jnp.asarray(y_np)),
            edge_loss_weight,
            detail_loss_weight,
            spectrum_loss_weight,
            base_divergence_weight,
            base_divergence_floor,
        )
        losses.append(float(metrics["loss"]))
        pixel_losses.append(float(metrics["pixel_loss"]))
        edge_losses.append(float(metrics["edge_loss"]))
        detail_losses.append(float(metrics["detail_loss"]))
        spectrum_losses.append(float(metrics["spectrum_loss"]))
        collapse_losses.append(float(metrics["collapse_loss"]))
        base_maes.append(float(metrics["base_mae"]))
        psnrs.append(float(metrics["psnr"]))
        ssims.append(float(metrics["ssim"]))
        ms_ssims.append(float(metrics["ms_ssim"]))
        color_delta_es.append(float(metrics["color_delta_e"]))
        bicubic = bicubic_baseline_batch(x_np, y_np, input_mode=input_mode)
        bicubic_psnrs.extend(psnr_np(bicubic, y_np).tolist())
        bicubic_losses.append(float(np.mean(np.sqrt((bicubic - y_np) ** 2 + 1e-6))))
        bicubic_quality = reference_quality_step(jnp.asarray(bicubic), jnp.asarray(y_np))
        bicubic_ssims.append(float(bicubic_quality["ssim"]))
        bicubic_ms_ssims.append(float(bicubic_quality["ms_ssim"]))
        bicubic_color_delta_es.append(float(bicubic_quality["color_delta_e"]))
    mean_psnr = float(np.mean(psnrs))
    mean_bicubic_psnr = float(np.mean(bicubic_psnrs))
    mean_ssim = float(np.mean(ssims))
    mean_ms_ssim = float(np.mean(ms_ssims))
    mean_color_delta_e = float(np.mean(color_delta_es))
    mean_bicubic_ssim = float(np.mean(bicubic_ssims))
    mean_bicubic_ms_ssim = float(np.mean(bicubic_ms_ssims))
    mean_bicubic_color_delta_e = float(np.mean(bicubic_color_delta_es))
    return {
        "eval/loss": float(np.mean(losses)),
        "eval/pixel_loss": float(np.mean(pixel_losses)),
        "eval/edge_loss": float(np.mean(edge_losses)),
        "eval/detail_loss": float(np.mean(detail_losses)),
        "eval/spectrum_loss": float(np.mean(spectrum_losses)),
        "eval/collapse_loss": float(np.mean(collapse_losses)),
        "eval/base_mae": float(np.mean(base_maes)),
        "eval/psnr": mean_psnr,
        "eval/ssim": mean_ssim,
        "eval/ms_ssim": mean_ms_ssim,
        "eval/color_delta_e": mean_color_delta_e,
        "eval/bicubic_loss": float(np.mean(bicubic_losses)),
        "eval/bicubic_psnr": mean_bicubic_psnr,
        "eval/bicubic_ssim": mean_bicubic_ssim,
        "eval/bicubic_ms_ssim": mean_bicubic_ms_ssim,
        "eval/bicubic_color_delta_e": mean_bicubic_color_delta_e,
        "eval/psnr_gain_vs_bicubic": mean_psnr - mean_bicubic_psnr,
        "eval/ssim_gain_vs_bicubic": mean_ssim - mean_bicubic_ssim,
        "eval/ms_ssim_gain_vs_bicubic": mean_ms_ssim - mean_bicubic_ms_ssim,
        "eval/color_delta_e_improvement_vs_bicubic": (
            mean_bicubic_color_delta_e - mean_color_delta_e
        ),
    }


def resolve_epoch_steps(args: argparse.Namespace) -> tuple[int, int]:
    train_images = count_training_items(args.data)
    if args.steps_per_epoch > 0:
        return args.steps_per_epoch, train_images
    return max(1, int(np.ceil(train_images / args.batch_size))), train_images


def should_log_for_epoch(
    *,
    step: int,
    steps_per_epoch: int,
    every_epochs: int,
) -> bool:
    if every_epochs <= 0:
        return False
    if step % steps_per_epoch:
        return False
    epoch = step // steps_per_epoch
    return epoch % every_epochs == 0


def main() -> None:
    logging.getLogger("absl").setLevel(logging.ERROR)
    args = parse_args()
    config = build_model_config(args)
    input_mode = resolve_input_mode(args, config)
    args.input_mode = input_mode
    validate_args(args, config, input_mode)

    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    model = create_model(config)
    tx, lr_schedule = make_optimizer(args)
    state = create_state(args, model, tx=tx)
    edge_loss_weight = jnp.asarray(args.edge_loss_weight, dtype=jnp.float32)
    detail_loss_weight = jnp.asarray(args.detail_loss_weight, dtype=jnp.float32)
    spectrum_loss_weight = jnp.asarray(args.spectrum_loss_weight, dtype=jnp.float32)
    base_divergence_weight = jnp.asarray(args.base_divergence_weight, dtype=jnp.float32)
    base_divergence_floor = jnp.asarray(args.base_divergence_floor, dtype=jnp.float32)
    skip_grad_norm = jnp.asarray(args.skip_grad_norm, dtype=jnp.float32)
    skip_loss_threshold = jnp.asarray(args.skip_loss_threshold, dtype=jnp.float32)
    if args.resume:
        state = checkpoints.restore_checkpoint(Path(args.resume).expanduser().resolve(), state)
        if args.reset_optimizer_on_resume:
            state = state.replace(step=0, opt_state=tx.init(state.params))
    elif any(out_dir.glob("checkpoint_*")):
        state = checkpoints.restore_checkpoint(out_dir, state)

    steps_per_epoch, train_image_count = resolve_epoch_steps(args)
    sample_paths = list_sample_images(args.sample_dir, args.sample_max_images)

    train_batches = batch_iterator(
        args.data,
        batch_size=args.batch_size,
        crop_size=args.crop_size,
        scale=args.scale,
        seed=args.seed,
        input_mode=input_mode,
        degradation=args.degradation,
        augment=True,
        num_workers=args.data_workers,
        prefetch_batches=args.prefetch_batches,
        preload=args.preload_data,
        preload_progress=True,
    )

    eval_batches = None
    val_data = Path(args.val_data).expanduser() if args.val_data else None
    if val_data and val_data.exists() and args.eval_every > 0 and args.eval_batches > 0:
        eval_batches = batch_iterator(
            val_data,
            batch_size=args.batch_size,
            crop_size=args.crop_size,
            scale=args.scale,
            seed=args.seed + 100_003,
            input_mode=input_mode,
            degradation="bicubic",
            augment=False,
            num_workers=args.eval_data_workers,
            prefetch_batches=max(1, args.prefetch_batches // 2),
            preload=args.preload_data,
            preload_progress=args.eval_data_workers > 0,
        )
    eval_image_batches = []
    fixed_eval_needed = (
        args.image_log_every_epochs > 0
        or (args.net_perceptual_metrics != "off" and args.net_perceptual_every_epochs > 0)
    )
    if val_data and val_data.exists() and fixed_eval_needed and args.image_log_count > 0:
        eval_image_batches = make_fixed_eval_image_batches(
            val_data,
            batch_size=args.batch_size,
            crop_size=args.crop_size,
            scale=args.scale,
            seed=args.seed + 200_003,
            input_mode=input_mode,
            count=args.image_log_count,
        )

    first_train_batch = next(train_batches)
    net_perceptual = TorchPerceptualEvaluator(args.net_perceptual_metrics)

    metadata = {
        "args": vars(args),
        "model_config": asdict(config),
        "input_mode": input_mode,
        "param_count": count_params(state.params),
        "train_image_count": train_image_count,
        "train_data_layout": "pairs" if is_pair_dataset(args.data) else "hr",
        "val_data_layout": (
            "pairs"
            if args.val_data and Path(args.val_data).expanduser().exists() and is_pair_dataset(args.val_data)
            else "hr"
        ),
        "steps_per_epoch": steps_per_epoch,
        "sample_paths": [str(path) for path in sample_paths],
        "data_workers": args.data_workers,
        "eval_data_workers": args.eval_data_workers,
        "prefetch_batches": args.prefetch_batches,
        "preload_data": args.preload_data,
        "fixed_eval_image_count": min(args.image_log_count, len(eval_image_batches) * args.batch_size),
        "net_perceptual_available": net_perceptual.available,
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
    }
    (out_dir / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    run = init_wandb(args, metadata)

    print(
        "model={model} preset={preset} params={params:,} input_mode={input_mode} "
        "data={data} layout={layout} out={out} steps_per_epoch={steps_per_epoch} samples={samples} "
        "workers={workers} preload={preload} degradation={degradation} "
        "global_residual={global_residual} edge_loss_weight={edge_loss_weight} "
        "detail_loss_weight={detail_loss_weight} spectrum_loss_weight={spectrum_loss_weight} "
        "base_divergence_weight={base_divergence_weight} skip_grad_norm={skip_grad_norm}".format(
            model=config.model,
            preset=args.model_preset,
            params=metadata["param_count"],
            input_mode=input_mode,
            data=Path(args.data).expanduser(),
            layout=metadata["train_data_layout"],
            out=out_dir,
            steps_per_epoch=steps_per_epoch,
            samples=len(sample_paths),
            workers=args.data_workers,
            preload=args.preload_data,
            degradation=args.degradation,
            global_residual=config.global_residual,
            edge_loss_weight=args.edge_loss_weight,
            detail_loss_weight=args.detail_loss_weight,
            spectrum_loss_weight=args.spectrum_loss_weight,
            base_divergence_weight=args.base_divergence_weight,
            skip_grad_norm=args.skip_grad_norm,
        ),
        flush=True,
    )
    if run is not None and getattr(run, "url", None):
        print(f"wandb_url={run.url}", flush=True)

    start = time.time()
    last = start
    start_step = int(state.step) + 1
    show_progress = use_progress_bar(args)
    progress = tqdm(
        range(start_step, args.steps + 1),
        total=args.steps,
        initial=start_step - 1,
        desc="train",
        dynamic_ncols=True,
        leave=True,
        disable=not show_progress,
    )
    for step in progress:
        if first_train_batch is not None:
            x_np, y_np = first_train_batch
            first_train_batch = None
        else:
            x_np, y_np = next(train_batches)
        state, metrics = train_step(
            state,
            (jnp.asarray(x_np), jnp.asarray(y_np)),
            edge_loss_weight,
            detail_loss_weight,
            spectrum_loss_weight,
            base_divergence_weight,
            base_divergence_floor,
            skip_grad_norm,
            skip_loss_threshold,
        )

        if step == start_step:
            jax.block_until_ready(metrics["loss"])

        if step % args.log_every == 0 or step == start_step:
            now = time.time()
            interval = step - start_step + 1 if step == start_step else args.log_every
            steps_per_sec = interval / max(now - last, 1e-6)
            last = now
            logs = {
                "train/step": step,
                "train/loss": float(metrics["loss"]),
                "train/pixel_loss": float(metrics["pixel_loss"]),
                "train/edge_loss": float(metrics["edge_loss"]),
                "train/detail_loss": float(metrics["detail_loss"]),
                "train/spectrum_loss": float(metrics["spectrum_loss"]),
                "train/collapse_loss": float(metrics["collapse_loss"]),
                "train/base_mae": float(metrics["base_mae"]),
                "train/psnr": float(metrics["psnr"]),
                "train/grad_norm": float(metrics["grad_norm"]),
                "train/skip_update": float(metrics["skip_update"]),
                "train/param_norm": float(metrics["param_norm"]),
                "train/lr": float(lr_schedule(state.step)),
                "train/steps_per_sec": steps_per_sec,
            }
            if show_progress:
                progress.set_postfix(
                    loss=f"{logs['train/loss']:.4f}",
                    psnr=f"{logs['train/psnr']:.2f}",
                    lr=f"{logs['train/lr']:.1e}",
                    base=f"{logs['train/base_mae'] * 255.0:.1f}",
                    ips=f"{steps_per_sec:.1f}",
                )
            else:
                log_console(format_metrics(step, logs))
            if logs["train/skip_update"] >= 0.5:
                log_console(
                    f"skip update step={step} loss={logs['train/loss']:.5f} "
                    f"grad={logs['train/grad_norm']:.2f}",
                    progress if show_progress else None,
                )
            if run is not None:
                run.log(logs, step=step)

        eval_due = eval_batches is not None and (
            step % args.eval_every == 0 or (args.eval_at_start and step == start_step)
        )
        image_due = should_log_for_epoch(
            step=step,
            steps_per_epoch=steps_per_epoch,
            every_epochs=args.image_log_every_epochs,
        )
        sample_due = should_log_for_epoch(
            step=step,
            steps_per_epoch=steps_per_epoch,
            every_epochs=args.sample_log_every_epochs,
        )

        if eval_due:
            eval_logs = maybe_eval(
                state,
                eval_batches,
                eval_batch_count=args.eval_batches,
                edge_loss_weight=edge_loss_weight,
                detail_loss_weight=detail_loss_weight,
                spectrum_loss_weight=spectrum_loss_weight,
                base_divergence_weight=base_divergence_weight,
                base_divergence_floor=base_divergence_floor,
                input_mode=input_mode,
            )
            eval_logs["train/step"] = step
            log_console(
                f"eval step={step} loss={eval_logs['eval/loss']:.5f} "
                f"psnr={eval_logs['eval/psnr']:.2f} "
                f"bicubic={eval_logs['eval/bicubic_psnr']:.2f} "
                f"gain={eval_logs['eval/psnr_gain_vs_bicubic']:.2f} "
                f"ms_ssim={eval_logs['eval/ms_ssim']:.4f} "
                f"de={eval_logs['eval/color_delta_e']:.2f} "
                f"base_mae={eval_logs['eval/base_mae'] * 255.0:.2f}/255",
                progress if show_progress else None,
            )
            if run is not None:
                run.log(eval_logs, step=step)

        if run is not None and eval_image_batches and image_due:
            images = make_eval_image_logs(
                state,
                eval_image_batches,
                count=args.image_log_count,
                scale=args.scale,
                input_mode=input_mode,
            )
            if images:
                run.log({"train/step": step, "eval/images": images}, step=step)

        net_due = should_log_for_epoch(
            step=step,
            steps_per_epoch=steps_per_epoch,
            every_epochs=args.net_perceptual_every_epochs,
        )
        if run is not None and eval_image_batches and net_due and net_perceptual.available:
            net_logs = make_net_perceptual_logs(state, eval_image_batches, net_perceptual)
            if net_logs:
                net_logs["train/step"] = step
                run.log(net_logs, step=step)
                log_console(
                    "net eval step={step} {metrics}".format(
                        step=step,
                        metrics=" ".join(
                            f"{key.split('/')[-1]}={value:.4f}"
                            for key, value in sorted(net_logs.items())
                            if key != "train/step"
                        ),
                    ),
                    progress if show_progress else None,
                )

        if sample_paths and sample_due:
            epoch = max(1, step // steps_per_epoch)
            log_console(
                f"samples epoch={epoch} step={step} count={len(sample_paths)}",
                progress if show_progress else None,
            )
            save_and_log_user_samples(
                state,
                sample_paths,
                out_dir=out_dir,
                step=step,
                epoch=epoch,
                scale=args.scale,
                input_mode=input_mode,
                window_size=config.window_size,
                max_side=args.sample_max_side,
                run=run,
            )

        if step % args.save_every == 0 or step == args.steps:
            checkpoints.save_checkpoint(
                out_dir,
                state,
                step=step,
                overwrite=True,
                keep=args.keep_checkpoints,
            )
            log_console(f"checkpoint step={step} out={out_dir}", progress if show_progress else None)

    elapsed = time.time() - start
    progress.close()
    print(f"done steps={args.steps} elapsed_sec={elapsed:.1f} out={out_dir}")
    if run is not None:
        run.finish()


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    main()
