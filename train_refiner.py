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
from PIL import Image, ImageOps
from tqdm import tqdm

from sr_tpu.data import batch_iterator, count_training_items, is_pair_dataset
from sr_tpu.model import ModelConfig, apply_preset, create_model
from sr_tpu.refiner import RefinerConfig, create_refiner
from sr_tpu.train_state import TrainState
from train import (
    TorchPerceptualEvaluator,
    bicubic_baseline_batch,
    charbonnier,
    color_delta_e,
    comparison_strip,
    count_params,
    cubic_base_batch,
    detail_charbonnier,
    edge_charbonnier,
    format_metrics,
    init_wandb,
    list_sample_images,
    make_contact_sheet,
    make_fixed_eval_image_batches,
    prepare_sample_input,
    psnr,
    psnr_np,
    quality_metrics,
    reference_quality_step,
    resize_array,
    resize_sample,
    spectrum_loss,
    to_uint8,
    use_progress_bar,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a small residual refiner on top of a frozen SR model.")
    parser.add_argument("--base-checkpoint", required=True, help="Frozen base SR checkpoint or run directory.")
    parser.add_argument("--base-checkpoint-step", type=int, default=0)
    parser.add_argument("--data", default="~/datasets/sr/prepared/sr_real_x4_v3_sidd_medium/train_pairs")
    parser.add_argument("--val-data", default="~/datasets/sr/prepared/sr_real_x4_v3_sidd_medium/val_pairs")
    parser.add_argument("--out", default="checkpoints/refiner_x4")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--min-lr-ratio", type=float, default=0.05)
    parser.add_argument("--warmup-steps", type=int, default=3000)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=0.5)
    parser.add_argument("--skip-grad-norm", type=float, default=10.0)
    parser.add_argument("--skip-loss-threshold", type=float, default=0.8)
    parser.add_argument("--degradation", default="phone-real")
    parser.add_argument("--input-mode", choices=["auto", "lr", "bicubic_up"], default="auto")

    parser.add_argument("--refiner-features", type=int, default=64)
    parser.add_argument("--refiner-blocks", type=int, default=10)
    parser.add_argument("--refiner-residual-scale", type=float, default=0.25)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")

    parser.add_argument("--pixel-loss-weight", type=float, default=1.0)
    parser.add_argument("--edge-loss-weight", type=float, default=0.02)
    parser.add_argument("--detail-loss-weight", type=float, default=0.05)
    parser.add_argument("--spectrum-loss-weight", type=float, default=0.01)
    parser.add_argument("--residual-loss-weight", type=float, default=0.02)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-workers", type=int, default=4)
    parser.add_argument("--eval-data-workers", type=int, default=0)
    parser.add_argument("--prefetch-batches", type=int, default=16)
    parser.add_argument("--preload-data", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--eval-every", type=int, default=1000)
    parser.add_argument("--eval-batches", type=int, default=8)
    parser.add_argument("--steps-per-epoch", type=int, default=7080)
    parser.add_argument("--image-log-count", type=int, default=4)
    parser.add_argument("--image-log-every-epochs", type=int, default=1)
    parser.add_argument("--sample-dir", default="usr_samples")
    parser.add_argument("--sample-max-images", type=int, default=8)
    parser.add_argument("--sample-max-side", type=int, default=512)
    parser.add_argument("--sample-log-every-epochs", type=int, default=1)
    parser.add_argument("--net-perceptual-metrics", choices=["off", "lpips", "dists", "all"], default="all")
    parser.add_argument("--net-perceptual-every-epochs", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=5000)
    parser.add_argument("--keep-checkpoints", type=int, default=20)
    parser.add_argument("--resume-refiner", default="")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="sr-tpu")
    parser.add_argument("--wandb-run-name", default="")
    parser.add_argument("--wandb-mode", choices=["auto", "online", "offline", "disabled"], default="auto")
    parser.add_argument("--wandb-tags", default="refiner,tpu,sr-x4")
    parser.add_argument("--wandb-quiet", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--progress", choices=["auto", "on", "off"], default="auto")

    parser.add_argument(
        "--base-model-preset",
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
        default="atd_v2_xlarge",
        help="Fallback only when base run_config.json is missing.",
    )
    return parser.parse_args()


def load_base_config(args: argparse.Namespace) -> tuple[ModelConfig, str]:
    checkpoint = Path(args.base_checkpoint).expanduser().resolve()
    candidates = [checkpoint / "run_config.json"]
    if checkpoint.name.startswith("checkpoint_"):
        candidates.append(checkpoint.parent / "run_config.json")
    for run_config in candidates:
        if run_config.exists():
            metadata = json.loads(run_config.read_text(encoding="utf-8"))
            config = ModelConfig(**metadata["model_config"])
            input_mode = metadata.get("input_mode") or metadata.get("args", {}).get("input_mode", "auto")
            if input_mode == "auto":
                input_mode = "bicubic_up" if config.model == "edsr_lite" else "lr"
            if args.input_mode != "auto":
                input_mode = args.input_mode
            return config, input_mode

    config = apply_preset(ModelConfig(scale=args.scale, dtype=args.dtype), args.base_model_preset)
    input_mode = args.input_mode
    if input_mode == "auto":
        input_mode = "bicubic_up" if config.model == "edsr_lite" else "lr"
    return config, input_mode


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


def create_refiner_state(
    args: argparse.Namespace,
    refiner,
    tx,
    *,
    input_channels: int,
) -> TrainState:
    key = jax.random.PRNGKey(args.seed + 99)
    feature_shape = (args.batch_size, args.crop_size, args.crop_size, input_channels)
    base_shape = (args.batch_size, args.crop_size, args.crop_size, 3)
    variables = refiner.init(key, jnp.ones(feature_shape, jnp.float32), jnp.ones(base_shape, jnp.float32))
    return TrainState.create(apply_fn=refiner.apply, params=variables["params"], tx=tx)


def refiner_features(base: jnp.ndarray, bicubic: jnp.ndarray) -> jnp.ndarray:
    from train import _filter_image

    base = jnp.clip(base, 0.0, 1.0).astype(jnp.float32)
    bicubic = jnp.clip(bicubic, 0.0, 1.0).astype(jnp.float32)
    base_hp = base - _filter_image(base)
    bicubic_hp = bicubic - _filter_image(bicubic)
    return jnp.concatenate([base, bicubic, base - bicubic, base_hp, bicubic_hp], axis=-1)


def make_train_step(base_model, refiner):
    @jax.jit
    def train_step(
        state: TrainState,
        base_params,
        batch: tuple[jnp.ndarray, jnp.ndarray],
        loss_weights: tuple[jnp.ndarray, ...],
        skip_grad_norm: jnp.ndarray,
        skip_loss_threshold: jnp.ndarray,
    ):
        x, y = batch
        pixel_w, edge_w, detail_w, spectrum_w, residual_w = loss_weights
        base = jax.lax.stop_gradient(jnp.clip(base_model.apply({"params": base_params}, x), 0.0, 1.0))
        bicubic = jax.lax.stop_gradient(cubic_base_batch(x, y))
        features = jax.lax.stop_gradient(refiner_features(base, bicubic))

        def loss_fn(params):
            pred = state.apply_fn({"params": params}, features, base)
            pixel_loss = charbonnier(pred, y)
            edge_loss = edge_charbonnier(pred, y)
            detail_loss = detail_charbonnier(pred, y)
            spectral_loss = spectrum_loss(pred, y)
            residual_loss = charbonnier(pred, base)
            loss = (
                pixel_w * pixel_loss
                + edge_w * edge_loss
                + detail_w * detail_loss
                + spectrum_w * spectral_loss
                + residual_w * residual_loss
            )
            return loss, (pred, pixel_loss, edge_loss, detail_loss, spectral_loss, residual_loss)

        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        pred, pixel_loss, edge_loss, detail_loss, spectral_loss, residual_loss = aux
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
            "residual_loss": residual_loss,
            "psnr": psnr(pred, y),
            "base_psnr": psnr(base, y),
            "bicubic_psnr": psnr(bicubic, y),
            "base_mae": jnp.mean(jnp.abs(pred - base)),
            "grad_norm": grad_norm,
            "skip_update": skip_update.astype(jnp.float32),
            "param_norm": optax.global_norm(state.params),
        }

    return train_step


def make_eval_step(base_model, refiner):
    @jax.jit
    def eval_step(
        state: TrainState,
        base_params,
        batch: tuple[jnp.ndarray, jnp.ndarray],
        loss_weights: tuple[jnp.ndarray, ...],
    ):
        x, y = batch
        pixel_w, edge_w, detail_w, spectrum_w, residual_w = loss_weights
        base = jnp.clip(base_model.apply({"params": base_params}, x), 0.0, 1.0)
        bicubic = cubic_base_batch(x, y)
        features = refiner_features(base, bicubic)
        pred = state.apply_fn({"params": state.params}, features, base)
        pixel_loss = charbonnier(pred, y)
        edge_loss = edge_charbonnier(pred, y)
        detail_loss = detail_charbonnier(pred, y)
        spectral_loss = spectrum_loss(pred, y)
        residual_loss = charbonnier(pred, base)
        quality = quality_metrics(pred, y)
        base_quality = quality_metrics(base, y)
        return {
            "loss": (
                pixel_w * pixel_loss
                + edge_w * edge_loss
                + detail_w * detail_loss
                + spectrum_w * spectral_loss
                + residual_w * residual_loss
            ),
            "pixel_loss": pixel_loss,
            "edge_loss": edge_loss,
            "detail_loss": detail_loss,
            "spectrum_loss": spectral_loss,
            "residual_loss": residual_loss,
            "psnr": psnr(pred, y),
            "base_psnr": psnr(base, y),
            "bicubic_psnr": psnr(bicubic, y),
            "base_mae": jnp.mean(jnp.abs(pred - base)),
            "ms_ssim": quality["ms_ssim"],
            "color_delta_e": quality["color_delta_e"],
            "base_ms_ssim": base_quality["ms_ssim"],
            "base_color_delta_e": base_quality["color_delta_e"],
        }

    return eval_step


def predict_refined(base_model, refiner, base_params, refiner_state, x: jnp.ndarray, target_shape: tuple[int, int]):
    dummy_target = jnp.zeros((x.shape[0], target_shape[0], target_shape[1], 3), dtype=jnp.float32)
    base = jnp.clip(base_model.apply({"params": base_params}, x), 0.0, 1.0)
    bicubic = cubic_base_batch(x, dummy_target)
    features = refiner_features(base, bicubic)
    pred = refiner_state.apply_fn({"params": refiner_state.params}, features, base)
    return jnp.clip(base, 0.0, 1.0), jnp.clip(pred, 0.0, 1.0), jnp.clip(bicubic, 0.0, 1.0)


def maybe_eval(eval_step, state, base_params, eval_batches, loss_weights, *, eval_batch_count: int, input_mode: str):
    values: dict[str, list[float]] = {
        "loss": [],
        "pixel_loss": [],
        "edge_loss": [],
        "detail_loss": [],
        "spectrum_loss": [],
        "residual_loss": [],
        "psnr": [],
        "base_psnr": [],
        "bicubic_psnr": [],
        "base_mae": [],
        "ms_ssim": [],
        "color_delta_e": [],
        "base_ms_ssim": [],
        "base_color_delta_e": [],
    }
    bicubic_psnrs = []
    for _ in range(eval_batch_count):
        x_np, y_np = next(eval_batches)
        metrics = eval_step(state, base_params, (jnp.asarray(x_np), jnp.asarray(y_np)), loss_weights)
        for key in values:
            values[key].append(float(metrics[key]))
        bicubic = bicubic_baseline_batch(x_np, y_np, input_mode=input_mode)
        bicubic_psnrs.extend(psnr_np(bicubic, y_np).tolist())
    logs = {f"eval/{key}": float(np.mean(items)) for key, items in values.items()}
    logs["eval/psnr_gain_vs_base"] = logs["eval/psnr"] - logs["eval/base_psnr"]
    logs["eval/psnr_gain_vs_bicubic"] = logs["eval/psnr"] - float(np.mean(bicubic_psnrs))
    logs["eval/ms_ssim_gain_vs_base"] = logs["eval/ms_ssim"] - logs["eval/base_ms_ssim"]
    logs["eval/color_delta_e_improvement_vs_base"] = logs["eval/base_color_delta_e"] - logs["eval/color_delta_e"]
    return logs


def make_eval_image_logs(base_model, refiner, base_params, state, eval_image_batches, *, count: int, scale: int, input_mode: str):
    if count <= 0 or not eval_image_batches:
        return None
    import wandb

    rows = []
    remaining = count
    image_index = 1
    for x_np, y_np in eval_image_batches:
        if remaining <= 0:
            break
        base_np, pred_np, _ = predict_refined(
            base_model,
            refiner,
            base_params,
            state,
            jnp.asarray(x_np),
            y_np.shape[1:3],
        )
        base_np = np.asarray(base_np)
        pred_np = np.asarray(pred_np)
        batch = min(remaining, x_np.shape[0])
        for index in range(batch):
            target = to_uint8(y_np[index])
            base_u8 = to_uint8(base_np[index])
            pred_u8 = to_uint8(pred_np[index])
            if input_mode == "lr":
                lr_up = resize_array(x_np[index], (target.shape[1], target.shape[0]), Image.Resampling.BICUBIC)
            else:
                lr_up = to_uint8(x_np[index])
            diff = to_uint8(np.abs(pred_np[index] - y_np[index]) * 4.0)
            rows.append((f"val {image_index:02d}", [lr_up, base_u8, pred_u8, target, diff]))
            image_index += 1
        remaining -= batch
    sheet = make_contact_sheet(
        rows,
        column_labels=["input", "base", "refined", "target", "absdiff x4"],
        cell_width=220,
    )
    return wandb.Image(sheet, caption="fixed validation: input | base | refined | target | absdiff x4")


def make_net_logs(base_model, refiner, base_params, state, eval_image_batches, evaluator):
    if not evaluator.available or not eval_image_batches:
        return {}
    preds = []
    bases = []
    targets = []
    for x_np, y_np in eval_image_batches:
        base_np, pred_np, _ = predict_refined(
            base_model,
            refiner,
            base_params,
            state,
            jnp.asarray(x_np),
            y_np.shape[1:3],
        )
        bases.append(np.asarray(base_np))
        preds.append(np.asarray(pred_np))
        targets.append(y_np)
    pred_logs = evaluator.compute(np.concatenate(preds, axis=0), np.concatenate(targets, axis=0))
    base_logs = evaluator.compute(np.concatenate(bases, axis=0), np.concatenate(targets, axis=0))
    logs = {key.replace("eval/", "eval/refiner_"): value for key, value in pred_logs.items()}
    logs.update({key.replace("eval/", "eval/base_"): value for key, value in base_logs.items()})
    for key, value in list(logs.items()):
        if key.startswith("eval/refiner_"):
            base_key = key.replace("eval/refiner_", "eval/base_")
            if base_key in logs:
                logs[key.replace("eval/refiner_", "eval/refiner_improvement_")] = logs[base_key] - value
    return logs


def save_and_log_user_samples(
    base_model,
    refiner,
    base_params,
    state,
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
    rows = []
    sample_metrics: dict[str, float] = {}
    for path in sample_paths:
        image = resize_sample(ImageOps.exif_transpose(Image.open(path)).convert("RGB"), max_side)
        x_np, original_hw = prepare_sample_input(
            image,
            scale=scale,
            input_mode=input_mode,
            window_size=window_size,
        )
        if input_mode == "lr":
            height, width = original_hw
            target_shape = (height * scale, width * scale)
            bicubic_img = image.resize((width * scale, height * scale), Image.Resampling.BICUBIC)
        else:
            target_shape = original_hw
            height, width = original_hw
            bicubic_img = image.resize((width, height), Image.Resampling.BICUBIC)
        base_np, pred_np, _ = predict_refined(
            base_model,
            refiner,
            base_params,
            state,
            jnp.asarray(x_np),
            target_shape,
        )
        base = np.asarray(base_np)[0][: target_shape[0], : target_shape[1]]
        pred = np.asarray(pred_np)[0][: target_shape[0], : target_shape[1]]
        base_u8 = to_uint8(base)
        pred_u8 = to_uint8(pred)
        bicubic_u8 = np.asarray(bicubic_img)
        Image.fromarray(base_u8).save(output_dir / f"{path.stem}_base_x{scale}.png")
        Image.fromarray(pred_u8).save(output_dir / f"{path.stem}_refined_x{scale}.png")
        residual = np.abs(pred_u8.astype(np.float32) - base_u8.astype(np.float32))
        heat = np.stack(
            [
                np.clip(residual.mean(axis=-1) * 64.0, 0.0, 255.0).astype(np.uint8),
                np.zeros(residual.shape[:2], dtype=np.uint8),
                255 - np.clip(residual.mean(axis=-1) * 64.0, 0.0, 255.0).astype(np.uint8),
            ],
            axis=-1,
        )
        rows.append((path.stem, [bicubic_u8, base_u8, pred_u8, heat]))
        sample_metrics[f"samples/{path.stem}_refined_minus_base_mae"] = float(residual.mean())
        sample_metrics[f"samples/{path.stem}_refined_minus_base_p99"] = float(np.percentile(residual, 99))

    sheet = make_contact_sheet(
        rows,
        column_labels=["bicubic", "base", "refined", "abs(refined-base) x64"],
        cell_width=330,
    )
    Image.fromarray(sheet).save(output_dir / "_contact_sheet.jpg", quality=92)
    if run is not None and wandb is not None:
        sample_metrics.update(
            {
                "train/step": step,
                "samples/usr_samples": wandb.Image(
                    sheet,
                    caption=f"refiner usr_samples epoch {epoch} step {step}: bicubic | base | refined | delta",
                ),
            }
        )
        run.log(sample_metrics, step=step)


def should_log_for_epoch(*, step: int, steps_per_epoch: int, every_epochs: int) -> bool:
    if every_epochs <= 0 or step % steps_per_epoch:
        return False
    return (step // steps_per_epoch) % every_epochs == 0


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    base_config, input_mode = load_base_config(args)
    args.input_mode = input_mode
    base_model = create_model(base_config)
    base_input_size = args.crop_size if input_mode == "bicubic_up" else args.crop_size // args.scale
    base_dummy = argparse.Namespace(seed=args.seed, crop_size=args.crop_size, scale=args.scale, input_mode=input_mode)
    base_state = TrainState.create(
        apply_fn=base_model.apply,
        params=base_model.init(
            jax.random.PRNGKey(args.seed),
            jnp.ones((args.batch_size, base_input_size, base_input_size, 3), jnp.float32),
        )["params"],
        tx=optax.adamw(0.0),
    )
    base_checkpoint = Path(args.base_checkpoint).expanduser().resolve()
    base_state = checkpoints.restore_checkpoint(
        base_checkpoint,
        base_state,
        step=args.base_checkpoint_step or None,
    )

    refiner_config = RefinerConfig(
        features=args.refiner_features,
        blocks=args.refiner_blocks,
        residual_scale=args.refiner_residual_scale,
        dtype=args.dtype,
    )
    refiner = create_refiner(refiner_config)
    tx, lr_schedule = make_optimizer(args)
    refiner_state = create_refiner_state(args, refiner, tx, input_channels=15)
    if args.resume_refiner:
        refiner_state = checkpoints.restore_checkpoint(Path(args.resume_refiner).expanduser().resolve(), refiner_state)
    elif any(out_dir.glob("checkpoint_*")):
        refiner_state = checkpoints.restore_checkpoint(out_dir, refiner_state)

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
    sample_paths = list_sample_images(args.sample_dir, args.sample_max_images)
    net_perceptual = TorchPerceptualEvaluator(args.net_perceptual_metrics)
    loss_weights = (
        jnp.asarray(args.pixel_loss_weight, dtype=jnp.float32),
        jnp.asarray(args.edge_loss_weight, dtype=jnp.float32),
        jnp.asarray(args.detail_loss_weight, dtype=jnp.float32),
        jnp.asarray(args.spectrum_loss_weight, dtype=jnp.float32),
        jnp.asarray(args.residual_loss_weight, dtype=jnp.float32),
    )
    skip_grad_norm = jnp.asarray(args.skip_grad_norm, dtype=jnp.float32)
    skip_loss_threshold = jnp.asarray(args.skip_loss_threshold, dtype=jnp.float32)

    train_step = make_train_step(base_model, refiner)
    eval_step = make_eval_step(base_model, refiner)
    train_item_count = count_training_items(args.data)
    metadata = {
        "args": vars(args),
        "base_model_config": asdict(base_config),
        "refiner_config": asdict(refiner_config),
        "input_mode": input_mode,
        "base_checkpoint": str(base_checkpoint),
        "base_step": int(base_state.step),
        "base_param_count": count_params(base_state.params),
        "refiner_param_count": count_params(refiner_state.params),
        "train_image_count": train_item_count,
        "train_data_layout": "pairs" if is_pair_dataset(args.data) else "hr",
        "steps_per_epoch": args.steps_per_epoch,
        "sample_paths": [str(path) for path in sample_paths],
        "net_perceptual_available": net_perceptual.available,
        "jax": jax.__version__,
        "backend": jax.default_backend(),
        "devices": [str(device) for device in jax.devices()],
    }
    (out_dir / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    run = init_wandb(args, metadata)
    print(
        "refiner base_step={base_step} base_params={base_params:,} refiner_params={refiner_params:,} "
        "data={data} out={out} input_mode={input_mode} degradation={degradation} "
        "features={features} blocks={blocks} residual_scale={residual_scale}".format(
            base_step=metadata["base_step"],
            base_params=metadata["base_param_count"],
            refiner_params=metadata["refiner_param_count"],
            data=Path(args.data).expanduser(),
            out=out_dir,
            input_mode=input_mode,
            degradation=args.degradation,
            features=refiner_config.features,
            blocks=refiner_config.blocks,
            residual_scale=refiner_config.residual_scale,
        ),
        flush=True,
    )
    if run is not None and getattr(run, "url", None):
        print(f"wandb_url={run.url}", flush=True)

    start = time.time()
    last = start
    start_step = int(refiner_state.step) + 1
    show_progress = use_progress_bar(args)
    progress = tqdm(
        range(start_step, args.steps + 1),
        total=args.steps,
        initial=start_step - 1,
        desc="refiner",
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
        refiner_state, metrics = train_step(
            refiner_state,
            base_state.params,
            (jnp.asarray(x_np), jnp.asarray(y_np)),
            loss_weights,
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
                "train/residual_loss": float(metrics["residual_loss"]),
                "train/psnr": float(metrics["psnr"]),
                "train/base_psnr": float(metrics["base_psnr"]),
                "train/bicubic_psnr": float(metrics["bicubic_psnr"]),
                "train/psnr_gain_vs_base": float(metrics["psnr"] - metrics["base_psnr"]),
                "train/base_mae": float(metrics["base_mae"]),
                "train/grad_norm": float(metrics["grad_norm"]),
                "train/skip_update": float(metrics["skip_update"]),
                "train/param_norm": float(metrics["param_norm"]),
                "train/lr": float(lr_schedule(refiner_state.step)),
                "train/steps_per_sec": steps_per_sec,
            }
            if show_progress:
                progress.set_postfix(
                    loss=f"{logs['train/loss']:.4f}",
                    psnr=f"{logs['train/psnr']:.2f}",
                    gain=f"{logs['train/psnr_gain_vs_base']:.2f}",
                    base=f"{logs['train/base_mae'] * 255.0:.1f}",
                    lr=f"{logs['train/lr']:.1e}",
                    ips=f"{steps_per_sec:.1f}",
                )
            else:
                print(
                    f"step={step} loss={logs['train/loss']:.5f} psnr={logs['train/psnr']:.2f} "
                    f"base_psnr={logs['train/base_psnr']:.2f} gain={logs['train/psnr_gain_vs_base']:.2f} "
                    f"base_mae={logs['train/base_mae'] * 255.0:.2f}/255 lr={logs['train/lr']:.2e}",
                    flush=True,
                )
            if run is not None:
                run.log(logs, step=step)

        eval_due = eval_batches is not None and step % args.eval_every == 0
        image_due = should_log_for_epoch(
            step=step,
            steps_per_epoch=args.steps_per_epoch,
            every_epochs=args.image_log_every_epochs,
        )
        sample_due = should_log_for_epoch(
            step=step,
            steps_per_epoch=args.steps_per_epoch,
            every_epochs=args.sample_log_every_epochs,
        )
        net_due = should_log_for_epoch(
            step=step,
            steps_per_epoch=args.steps_per_epoch,
            every_epochs=args.net_perceptual_every_epochs,
        )

        if eval_due:
            eval_logs = maybe_eval(
                eval_step,
                refiner_state,
                base_state.params,
                eval_batches,
                loss_weights,
                eval_batch_count=args.eval_batches,
                input_mode=input_mode,
            )
            eval_logs["train/step"] = step
            if show_progress:
                progress.write(
                    f"eval step={step} loss={eval_logs['eval/loss']:.5f} "
                    f"psnr={eval_logs['eval/psnr']:.2f} base={eval_logs['eval/base_psnr']:.2f} "
                    f"gain={eval_logs['eval/psnr_gain_vs_base']:.2f} "
                    f"ms={eval_logs['eval/ms_ssim']:.4f} de={eval_logs['eval/color_delta_e']:.2f} "
                    f"base_mae={eval_logs['eval/base_mae'] * 255.0:.2f}/255"
                )
            if run is not None:
                run.log(eval_logs, step=step)

        if run is not None and eval_image_batches and image_due:
            image = make_eval_image_logs(
                base_model,
                refiner,
                base_state.params,
                refiner_state,
                eval_image_batches,
                count=args.image_log_count,
                scale=args.scale,
                input_mode=input_mode,
            )
            if image is not None:
                run.log({"train/step": step, "eval/images": image}, step=step)

        if run is not None and eval_image_batches and net_due and net_perceptual.available:
            net_logs = make_net_logs(base_model, refiner, base_state.params, refiner_state, eval_image_batches, net_perceptual)
            if net_logs:
                net_logs["train/step"] = step
                run.log(net_logs, step=step)
                if show_progress:
                    progress.write(
                        "net eval step={step} {metrics}".format(
                            step=step,
                            metrics=" ".join(
                                f"{key.split('/')[-1]}={value:.4f}"
                                for key, value in sorted(net_logs.items())
                                if key != "train/step"
                            ),
                        )
                    )

        if sample_paths and sample_due:
            epoch = max(1, step // args.steps_per_epoch)
            if show_progress:
                progress.write(f"samples epoch={epoch} step={step} count={len(sample_paths)}")
            save_and_log_user_samples(
                base_model,
                refiner,
                base_state.params,
                refiner_state,
                sample_paths,
                out_dir=out_dir,
                step=step,
                epoch=epoch,
                scale=args.scale,
                input_mode=input_mode,
                window_size=base_config.window_size,
                max_side=args.sample_max_side,
                run=run,
            )

        if step % args.save_every == 0 or step == args.steps:
            checkpoints.save_checkpoint(
                out_dir,
                refiner_state,
                step=step,
                overwrite=True,
                keep=args.keep_checkpoints,
            )
            if show_progress:
                progress.write(f"checkpoint step={step} out={out_dir}")

    progress.close()
    print(f"done steps={args.steps} elapsed_sec={time.time() - start:.1f} out={out_dir}", flush=True)
    if run is not None:
        run.finish()


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    main()
