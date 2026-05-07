from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class CheckpointSpec:
    root: Path
    step: int | None
    label: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run x4 super-resolution inference.")
    parser.add_argument(
        "--checkpoint",
        nargs="+",
        required=True,
        help="Checkpoint run directory, checkpoint_N directory, or several of them.",
    )
    parser.add_argument(
        "--checkpoint-step",
        nargs="*",
        type=int,
        default=None,
        help="Optional step(s) to restore from each checkpoint run directory.",
    )
    parser.add_argument("--input", required=True, help="Input image file or directory.")
    parser.add_argument("--output", required=True, help="Output file or directory.")
    parser.add_argument(
        "--platform",
        choices=["auto", "cpu", "tpu"],
        default="auto",
        help="Set JAX_PLATFORMS before importing JAX. Use cpu while training owns the TPU.",
    )
    parser.add_argument("--save-bicubic", action="store_true", help="Save x4 bicubic baselines.")
    parser.add_argument("--compare", action="store_true", help="Save per-image comparison contact sheets.")
    parser.add_argument("--compare-width", type=int, default=360, help="Cell width for comparison sheets.")
    parser.add_argument(
        "--max-side",
        type=int,
        default=0,
        help="Resize input LR images so their longest side is at most this value. 0 disables.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=0,
        help="Tile model input in pixels. For LR models this is LR tile size. 0 disables.",
    )
    parser.add_argument("--tile-overlap", type=int, default=16)
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
        help="Fallback when run_config.json is not found.",
    )
    parser.add_argument("--model", choices=["edsr_lite", "hat_atd", "atd_v2"], default="hat_atd")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--features", type=int, default=96)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--heads", type=int, default=6)
    parser.add_argument("--window-size", type=int, default=8)
    parser.add_argument("--mlp-ratio", type=float, default=2.0)
    parser.add_argument("--token-count", type=int, default=64)
    parser.add_argument("--channel-reduction", type=int, default=4)
    parser.add_argument("--residual-scale", type=float, default=0.15)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--input-mode", choices=["auto", "lr", "bicubic_up"], default="auto")
    return parser.parse_args()


def configure_platform(platform: str) -> None:
    if platform != "auto":
        os.environ["JAX_PLATFORMS"] = platform


def import_runtime():
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training import checkpoints

    from sr_tpu.model import ModelConfig, apply_preset, create_model
    from train import create_state, prepare_sample_input, to_uint8

    return {
        "jax": jax,
        "jnp": jnp,
        "optax": optax,
        "checkpoints": checkpoints,
        "ModelConfig": ModelConfig,
        "apply_preset": apply_preset,
        "create_model": create_model,
        "create_state": create_state,
        "prepare_sample_input": prepare_sample_input,
        "to_uint8": to_uint8,
    }


def list_input_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.exists():
        raise FileNotFoundError(f"Input does not exist: {input_path}")
    images = [
        path
        for path in sorted(input_path.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        raise FileNotFoundError(f"No images found under: {input_path}")
    return images


def parse_checkpoint_dir(path: Path) -> tuple[Path, int | None]:
    if path.name.startswith("checkpoint_"):
        try:
            return path.parent, int(path.name.split("_", 1)[1])
        except ValueError:
            return path.parent, None
    return path, None


def checkpoint_specs(args: argparse.Namespace) -> list[CheckpointSpec]:
    specs = []
    requested_steps = args.checkpoint_step if args.checkpoint_step else [None]
    for raw in args.checkpoint:
        path = Path(raw).expanduser().resolve()
        root, embedded_step = parse_checkpoint_dir(path)
        steps = requested_steps
        if embedded_step is not None and args.checkpoint_step is None:
            steps = [embedded_step]
        for step in steps:
            actual_step = embedded_step if embedded_step is not None and step is None else step
            label = root.name if actual_step is None else f"{root.name}_step_{actual_step:06d}"
            specs.append(CheckpointSpec(root=root, step=actual_step, label=label))
    return specs


def config_from_args(args: argparse.Namespace, runtime: dict[str, Any]):
    config = runtime["ModelConfig"](
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
    )
    return runtime["apply_preset"](config, args.model_preset)


def load_training_config(checkpoint: Path, args: argparse.Namespace, runtime: dict[str, Any]):
    run_config = checkpoint / "run_config.json"
    if not run_config.exists() and checkpoint.name.startswith("checkpoint_"):
        run_config = checkpoint.parent / "run_config.json"
    if run_config.exists():
        metadata = json.loads(run_config.read_text(encoding="utf-8"))
        config = runtime["ModelConfig"](**metadata["model_config"])
        input_mode = metadata.get("input_mode") or metadata.get("args", {}).get("input_mode", "auto")
        if input_mode == "auto":
            input_mode = "bicubic_up" if config.model == "edsr_lite" else "lr"
        return config, input_mode

    config = config_from_args(args, runtime)
    input_mode = args.input_mode
    if input_mode == "auto":
        input_mode = "bicubic_up" if config.model == "edsr_lite" else "lr"
    return config, input_mode


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


def start_positions(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    positions = list(range(0, max(1, length - tile_size + 1), stride))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def predict_full(model, state, x_np: np.ndarray, runtime: dict[str, Any]) -> np.ndarray:
    jnp = runtime["jnp"]
    pred = model.apply({"params": state.params}, jnp.asarray(x_np))
    return np.asarray(jnp.clip(pred[0], 0.0, 1.0))


def predict_tiled(
    model,
    state,
    x_np: np.ndarray,
    *,
    scale: int,
    window_size: int,
    tile_size: int,
    tile_overlap: int,
    runtime: dict[str, Any],
) -> np.ndarray:
    if tile_size <= 0:
        return predict_full(model, state, x_np, runtime)

    height, width = x_np.shape[1:3]
    tile_size = max(window_size, tile_size - tile_size % window_size)
    tile_size = min(tile_size, height, width)
    tile_size = max(window_size, tile_size)
    overlap = max(0, min(tile_overlap, tile_size - window_size))
    overlap = overlap - overlap % window_size
    stride = max(window_size, tile_size - overlap)
    if height <= tile_size and width <= tile_size:
        return predict_full(model, state, x_np, runtime)

    y_positions = start_positions(height, tile_size, stride)
    x_positions = start_positions(width, tile_size, stride)
    canvas = np.zeros((height * scale, width * scale, x_np.shape[-1]), dtype=np.float32)
    weights = np.zeros((height * scale, width * scale, 1), dtype=np.float32)
    for top in y_positions:
        for left in x_positions:
            tile = x_np[:, top : top + tile_size, left : left + tile_size, :]
            pred = predict_full(model, state, tile, runtime)
            out_top = top * scale
            out_left = left * scale
            out_h = pred.shape[0]
            out_w = pred.shape[1]
            canvas[out_top : out_top + out_h, out_left : out_left + out_w] += pred
            weights[out_top : out_top + out_h, out_left : out_left + out_w] += 1.0
    return canvas / np.maximum(weights, 1e-6)


def init_input_shape(
    first_x: np.ndarray,
    *,
    input_mode: str,
    scale: int,
    window_size: int,
    tile_size: int,
) -> tuple[int, int, int, int]:
    if tile_size <= 0:
        return first_x.shape
    if input_mode == "lr":
        size = max(window_size, tile_size - tile_size % window_size)
    else:
        base = max(window_size, tile_size - tile_size % window_size)
        size = base * scale
    return (first_x.shape[0], size, size, first_x.shape[-1])


def bicubic_baseline(image: Image.Image, scale: int, input_mode: str, output_hw: tuple[int, int]) -> Image.Image:
    if input_mode == "lr":
        return image.resize((image.width * scale, image.height * scale), Image.Resampling.BICUBIC)
    return image.resize((output_hw[1], output_hw[0]), Image.Resampling.BICUBIC)


def to_uint8_np(image: np.ndarray) -> np.ndarray:
    return np.asarray(np.clip(image, 0.0, 1.0) * 255.0 + 0.5, dtype=np.uint8)


def resize_to_width(image: np.ndarray, width: int) -> np.ndarray:
    if image.shape[1] == width:
        return image
    height = max(1, round(image.shape[0] * width / image.shape[1]))
    return np.asarray(Image.fromarray(image).resize((width, height), Image.Resampling.BICUBIC))


def labeled_cell(image: np.ndarray, label: str, width: int) -> np.ndarray:
    image = resize_to_width(image, width)
    header = 28
    canvas = np.full((image.shape[0] + header, image.shape[1], 3), 248, dtype=np.uint8)
    canvas[header:] = image
    pil = Image.fromarray(canvas)
    ImageDraw.Draw(pil).text((6, 7), label, fill=(20, 20, 20))
    return np.asarray(pil)


def make_contact_sheet(cells: list[tuple[str, np.ndarray]], width: int, gutter: int = 8) -> np.ndarray:
    rendered = [labeled_cell(image, label, width) for label, image in cells]
    row_height = max(cell.shape[0] for cell in rendered)
    padded = []
    for cell in rendered:
        canvas = np.full((row_height, cell.shape[1], 3), 255, dtype=np.uint8)
        canvas[: cell.shape[0], : cell.shape[1]] = cell
        padded.append(canvas)
        padded.append(np.full((row_height, gutter, 3), 255, dtype=np.uint8))
    return np.concatenate(padded[:-1], axis=1)


def resolve_output_path(
    output_root: Path,
    *,
    spec: CheckpointSpec,
    input_path: Path,
    single_file_output: bool,
) -> Path:
    if single_file_output:
        return output_root
    return output_root / spec.label / f"{input_path.stem}_x4.png"


def main() -> None:
    args = parse_args()
    configure_platform(args.platform)
    runtime = import_runtime()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser()
    inputs = list_input_images(input_path)
    specs = checkpoint_specs(args)
    single_file_output = len(inputs) == 1 and len(specs) == 1 and output_path.suffix
    if not single_file_output:
        output_path.mkdir(parents=True, exist_ok=True)

    metrics_rows = []
    for spec in specs:
        config, input_mode = load_training_config(spec.root, args, runtime)
        first_image = resize_sample(ImageOps.exif_transpose(Image.open(inputs[0])).convert("RGB"), args.max_side)
        first_x, _ = runtime["prepare_sample_input"](
            first_image,
            scale=config.scale,
            input_mode=input_mode,
            window_size=config.window_size,
        )
        model = runtime["create_model"](config)
        input_shape = init_input_shape(
            first_x,
            input_mode=input_mode,
            scale=config.scale,
            window_size=config.window_size,
            tile_size=args.tile_size,
        )
        dummy_args = argparse.Namespace(seed=0, scale=config.scale, crop_size=input_shape[1] * config.scale)
        state = runtime["create_state"](
            dummy_args,
            model,
            input_shape=input_shape,
            tx=runtime["optax"].adamw(0.0),
        )
        state = runtime["checkpoints"].restore_checkpoint(spec.root, state, step=spec.step)
        actual_step = int(state.step)
        print(
            f"loaded {spec.root} step={actual_step} label={spec.label} "
            f"model={config.model} scale={config.scale} input_mode={input_mode}",
            flush=True,
        )

        for path in inputs:
            image = resize_sample(ImageOps.exif_transpose(Image.open(path)).convert("RGB"), args.max_side)
            x_np, original_hw = runtime["prepare_sample_input"](
                image,
                scale=config.scale,
                input_mode=input_mode,
                window_size=config.window_size,
            )
            pred_np = predict_tiled(
                model,
                state,
                x_np,
                scale=config.scale if input_mode == "lr" else 1,
                window_size=config.window_size,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
                runtime=runtime,
            )
            if input_mode == "lr":
                height, width = original_hw
                pred_np = pred_np[: height * config.scale, : width * config.scale]
            else:
                height, width = original_hw
                pred_np = pred_np[:height, :width]

            pred_u8 = runtime["to_uint8"](pred_np)
            out_file = resolve_output_path(
                output_path,
                spec=CheckpointSpec(spec.root, actual_step, f"{spec.root.name}_step_{actual_step:06d}"),
                input_path=path,
                single_file_output=single_file_output,
            )
            out_file.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(pred_u8).save(out_file)

            bicubic = bicubic_baseline(image, config.scale, input_mode, pred_np.shape[:2])
            bicubic_u8 = np.asarray(bicubic)
            residual = np.abs(pred_u8.astype(np.float32) - bicubic_u8.astype(np.float32))
            metrics_rows.append(
                {
                    "checkpoint": str(spec.root),
                    "step": actual_step,
                    "input": str(path),
                    "output": str(out_file),
                    "width": pred_u8.shape[1],
                    "height": pred_u8.shape[0],
                    "mae_vs_bicubic": float(residual.mean()),
                    "p99_vs_bicubic": float(np.percentile(residual, 99)),
                    "max_vs_bicubic": float(residual.max()),
                }
            )
            print(
                f"saved {out_file} mae_vs_bicubic={residual.mean():.3f}/255 "
                f"p99={np.percentile(residual, 99):.1f}",
                flush=True,
            )

            if args.save_bicubic and not single_file_output:
                bicubic_dir = output_path / "bicubic"
                bicubic_dir.mkdir(parents=True, exist_ok=True)
                bicubic.save(bicubic_dir / f"{path.stem}_x4_bicubic.png")

    if not single_file_output and metrics_rows:
        metrics_path = output_path / "metrics.csv"
        with metrics_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics_rows[0].keys()))
            writer.writeheader()
            writer.writerows(metrics_rows)
        print(f"saved {metrics_path}", flush=True)

    if args.compare and not single_file_output:
        by_input: dict[str, list[dict[str, Any]]] = {}
        for row in metrics_rows:
            by_input.setdefault(row["input"], []).append(row)
        compare_dir = output_path / "compare"
        compare_dir.mkdir(parents=True, exist_ok=True)
        for raw_input, rows in by_input.items():
            source = Path(raw_input)
            image = resize_sample(ImageOps.exif_transpose(Image.open(source)).convert("RGB"), args.max_side)
            scale = int(rows[0]["width"] / image.width)
            bicubic = np.asarray(image.resize((image.width * scale, image.height * scale), Image.Resampling.BICUBIC))
            cells = [("bicubic", bicubic)]
            for row in rows:
                label = f"step {int(row['step'])}"
                cells.append((label, np.asarray(Image.open(row["output"]).convert("RGB"))))
            sheet = make_contact_sheet(cells, args.compare_width)
            sheet_path = compare_dir / f"{source.stem}_compare.jpg"
            Image.fromarray(sheet).save(sheet_path, quality=92)
            print(f"saved {sheet_path}", flush=True)


if __name__ == "__main__":
    main()
