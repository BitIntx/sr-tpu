from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen SR base plus residual refiner inference.")
    parser.add_argument("--base-checkpoint", required=True)
    parser.add_argument("--base-checkpoint-step", type=int, default=0)
    parser.add_argument("--refiner-checkpoint", required=True)
    parser.add_argument("--refiner-checkpoint-step", type=int, default=0)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--platform", choices=["auto", "cpu", "tpu"], default="auto")
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--save-base", action="store_true")
    parser.add_argument("--save-bicubic", action="store_true")
    parser.add_argument("--compare-width", type=int, default=360)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--input-mode", choices=["auto", "lr", "bicubic_up"], default="auto")
    parser.add_argument("--base-model-preset", default="atd_v2_xlarge")
    return parser.parse_args()


def configure_platform(platform: str) -> None:
    if platform != "auto":
        os.environ["JAX_PLATFORMS"] = platform


def import_runtime():
    import jax
    import jax.numpy as jnp
    import optax
    from flax.training import checkpoints

    from sr_tpu.refiner import RefinerConfig, create_refiner
    from sr_tpu.train_state import TrainState
    from train import create_state, prepare_sample_input, resize_sample, to_uint8
    from train_refiner import create_refiner_state, load_base_config, predict_refined
    from sr_tpu.model import create_model

    return {
        "jax": jax,
        "jnp": jnp,
        "optax": optax,
        "checkpoints": checkpoints,
        "RefinerConfig": RefinerConfig,
        "TrainState": TrainState,
        "create_model": create_model,
        "create_refiner": create_refiner,
        "create_state": create_state,
        "create_refiner_state": create_refiner_state,
        "load_base_config": load_base_config,
        "predict_refined": predict_refined,
        "prepare_sample_input": prepare_sample_input,
        "resize_sample": resize_sample,
        "to_uint8": to_uint8,
    }


def list_input_images(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    images = [
        path
        for path in sorted(input_path.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not images:
        raise FileNotFoundError(f"No images found under: {input_path}")
    return images


def load_refiner_config(refiner_checkpoint: Path, runtime: dict[str, Any]):
    root = refiner_checkpoint.parent if refiner_checkpoint.name.startswith("checkpoint_") else refiner_checkpoint
    run_config = root / "run_config.json"
    if run_config.exists():
        metadata = json.loads(run_config.read_text(encoding="utf-8"))
        return runtime["RefinerConfig"](**metadata["refiner_config"])
    return runtime["RefinerConfig"]()


def checkpoint_root_and_step(path: Path, requested_step: int) -> tuple[Path, int | None]:
    if path.name.startswith("checkpoint_"):
        try:
            return path.parent, int(path.name.split("_", 1)[1])
        except ValueError:
            return path.parent, None
    return path, requested_step or None


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


def contact_sheet(cells: list[tuple[str, np.ndarray]], width: int, gutter: int = 8) -> np.ndarray:
    rendered = [labeled_cell(image, label, width) for label, image in cells]
    row_height = max(cell.shape[0] for cell in rendered)
    padded = []
    for cell in rendered:
        canvas = np.full((row_height, cell.shape[1], 3), 255, dtype=np.uint8)
        canvas[: cell.shape[0], : cell.shape[1]] = cell
        padded.append(canvas)
        padded.append(np.full((row_height, gutter, 3), 255, dtype=np.uint8))
    return np.concatenate(padded[:-1], axis=1)


def main() -> None:
    args = parse_args()
    configure_platform(args.platform)
    runtime = import_runtime()

    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser()
    images = list_input_images(input_path)
    single_output = len(images) == 1 and output_path.suffix
    if not single_output:
        output_path.mkdir(parents=True, exist_ok=True)

    base_checkpoint = Path(args.base_checkpoint).expanduser().resolve()
    refiner_checkpoint = Path(args.refiner_checkpoint).expanduser().resolve()
    base_config, input_mode = runtime["load_base_config"](args)
    refiner_config = load_refiner_config(refiner_checkpoint, runtime)
    base_model = runtime["create_model"](base_config)
    refiner = runtime["create_refiner"](refiner_config)

    first_image = runtime["resize_sample"](
        ImageOps.exif_transpose(Image.open(images[0])).convert("RGB"),
        args.max_side,
    )
    first_x, _ = runtime["prepare_sample_input"](
        first_image,
        scale=base_config.scale,
        input_mode=input_mode,
        window_size=base_config.window_size,
    )
    base_state = runtime["create_state"](
        argparse.Namespace(seed=0, scale=base_config.scale, crop_size=first_x.shape[1] * base_config.scale, input_mode=input_mode),
        base_model,
        input_shape=first_x.shape,
        tx=runtime["optax"].adamw(0.0),
    )
    base_root, base_step = checkpoint_root_and_step(base_checkpoint, args.base_checkpoint_step)
    base_state = runtime["checkpoints"].restore_checkpoint(base_root, base_state, step=base_step)

    refiner_state = runtime["create_refiner_state"](
        argparse.Namespace(seed=0, batch_size=1, crop_size=64),
        refiner,
        runtime["optax"].adamw(0.0),
        input_channels=15,
    )
    refiner_root, refiner_step = checkpoint_root_and_step(refiner_checkpoint, args.refiner_checkpoint_step)
    refiner_state = runtime["checkpoints"].restore_checkpoint(refiner_root, refiner_state, step=refiner_step)
    print(
        f"loaded base_step={int(base_state.step)} refiner_step={int(refiner_state.step)} "
        f"input_mode={input_mode}",
        flush=True,
    )

    rows = []
    for image_path in images:
        image = runtime["resize_sample"](
            ImageOps.exif_transpose(Image.open(image_path)).convert("RGB"),
            args.max_side,
        )
        x_np, original_hw = runtime["prepare_sample_input"](
            image,
            scale=base_config.scale,
            input_mode=input_mode,
            window_size=base_config.window_size,
        )
        if input_mode == "lr":
            height, width = original_hw
            target_shape = (height * base_config.scale, width * base_config.scale)
            bicubic = image.resize((width * base_config.scale, height * base_config.scale), Image.Resampling.BICUBIC)
        else:
            target_shape = original_hw
            height, width = original_hw
            bicubic = image.resize((width, height), Image.Resampling.BICUBIC)

        base_np, pred_np, _ = runtime["predict_refined"](
            base_model,
            refiner,
            base_state.params,
            refiner_state,
            runtime["jnp"].asarray(x_np),
            target_shape,
        )
        base_u8 = runtime["to_uint8"](np.asarray(base_np)[0][: target_shape[0], : target_shape[1]])
        pred_u8 = runtime["to_uint8"](np.asarray(pred_np)[0][: target_shape[0], : target_shape[1]])
        bicubic_u8 = np.asarray(bicubic)

        if single_output:
            out_file = output_path
        else:
            out_file = output_path / f"{image_path.stem}_refined_x{base_config.scale}.png"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pred_u8).save(out_file)
        if args.save_base and not single_output:
            Image.fromarray(base_u8).save(output_path / f"{image_path.stem}_base_x{base_config.scale}.png")
        if args.save_bicubic and not single_output:
            Image.fromarray(bicubic_u8).save(output_path / f"{image_path.stem}_bicubic_x{base_config.scale}.png")
        if args.compare and not single_output:
            delta = np.abs(pred_u8.astype(np.float32) - base_u8.astype(np.float32))
            heat = np.stack(
                [
                    np.clip(delta.mean(axis=-1) * 64.0, 0.0, 255.0).astype(np.uint8),
                    np.zeros(delta.shape[:2], dtype=np.uint8),
                    255 - np.clip(delta.mean(axis=-1) * 64.0, 0.0, 255.0).astype(np.uint8),
                ],
                axis=-1,
            )
            sheet = contact_sheet(
                [("bicubic", bicubic_u8), ("base", base_u8), ("refined", pred_u8), ("delta x64", heat)],
                args.compare_width,
            )
            compare_dir = output_path / "compare"
            compare_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(sheet).save(compare_dir / f"{image_path.stem}_compare.jpg", quality=92)

        residual = np.abs(pred_u8.astype(np.float32) - base_u8.astype(np.float32))
        rows.append(
            {
                "input": str(image_path),
                "output": str(out_file),
                "base_step": int(base_state.step),
                "refiner_step": int(refiner_state.step),
                "mae_refined_vs_base": float(residual.mean()),
                "p99_refined_vs_base": float(np.percentile(residual, 99)),
            }
        )
        print(f"saved {out_file} refined_vs_base_mae={residual.mean():.3f}/255", flush=True)

    if rows and not single_output:
        with (output_path / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
