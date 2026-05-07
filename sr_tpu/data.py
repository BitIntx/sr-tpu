from __future__ import annotations

import os
import queue
import random
import threading
import json
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
ImageCache = dict[Path, np.ndarray]
RESAMPLE_FILTERS = (
    Image.Resampling.BICUBIC,
    Image.Resampling.BILINEAR,
    Image.Resampling.LANCZOS,
    Image.Resampling.BOX,
)


@dataclass(frozen=True)
class PairRecord:
    noisy: Path
    target: Path
    name: str


def list_images(root: str | Path) -> list[Path]:
    root = Path(root).expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Image directory does not exist: {root}")
    images = []
    for dirpath, _, filenames in os.walk(root, followlinks=True):
        directory = Path(dirpath)
        images.extend(
            directory / filename
            for filename in filenames
            if Path(filename).suffix.lower() in IMAGE_EXTENSIONS
        )
    images = sorted(path for path in images if path.is_file())
    if not images:
        raise FileNotFoundError(f"No images found under: {root}")
    return images


def _pair_manifest_path(root: str | Path) -> Path | None:
    path = Path(root).expanduser()
    if path.is_file() and path.suffix.lower() in {".jsonl", ".json"}:
        return path
    manifest = path / "pairs.jsonl"
    if manifest.exists():
        return manifest
    return None


def is_pair_dataset(root: str | Path) -> bool:
    return _pair_manifest_path(root) is not None


def _resolve_manifest_path(value: str, *, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path


def list_pair_records(root: str | Path) -> list[PairRecord]:
    manifest = _pair_manifest_path(root)
    if manifest is None:
        raise FileNotFoundError(f"No pairs.jsonl manifest found under: {root}")
    base = manifest.parent
    records: list[PairRecord] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            noisy_value = row.get("noisy") or row.get("lr") or row.get("input")
            target_value = row.get("target") or row.get("gt") or row.get("hr")
            if not noisy_value or not target_value:
                raise ValueError(
                    f"{manifest}:{line_number} needs noisy/lr/input and target/gt/hr fields"
                )
            noisy = _resolve_manifest_path(str(noisy_value), base=base)
            target = _resolve_manifest_path(str(target_value), base=base)
            name = str(row.get("name") or noisy.stem)
            records.append(PairRecord(noisy=noisy, target=target, name=name))
    if not records:
        raise FileNotFoundError(f"No pairs found in: {manifest}")
    missing = [record for record in records if not record.noisy.exists() or not record.target.exists()]
    if missing:
        sample = missing[0]
        raise FileNotFoundError(
            f"Pair manifest references missing files, e.g. noisy={sample.noisy} target={sample.target}"
        )
    return records


def count_training_items(root: str | Path) -> int:
    if is_pair_dataset(root):
        return len(list_pair_records(root))
    return len(list_images(root))


def _canonical(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:
        return path


def preload_image_cache(images: list[Path], *, progress: bool = True) -> ImageCache:
    unique: list[tuple[Path, Path]] = []
    seen: set[Path] = set()
    for path in images:
        key = _canonical(path)
        if key in seen:
            continue
        unique.append((path, key))
        seen.add(key)

    cache: ImageCache = {}
    for index, (path, key) in enumerate(unique, start=1):
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            cache[key] = np.asarray(image, dtype=np.uint8).copy()
        if progress and (index % 500 == 0 or index == len(unique)):
            loaded_gib = sum(array.nbytes for array in cache.values()) / (1024**3)
            print(
                f"preload images {index}/{len(unique)} unique, ram={loaded_gib:.2f} GiB",
                flush=True,
            )
    return cache


def _load_rgb(path: Path, cache: ImageCache | None = None) -> Image.Image:
    if cache is not None:
        try:
            return Image.fromarray(cache[_canonical(path)])
        except KeyError as exc:
            raise FileNotFoundError(f"Image is missing from preload cache: {path}") from exc
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def _random_crop(image: Image.Image, crop_size: int, rng: random.Random) -> Image.Image:
    width, height = image.size
    if width < crop_size or height < crop_size:
        scale = crop_size / min(width, height)
        new_size = (max(crop_size, round(width * scale)), max(crop_size, round(height * scale)))
        image = image.resize(new_size, Image.Resampling.BICUBIC)
        width, height = image.size

    left = rng.randint(0, width - crop_size)
    top = rng.randint(0, height - crop_size)
    return image.crop((left, top, left + crop_size, top + crop_size))


def _augment(image: Image.Image, rng: random.Random) -> Image.Image:
    if rng.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    rotations = rng.randint(0, 3)
    if rotations:
        image = image.rotate(90 * rotations, expand=True)
    return image


def _augment_pair(
    noisy: Image.Image,
    target: Image.Image,
    rng: random.Random,
) -> tuple[Image.Image, Image.Image]:
    if rng.random() < 0.5:
        noisy = noisy.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        target = target.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if rng.random() < 0.5:
        noisy = noisy.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        target = target.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    rotations = rng.randint(0, 3)
    if rotations:
        noisy = noisy.rotate(90 * rotations, expand=True)
        target = target.rotate(90 * rotations, expand=True)
    return noisy, target


def _jpeg_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def _webp_roundtrip(image: Image.Image, quality: int) -> Image.Image:
    buffer = BytesIO()
    image.save(buffer, format="WEBP", quality=quality, method=4)
    buffer.seek(0)
    with Image.open(buffer) as decoded:
        return decoded.convert("RGB")


def _np_rng(rng: random.Random) -> np.random.Generator:
    return np.random.default_rng(rng.randrange(0, 2**32))


def _motion_blur(image: Image.Image, rng: random.Random, size: int) -> Image.Image:
    kernel = np.zeros((size, size), dtype=np.float32)
    direction = rng.choice(("horizontal", "vertical", "diag", "anti_diag"))
    if direction == "horizontal":
        kernel[size // 2, :] = 1.0
    elif direction == "vertical":
        kernel[:, size // 2] = 1.0
    elif direction == "diag":
        np.fill_diagonal(kernel, 1.0)
    else:
        np.fill_diagonal(np.fliplr(kernel), 1.0)
    kernel /= kernel.sum()
    return image.filter(ImageFilter.Kernel((size, size), kernel.reshape(-1).tolist(), scale=1.0))


def _color_jitter(image: Image.Image, rng: random.Random, strength: float) -> Image.Image:
    enhancers = [
        ImageEnhance.Brightness,
        ImageEnhance.Contrast,
        ImageEnhance.Color,
    ]
    rng.shuffle(enhancers)
    for enhancer in enhancers:
        factor = 1.0 + rng.uniform(-strength, strength)
        image = enhancer(image).enhance(max(0.05, factor))
    return image


def _add_rgb_noise(
    image: Image.Image,
    rng: random.Random,
    *,
    gray_sigma: float,
    color_sigma: float,
) -> Image.Image:
    generator = _np_rng(rng)
    arr = np.asarray(image, dtype=np.float32)
    if gray_sigma > 0:
        arr += generator.normal(0.0, gray_sigma, arr.shape[:2] + (1,))
    if color_sigma > 0:
        arr += generator.normal(0.0, color_sigma, arr.shape)
    return Image.fromarray(np.clip(arr, 0.0, 255.0).astype(np.uint8), mode="RGB")


def _add_chroma_noise(image: Image.Image, rng: random.Random, sigma: float) -> Image.Image:
    generator = _np_rng(rng)
    arr = np.asarray(image.convert("YCbCr"), dtype=np.float32)
    arr[..., 1:] += generator.normal(0.0, sigma, arr[..., 1:].shape)
    return Image.fromarray(np.clip(arr, 0.0, 255.0).astype(np.uint8), mode="YCbCr").convert("RGB")


def _add_banding(image: Image.Image, rng: random.Random, strength: float) -> Image.Image:
    generator = _np_rng(rng)
    arr = np.asarray(image, dtype=np.float32)
    height = arr.shape[0]
    bands = generator.normal(0.0, strength, (height, 1, 1)).astype(np.float32)
    if rng.random() < 0.5:
        bands = np.cumsum(bands, axis=0)
        bands -= bands.mean()
        bands /= max(float(np.std(bands)), 1e-6)
        bands *= strength
    arr += bands
    return Image.fromarray(np.clip(arr, 0.0, 255.0).astype(np.uint8), mode="RGB")


def _resize_to_lr(
    image: Image.Image,
    lr_size: int,
    rng: random.Random,
    degradation: str,
) -> Image.Image:
    output_size = (lr_size, lr_size)
    if degradation == "bicubic":
        return image.resize(output_size, Image.Resampling.BICUBIC)

    if degradation == "mixed-light":
        return image.resize(output_size, rng.choice(RESAMPLE_FILTERS[:3]))

    if degradation not in {"mixed-real", "phone-real"}:
        raise ValueError(f"Unsupported degradation: {degradation}")

    if rng.random() < (0.55 if degradation == "mixed-real" else 0.75):
        factor = rng.uniform(0.35, 2.0) if degradation == "phone-real" else rng.uniform(0.45, 1.8)
        mid_size = max(8, round(lr_size * factor))
        image = image.resize((mid_size, mid_size), rng.choice(RESAMPLE_FILTERS))

    if rng.random() < (0.18 if degradation == "mixed-real" else 0.30):
        factor_x = rng.uniform(0.75, 1.25)
        factor_y = rng.uniform(0.75, 1.25)
        mid_size = (
            max(8, round(lr_size * factor_x)),
            max(8, round(lr_size * factor_y)),
        )
        image = image.resize(mid_size, rng.choice(RESAMPLE_FILTERS))

    return image.resize(output_size, rng.choice(RESAMPLE_FILTERS))


def _degrade_before_resize(image: Image.Image, rng: random.Random, degradation: str) -> Image.Image:
    if degradation == "bicubic":
        return image
    if degradation not in {"mixed-light", "mixed-real", "phone-real"}:
        raise ValueError(f"Unsupported degradation: {degradation}")

    blur_prob = 0.25 if degradation == "mixed-light" else 0.75
    blur_max = 1.2 if degradation == "mixed-light" else 3.2
    if degradation == "phone-real":
        blur_prob = 0.85
        blur_max = 4.0
    if rng.random() < blur_prob:
        radius = rng.uniform(0.2, blur_max)
        image = image.filter(ImageFilter.GaussianBlur(radius=radius))
    if degradation in {"mixed-real", "phone-real"} and rng.random() < (0.35 if degradation == "mixed-real" else 0.45):
        image = _motion_blur(image, rng, rng.choice((3, 5)))
    if degradation in {"mixed-real", "phone-real"} and rng.random() < (0.25 if degradation == "mixed-real" else 0.55):
        image = _jpeg_roundtrip(image, rng.randint(45, 95))
    if degradation == "phone-real" and rng.random() < 0.20:
        image = _webp_roundtrip(image, rng.randint(35, 90))
    if degradation in {"mixed-real", "phone-real"} and rng.random() < (0.35 if degradation == "mixed-real" else 0.30):
        image = _color_jitter(image, rng, strength=0.18)
    return image


def _degrade_after_resize(lr: Image.Image, rng: random.Random, degradation: str) -> Image.Image:
    if degradation == "bicubic":
        return lr
    if degradation not in {"mixed-light", "mixed-real", "phone-real"}:
        raise ValueError(f"Unsupported degradation: {degradation}")

    jpeg_prob = 0.25 if degradation == "mixed-light" else 0.75
    if degradation == "phone-real":
        jpeg_prob = 0.90
    if rng.random() < jpeg_prob:
        if degradation == "mixed-light":
            quality = rng.randint(75, 95)
        elif degradation == "mixed-real":
            quality = rng.randint(32, 92)
        else:
            quality = rng.randint(20, 88)
        lr = _jpeg_roundtrip(lr, quality)
    if degradation in {"mixed-real", "phone-real"} and rng.random() < (0.25 if degradation == "mixed-real" else 0.45):
        lr = _jpeg_roundtrip(lr, rng.randint(18, 80))
    if degradation == "phone-real" and rng.random() < 0.25:
        lr = _webp_roundtrip(lr, rng.randint(20, 78))

    noise_prob = 0.15 if degradation == "mixed-light" else 0.7
    if degradation == "phone-real":
        noise_prob = 0.90
    if rng.random() < noise_prob:
        if degradation == "mixed-light":
            gray_sigma = rng.uniform(0.5, 3.0)
            color_sigma = rng.uniform(0.0, 1.5)
        elif degradation == "mixed-real":
            gray_sigma = rng.uniform(0.5, 8.0)
            color_sigma = rng.uniform(0.0, 5.0)
        else:
            gray_sigma = rng.uniform(1.0, 12.0)
            color_sigma = rng.uniform(0.5, 8.0)
        lr = _add_rgb_noise(lr, rng, gray_sigma=gray_sigma, color_sigma=color_sigma)
    if degradation in {"mixed-real", "phone-real"} and rng.random() < (0.45 if degradation == "mixed-real" else 0.70):
        lr = _add_chroma_noise(lr, rng, rng.uniform(2.0, 14.0 if degradation == "phone-real" else 10.0))
    if degradation == "phone-real" and rng.random() < 0.25:
        lr = _add_banding(lr, rng, rng.uniform(0.3, 2.5))
    if degradation in {"mixed-real", "phone-real"} and rng.random() < (0.15 if degradation == "mixed-real" else 0.25):
        lr = lr.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.15, 0.8)))
    if degradation == "phone-real" and rng.random() < 0.20:
        lr = lr.filter(ImageFilter.UnsharpMask(radius=rng.uniform(0.5, 1.4), percent=rng.randint(60, 180), threshold=2))

    return lr


def _resolve_degradation(degradation: str, rng: random.Random) -> str:
    if degradation == "mixed-balanced":
        sample = rng.random()
        if sample < 0.30:
            return "bicubic"
        if sample < 0.70:
            return "mixed-light"
        return "mixed-real"
    if degradation == "mixed-sharp":
        sample = rng.random()
        if sample < 0.55:
            return "bicubic"
        if sample < 0.85:
            return "mixed-light"
        return "mixed-real"
    if degradation == "mixed-denoise":
        sample = rng.random()
        if sample < 0.10:
            return "bicubic"
        if sample < 0.35:
            return "mixed-real"
        return "phone-real"
    return degradation


def make_pair(
    hr: Image.Image,
    scale: int,
    *,
    input_mode: str = "bicubic_up",
    degradation: str = "bicubic",
    rng: random.Random | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = rng or random.Random()
    degradation = _resolve_degradation(degradation, rng)
    crop_size = hr.size[0]
    lr_size = crop_size // scale
    degraded = _degrade_before_resize(hr, rng, degradation)
    lr = _resize_to_lr(degraded, lr_size, rng, degradation)
    lr = _degrade_after_resize(lr, rng, degradation)
    if input_mode == "bicubic_up":
        x_image = lr.resize((crop_size, crop_size), Image.Resampling.BICUBIC)
    elif input_mode == "lr":
        x_image = lr
    else:
        raise ValueError(f"Unsupported input_mode: {input_mode}")
    x = np.asarray(x_image, dtype=np.float32) / 255.0
    y = np.asarray(hr, dtype=np.float32) / 255.0
    return x, y


def _aligned_random_crop(
    noisy: Image.Image,
    target: Image.Image,
    crop_size: int,
    rng: random.Random,
) -> tuple[Image.Image, Image.Image]:
    width = min(noisy.size[0], target.size[0])
    height = min(noisy.size[1], target.size[1])
    if width < crop_size or height < crop_size:
        scale = crop_size / min(width, height)
        new_size = (max(crop_size, round(width * scale)), max(crop_size, round(height * scale)))
        noisy = noisy.resize(new_size, Image.Resampling.BICUBIC)
        target = target.resize(new_size, Image.Resampling.BICUBIC)
        width, height = new_size
    else:
        noisy = noisy.crop((0, 0, width, height))
        target = target.crop((0, 0, width, height))

    left = rng.randint(0, width - crop_size)
    top = rng.randint(0, height - crop_size)
    box = (left, top, left + crop_size, top + crop_size)
    return noisy.crop(box), target.crop(box)


def make_pair_from_noisy_target(
    noisy: Image.Image,
    target: Image.Image,
    scale: int,
    *,
    input_mode: str = "bicubic_up",
    degradation: str = "bicubic",
    rng: random.Random | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    rng = rng or random.Random()
    degradation = _resolve_degradation(degradation, rng)
    crop_size = target.size[0]
    lr_size = crop_size // scale
    degraded = _degrade_before_resize(noisy, rng, degradation)
    lr = _resize_to_lr(degraded, lr_size, rng, degradation)
    lr = _degrade_after_resize(lr, rng, degradation)
    if input_mode == "bicubic_up":
        x_image = lr.resize((crop_size, crop_size), Image.Resampling.BICUBIC)
    elif input_mode == "lr":
        x_image = lr
    else:
        raise ValueError(f"Unsupported input_mode: {input_mode}")
    x = np.asarray(x_image, dtype=np.float32) / 255.0
    y = np.asarray(target, dtype=np.float32) / 255.0
    return x, y


def batch_iterator(
    image_dir: str | Path,
    *,
    batch_size: int,
    crop_size: int,
    scale: int,
    seed: int,
    input_mode: str = "bicubic_up",
    degradation: str = "bicubic",
    augment: bool = True,
    num_workers: int = 0,
    prefetch_batches: int = 8,
    preload: bool = False,
    preload_progress: bool = True,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if crop_size % scale != 0:
        raise ValueError(f"crop_size must be divisible by scale, got {crop_size=} {scale=}")

    if is_pair_dataset(image_dir):
        records = list_pair_records(image_dir)
        cache_paths = [path for record in records for path in (record.noisy, record.target)]
        cache = preload_image_cache(cache_paths, progress=preload_progress) if preload else None

        def make_pair_batch(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
            xs: list[np.ndarray] = []
            ys: list[np.ndarray] = []
            for _ in range(batch_size):
                record = rng.choice(records)
                noisy = _load_rgb(record.noisy, cache)
                target = _load_rgb(record.target, cache)
                noisy_crop, target_crop = _aligned_random_crop(noisy, target, crop_size, rng)
                if augment:
                    noisy_crop, target_crop = _augment_pair(noisy_crop, target_crop, rng)
                x, y = make_pair_from_noisy_target(
                    noisy_crop,
                    target_crop,
                    scale,
                    input_mode=input_mode,
                    degradation=degradation,
                    rng=rng,
                )
                xs.append(x)
                ys.append(y)
            return np.stack(xs), np.stack(ys)

        return _threaded_iterator(
            make_pair_batch,
            seed=seed,
            num_workers=num_workers,
            prefetch_batches=prefetch_batches,
        )

    images = list_images(image_dir)
    cache = preload_image_cache(images, progress=preload_progress) if preload else None

    def make_batch(rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for _ in range(batch_size):
            image = _load_rgb(rng.choice(images), cache)
            crop = _random_crop(image, crop_size, rng)
            if augment:
                crop = _augment(crop, rng)
            x, y = make_pair(
                crop,
                scale,
                input_mode=input_mode,
                degradation=degradation,
                rng=rng,
            )
            xs.append(x)
            ys.append(y)
        return np.stack(xs), np.stack(ys)

    return _threaded_iterator(
        make_batch,
        seed=seed,
        num_workers=num_workers,
        prefetch_batches=prefetch_batches,
    )


def _threaded_iterator(
    make_batch,
    *,
    seed: int,
    num_workers: int,
    prefetch_batches: int,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if num_workers <= 0:
        rng = random.Random(seed)
        while True:
            yield make_batch(rng)

    output: queue.Queue[tuple[np.ndarray, np.ndarray] | BaseException] = queue.Queue(
        maxsize=max(1, prefetch_batches)
    )

    def worker(worker_id: int) -> None:
        rng = random.Random(seed + 1_000_003 * (worker_id + 1))
        while True:
            try:
                output.put(make_batch(rng))
            except BaseException as exc:
                output.put(exc)
                return

    for worker_id in range(num_workers):
        thread = threading.Thread(
            target=worker,
            args=(worker_id,),
            name=f"sr-data-worker-{worker_id}",
            daemon=True,
        )
        thread.start()

    while True:
        item = output.get()
        if isinstance(item, BaseException):
            raise item
        yield item
