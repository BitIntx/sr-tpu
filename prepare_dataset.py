from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class Source:
    name: str
    root: Path
    split: str
    tier: str
    repeats: int = 1


SOURCES = [
    Source(
        name="div2k_train",
        root=Path("~/datasets/sr/DIV2K/DIV2K_train_HR").expanduser(),
        split="train",
        tier="clean",
        repeats=4,
    ),
    Source(
        name="ost_train",
        root=Path("~/datasets/sr/OST/extracted/OutdoorSceneTrain_v2").expanduser(),
        split="train",
        tier="clean",
        repeats=1,
    ),
    Source(
        name="ost_seg_images",
        root=Path("~/datasets/sr/OST/extracted/OutdoorSeg/images").expanduser(),
        split="train",
        tier="jpeg_extra",
        repeats=1,
    ),
    Source(
        name="div2k_valid",
        root=Path("~/datasets/sr/DIV2K/DIV2K_valid_HR").expanduser(),
        split="val",
        tier="clean",
        repeats=1,
    ),
    Source(
        name="ost_test300",
        root=Path("~/datasets/sr/OST/extracted/OutdoorSceneTest300/OutdoorSceneTest300").expanduser(),
        split="val",
        tier="clean",
        repeats=1,
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare local SR training dataset links and validation pairs.")
    parser.add_argument("--out", default="~/datasets/sr/prepared/sr_x4_v1")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--min-side", type=int, default=256)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def list_images(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_info(path: Path) -> dict[str, int | str]:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        width, height = image.size
    return {
        "width": width,
        "height": height,
        "min_side": min(width, height),
    }


def safe_name(source: Source, path: Path, repeat: int | None = None) -> str:
    rel = path.relative_to(source.root)
    stem = "__".join(rel.with_suffix("").parts)
    suffix = path.suffix.lower()
    repeat_part = f"__r{repeat}" if repeat is not None else ""
    return f"{source.name}{repeat_part}__{stem}{suffix}"


def rel_symlink(target: Path, link: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        link.unlink()
    link.symlink_to(target)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def modcrop(image: Image.Image, scale: int) -> Image.Image:
    width, height = image.size
    width -= width % scale
    height -= height % scale
    return image.crop((0, 0, width, height))


def save_validation_pair(src_path: Path, name: str, out_dir: Path, scale: int) -> dict[str, str | int]:
    with Image.open(src_path) as image:
        hr = modcrop(ImageOps.exif_transpose(image).convert("RGB"), scale)
    width, height = hr.size
    lr = hr.resize((width // scale, height // scale), Image.Resampling.BICUBIC)
    lr_up = lr.resize((width, height), Image.Resampling.BICUBIC)

    hr_path = out_dir / "hr" / f"{name}.png"
    lr_path = out_dir / "lr" / f"{name}.png"
    lr_up_path = out_dir / "lr_up" / f"{name}.png"
    hr_path.parent.mkdir(parents=True, exist_ok=True)
    lr_path.parent.mkdir(parents=True, exist_ok=True)
    lr_up_path.parent.mkdir(parents=True, exist_ok=True)

    hr.save(hr_path)
    lr.save(lr_path)
    lr_up.save(lr_up_path)
    return {
        "hr": str(hr_path),
        "lr": str(lr_path),
        "lr_up": str(lr_up_path),
        "width": width,
        "height": height,
        "scale": scale,
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out).expanduser()
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifests_dir = out_dir / "manifests"
    train_clean_rows: list[dict] = []
    train_balanced_rows: list[dict] = []
    train_mixed_rows: list[dict] = []
    val_rows: list[dict] = []
    source_summary: dict[str, dict[str, int | str]] = {}

    for source in SOURCES:
        if not source.root.exists():
            raise FileNotFoundError(source.root)
        files = list_images(source.root)
        kept_rows: list[dict] = []
        skipped_small = 0
        for path in tqdm(files, desc=f"scan {source.name}"):
            info = image_info(path)
            if int(info["min_side"]) < args.min_side:
                skipped_small += 1
                continue
            row = {
                "source": source.name,
                "split": source.split,
                "tier": source.tier,
                "path": str(path),
                "relative_path": str(path.relative_to(source.root)),
                **info,
            }
            kept_rows.append(row)

        source_summary[source.name] = {
            "root": str(source.root),
            "split": source.split,
            "tier": source.tier,
            "total_images": len(files),
            "kept_images": len(kept_rows),
            "skipped_min_side": skipped_small,
            "repeats_for_balanced": source.repeats,
        }

        if source.split == "train":
            for row in kept_rows:
                src_path = Path(str(row["path"]))
                if source.tier == "clean":
                    clean_link = out_dir / "train_clean_hr" / safe_name(source, src_path)
                    rel_symlink(src_path, clean_link)
                    clean_row = row | {"link_path": str(clean_link)}
                    train_clean_rows.append(clean_row)

                    for repeat in range(source.repeats):
                        balanced_link = out_dir / "train_balanced_hr" / safe_name(source, src_path, repeat)
                        rel_symlink(src_path, balanced_link)
                        train_balanced_rows.append(row | {"link_path": str(balanced_link), "repeat": repeat})

                    mixed_link = out_dir / "train_mixed_hr" / safe_name(source, src_path)
                    rel_symlink(src_path, mixed_link)
                    train_mixed_rows.append(row | {"link_path": str(mixed_link), "repeat": 0})
                else:
                    extra_link = out_dir / "train_jpeg_extra_hr" / safe_name(source, src_path)
                    rel_symlink(src_path, extra_link)
                    mixed_link = out_dir / "train_mixed_hr" / safe_name(source, src_path)
                    rel_symlink(src_path, mixed_link)
                    train_mixed_rows.append(row | {"link_path": str(mixed_link), "repeat": 0})
        else:
            for row in kept_rows:
                src_path = Path(str(row["path"]))
                val_link = out_dir / "val_hr" / safe_name(source, src_path)
                rel_symlink(src_path, val_link)
                val_name = val_link.with_suffix("").name
                pair_info = save_validation_pair(src_path, val_name, out_dir / "val_x4_bicubic", args.scale)
                val_rows.append(row | {"link_path": str(val_link), "pair": pair_info})

    write_jsonl(manifests_dir / "train_clean_hr.jsonl", train_clean_rows)
    write_jsonl(manifests_dir / "train_balanced_hr.jsonl", train_balanced_rows)
    write_jsonl(manifests_dir / "train_mixed_hr.jsonl", train_mixed_rows)
    write_jsonl(manifests_dir / "val_hr.jsonl", val_rows)

    metadata = {
        "name": "sr_x4_v1",
        "scale": args.scale,
        "min_side": args.min_side,
        "layout": {
            "train_clean_hr": "DIV2K train HR + OST train PNG, no repeats",
            "train_balanced_hr": "clean training set with DIV2K repeated for sampling balance",
            "train_jpeg_extra_hr": "OST segmentation JPEG images, optional extra data",
            "train_mixed_hr": "clean training set + optional JPEG extra images",
            "val_hr": "DIV2K valid HR + OST test300 HR",
            "val_x4_bicubic": "modcropped HR, generated x4 LR, and bicubic-upscaled LR",
        },
        "sources": source_summary,
        "counts": {
            "train_clean_hr": len(train_clean_rows),
            "train_balanced_hr": len(train_balanced_rows),
            "train_mixed_hr": len(train_mixed_rows),
            "val_hr": len(val_rows),
        },
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# sr_x4_v1",
                "",
                "Prepared local x4 super-resolution dataset.",
                "",
                "Use `train_balanced_hr` for the first clean baseline:",
                "",
                "```bash",
                "python train.py --data ~/datasets/sr/prepared/sr_x4_v1/train_balanced_hr --scale 4 --crop-size 256",
                "```",
                "",
                "Use `train_mixed_hr` only when you want to include the extra OST JPEG images.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    print(json.dumps(metadata["counts"], indent=2, sort_keys=True))
    print(f"prepared: {out_dir}")


if __name__ == "__main__":
    main()
