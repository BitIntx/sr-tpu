from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare noisy/clean pair manifests for real-world x4 restoration training."
    )
    parser.add_argument("--sidd-root", default="~/datasets/sr/real/sidd-small-srgb")
    parser.add_argument("--clean-hr-root", action="append", default=[])
    parser.add_argument("--out", default="~/datasets/sr/prepared/sr_real_x4_v1")
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--min-side", type=int, default=256)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sidd-repeat", type=int, default=12)
    parser.add_argument("--clean-repeat", type=int, default=1)
    parser.add_argument("--clean-limit", type=int, default=4000)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def list_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def image_info(path: Path) -> dict[str, int]:
    with Image.open(path) as image:
        width, height = ImageOps.exif_transpose(image).size
    return {"width": width, "height": height, "min_side": min(width, height)}


def _case_replace(text: str, old: str, new: str) -> str:
    return re.sub(re.escape(old), new, text, flags=re.IGNORECASE)


def _pair_key(path: Path, root: Path) -> str:
    rel = path.relative_to(root).as_posix().lower()
    rel = re.sub(r"noisy[_-]?srgb|gt[_-]?srgb", "pair_srgb", rel)
    rel = re.sub(r"noisy|ground[_-]?truth|gt", "pair", rel)
    rel = re.sub(r"input|target|clean", "pair", rel)
    return rel


def find_sidd_pairs(root: Path, *, min_side: int) -> list[dict]:
    images = list_images(root)
    target_by_key: dict[str, Path] = {}
    for path in images:
        lower = path.as_posix().lower()
        if "gt" in lower or "ground" in lower or "clean" in lower or "target" in lower:
            target_by_key[_pair_key(path, root)] = path

    pairs = []
    seen: set[tuple[Path, Path]] = set()
    for noisy in tqdm(images, desc="scan SIDD pairs"):
        lower = noisy.as_posix().lower()
        if "noisy" not in lower and "input" not in lower:
            continue

        candidates = []
        for old, new in (
            ("NOISY_SRGB", "GT_SRGB"),
            ("Noisy_SRGB", "GT_SRGB"),
            ("noisy_srgb", "gt_srgb"),
            ("NOISY", "GT"),
            ("Noisy", "GT"),
            ("noisy", "gt"),
            ("input", "target"),
        ):
            candidate = Path(_case_replace(noisy.as_posix(), old, new))
            candidates.append(candidate)

        target = next((candidate for candidate in candidates if candidate.exists()), None)
        if target is None:
            target = target_by_key.get(_pair_key(noisy, root))
        if target is None:
            continue

        key = (noisy.resolve(), target.resolve())
        if key in seen:
            continue
        seen.add(key)
        try:
            noisy_info = image_info(noisy)
            target_info = image_info(target)
        except OSError:
            continue
        if min(noisy_info["min_side"], target_info["min_side"]) < min_side:
            continue
        pairs.append(
            {
                "name": noisy.relative_to(root).with_suffix("").as_posix().replace("/", "__"),
                "source": "sidd",
                "kind": "real_noisy_pair",
                "noisy": str(noisy.resolve()),
                "target": str(target.resolve()),
                "width": min(noisy_info["width"], target_info["width"]),
                "height": min(noisy_info["height"], target_info["height"]),
            }
        )
    return sorted(pairs, key=lambda row: row["name"])


def find_clean_self_pairs(
    roots: list[Path],
    *,
    min_side: int,
    limit: int,
    seed: int,
) -> list[dict]:
    rows = []
    for root in roots:
        for path in list_images(root):
            try:
                info = image_info(path)
            except OSError:
                continue
            if info["min_side"] < min_side:
                continue
            rows.append(
                {
                    "name": f"clean__{root.name}__{path.stem}",
                    "source": root.name,
                    "kind": "synthetic_clean_self_pair",
                    "noisy": str(path.resolve()),
                    "target": str(path.resolve()),
                    **info,
                }
            )
    rng = random.Random(seed)
    rng.shuffle(rows)
    if limit > 0:
        rows = rows[:limit]
    return sorted(rows, key=lambda row: row["name"])


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def split_rows(rows: list[dict], *, val_ratio: float, seed: int) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    val_count = max(1, round(len(shuffled) * val_ratio)) if shuffled else 0
    val_names = {row["name"] for row in shuffled[:val_count]}
    train = [row for row in rows if row["name"] not in val_names]
    val = [row for row in rows if row["name"] in val_names]
    return train, val


def repeat_rows(rows: list[dict], repeats: int) -> list[dict]:
    if repeats <= 1:
        return rows
    repeated = []
    for row in rows:
        for repeat in range(repeats):
            copied = dict(row)
            copied["repeat"] = repeat
            copied["name"] = f"{row['name']}__r{repeat:02d}"
            repeated.append(copied)
    return repeated


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out).expanduser()
    if args.overwrite and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sidd_root = Path(args.sidd_root).expanduser()
    sidd_pairs = find_sidd_pairs(sidd_root, min_side=args.min_side)
    clean_roots = [Path(root).expanduser() for root in args.clean_hr_root]
    clean_pairs = find_clean_self_pairs(
        clean_roots,
        min_side=args.min_side,
        limit=args.clean_limit,
        seed=args.seed + 17,
    )

    sidd_train, sidd_val = split_rows(sidd_pairs, val_ratio=args.val_ratio, seed=args.seed)
    clean_train, clean_val = split_rows(clean_pairs, val_ratio=min(args.val_ratio, 0.05), seed=args.seed + 1)

    train_rows = repeat_rows(sidd_train, args.sidd_repeat) + repeat_rows(clean_train, args.clean_repeat)
    val_rows = sidd_val + clean_val
    random.Random(args.seed + 2).shuffle(train_rows)

    write_jsonl(out_dir / "train_pairs" / "pairs.jsonl", train_rows)
    write_jsonl(out_dir / "val_pairs" / "pairs.jsonl", val_rows)

    metadata = {
        "name": "sr_real_x4_v1",
        "scale": args.scale,
        "min_side": args.min_side,
        "sidd_root": str(sidd_root),
        "clean_hr_roots": [str(root) for root in clean_roots],
        "counts": {
            "sidd_pairs": len(sidd_pairs),
            "clean_self_pairs": len(clean_pairs),
            "train_pairs": len(train_rows),
            "val_pairs": len(val_rows),
        },
        "notes": [
            "Rows with kind=real_noisy_pair use SIDD noisy sRGB as the LR source before x4 downsampling.",
            "Rows with kind=synthetic_clean_self_pair use clean HR as both source and target; train with phone-real/mixed-denoise degradation.",
        ],
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "README.md").write_text(
        "\n".join(
            [
                "# sr_real_x4_v1",
                "",
                "Prepared pair-manifest dataset for real-world x4 restoration.",
                "",
                "Use `train_pairs` and `val_pairs` with `train.py`; the loader auto-detects `pairs.jsonl`.",
                "",
                "Recommended degradation: `mixed-denoise` or `phone-real`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(json.dumps(metadata["counts"], indent=2, sort_keys=True), flush=True)
    print(f"prepared: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
