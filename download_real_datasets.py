from __future__ import annotations

import argparse
import hashlib
import subprocess
import zipfile
from pathlib import Path


DATASETS = {
    "sidd-small-srgb": {
        "url": "http://130.63.97.225/share/SIDD_Small_sRGB_Only.zip",
        "filename": "SIDD_Small_sRGB_Only.zip",
        "md5": "796971867583bf14677dcae510e52538",
        "description": "SIDD Small sRGB noisy/GT image pairs, about 6 GB compressed.",
    },
    "sidd-medium-srgb": {
        "url": "http://130.63.97.225/share/SIDD_Medium_Srgb.zip",
        "filename": "SIDD_Medium_Srgb.zip",
        "md5": "",
        "description": "SIDD Medium sRGB noisy/GT image pairs, about 12 GB compressed.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download real-world restoration datasets.")
    parser.add_argument("--root", default="~/datasets/sr/real")
    parser.add_argument(
        "--dataset",
        choices=sorted(DATASETS),
        default="sidd-small-srgb",
    )
    parser.add_argument("--extract", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--check-md5", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def md5sum(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "curl",
        "-L",
        "-C",
        "-",
        "--retry",
        "8",
        "--retry-delay",
        "5",
        "--fail",
        "-o",
        str(path),
        url,
    ]
    print("download:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def extract_zip(archive: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        members = zf.infolist()
        for index, member in enumerate(members, start=1):
            zf.extract(member, out_dir)
            if index % 100 == 0 or index == len(members):
                print(f"extract {index}/{len(members)} -> {out_dir}", flush=True)


def main() -> None:
    args = parse_args()
    spec = DATASETS[args.dataset]
    root = Path(args.root).expanduser()
    archive = root / "archives" / str(spec["filename"])
    print(f"dataset={args.dataset}: {spec['description']}", flush=True)
    download(str(spec["url"]), archive)

    expected_md5 = str(spec.get("md5") or "")
    if args.check_md5 and expected_md5:
        actual_md5 = md5sum(archive)
        if actual_md5 != expected_md5:
            raise RuntimeError(f"MD5 mismatch for {archive}: expected {expected_md5}, got {actual_md5}")
        print(f"md5 ok: {actual_md5}", flush=True)

    if args.extract:
        extract_dir = root / args.dataset
        extract_zip(archive, extract_dir)
        print(f"extracted: {extract_dir}", flush=True)

    print(f"archive: {archive}", flush=True)


if __name__ == "__main__":
    main()
