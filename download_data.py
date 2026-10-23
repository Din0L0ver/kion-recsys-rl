#!/usr/bin/env python3
"""Download the public KION interactions and Russian item metadata."""

from __future__ import annotations

import argparse
import urllib.request
import zipfile
from pathlib import Path


BASE = "https://raw.githubusercontent.com/irsafilo/KION_DATASET/refs/heads/main"
# data_original.zip contains the complete 5.47M-row interactions file. The
# standalone interactions.csv.zip currently published in the repository ends
# at exactly 50 MiB in the middle of a CSV row, so it is not used.
FILES = ("data_original.zip",)


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        print(f"exists: {destination}")
        return
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"download: {url}")
    urllib.request.urlretrieve(url, temporary)
    temporary.replace(destination)


def ensure_valid_archive(url: str, destination: Path) -> None:
    if destination.exists():
        try:
            with zipfile.ZipFile(destination) as zipped:
                if zipped.testzip() is None:
                    print(f"valid archive exists: {destination}")
                    return
        except zipfile.BadZipFile:
            pass
        print(f"incomplete archive will be replaced: {destination}")
        destination.unlink()
    download(url, destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/raw"))
    args = parser.parse_args()
    for name in FILES:
        archive = args.output / name
        ensure_valid_archive(f"{BASE}/{name}", archive)
        with zipfile.ZipFile(archive) as zipped:
            zipped.extractall(args.output)
            print(f"extracted: {', '.join(zipped.namelist())}")
    interactions = args.output / "data_original" / "interactions.csv"
    with interactions.open("rb") as handle:
        rows = sum(1 for _ in handle) - 1
    if rows != 5_476_251:
        raise RuntimeError(f"Expected 5,476,251 interactions, found {rows:,}: {interactions}")
    print(f"validated: {rows:,} interactions")


if __name__ == "__main__":
    main()

