#!/usr/bin/env python3
"""Build a global TEC cube from the daily HDF5 files.

The output is written incrementally as an ``.npy`` memmap so the full
180 x 360 x time array does not need to be held in memory while building.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="/data/nonie/tec_data", help="Directory with daily TEC HDF5 files.")
    parser.add_argument(
        "--output",
        default="/home/juha/heimdall_global_results/raw_global_tec.npy",
        help="Output global TEC cube.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Only use the first N HDF5 files.")
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"], help="Output dtype.")
    return parser.parse_args()


def tec_dataset(path: Path) -> h5py.Dataset:
    h5 = h5py.File(path, "r")
    return h5["Data/Array Layout/2D Parameters/tec"]


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    files = sorted(input_dir.glob("*.hdf5"))
    if args.limit is not None:
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"No HDF5 files found in {input_dir}")

    with h5py.File(files[0], "r") as h5:
        first = h5["Data/Array Layout/2D Parameters/tec"]
        lat_count, lon_count, samples_per_file = first.shape

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    shape = (lat_count, lon_count, samples_per_file * len(files))
    print(f"Writing {output}")
    print(f"shape={shape} dtype={args.dtype} files={len(files)}")
    cube = np.lib.format.open_memmap(output, mode="w+", dtype=args.dtype, shape=shape)

    for file_index, path in enumerate(files):
        start = file_index * samples_per_file
        stop = start + samples_per_file
        print(f"{file_index + 1}/{len(files)} {path.name} -> samples {start}:{stop}", flush=True)
        with h5py.File(path, "r") as h5:
            data = h5["Data/Array Layout/2D Parameters/tec"]
            if data.shape != (lat_count, lon_count, samples_per_file):
                raise ValueError(f"Unexpected shape in {path}: {data.shape}")
            cube[:, :, start:stop] = data[:, :, :]
        cube.flush()

    print(f"Saved {output}")


if __name__ == "__main__":
    main()
