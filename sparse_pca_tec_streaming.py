#!/usr/bin/env python3
"""Estimate global local-time TEC sparse PCA from HDF5 files.

Training is sparse in time: only a random subset of time frames is loaded to
estimate the principal vectors.  Afterward, the final vectors are projected onto
all time frames in chunks so the full time axis is never held in memory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np

from principal_component_analysis import (
    compute_sparse_time_coefficients,
    find_sparse_principal_components,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="/data/nonie/tec_data", help="Directory with daily TEC HDF5 files.")
    parser.add_argument("--output-dir", default="/home/juha/heimdall_global_results", help="Output directory.")
    parser.add_argument("--components", type=int, default=9, help="Number of principal components.")
    parser.add_argument("--iterations", type=int, default=8, help="Sparse ALS iterations.")
    parser.add_argument("--jobs", type=int, default=1, help="Worker processes for ALS/projection solves.")
    parser.add_argument("--sample-frames", type=int, default=8192, help="Random time frames used for PCA training.")
    parser.add_argument("--project-chunk", type=int, default=288, help="Frames per projection chunk.")
    parser.add_argument("--ridge", type=float, default=1e-6, help="Small ridge term for weighted least squares.")
    parser.add_argument("--random-state", type=int, default=0, help="Random seed for training-frame selection.")
    parser.add_argument("--lat-min", type=float, default=-90.0, help="Minimum latitude of global TEC grid.")
    parser.add_argument("--lat-max", type=float, default=90.0, help="Maximum latitude of global TEC grid.")
    parser.add_argument(
        "--no-local-time-rotate",
        action="store_true",
        help="Do not rotate each frame into local-time coordinates.",
    )
    parser.add_argument("--limit-files", type=int, default=None, help="Use only the first N files, for testing.")
    parser.add_argument("--dtype", default="float64", choices=["float32", "float64"], help="Training cube dtype.")
    return parser.parse_args()


def hdf5_files(input_dir: str, limit_files: int | None = None) -> list[Path]:
    files = sorted(Path(input_dir).glob("*.hdf5"))
    if limit_files is not None:
        files = files[:limit_files]
    if not files:
        raise FileNotFoundError(f"No HDF5 files found in {input_dir}")
    return files


def grid_shape(files: list[Path]) -> tuple[int, int, int, int]:
    with h5py.File(files[0], "r") as h5:
        data = h5["Data/Array Layout/2D Parameters/tec"]
        n_lat, n_lon, samples_per_file = data.shape
    return n_lat, n_lon, samples_per_file, samples_per_file * len(files)


def local_time_roll(n_lon: int, samples_per_file: int, global_time_index: int) -> int:
    """Match the legacy center_midday rotation without materializing the cube."""
    roll = int(n_lon / samples_per_file * global_time_index)
    roll = (roll + n_lon // 2) % n_lon
    return roll


def read_frame_cube(
    files: list[Path],
    indices: np.ndarray,
    shape: tuple[int, int],
    samples_per_file: int,
    local_time_rotate: bool,
    dtype: str,
) -> np.ndarray:
    cube = np.empty((shape[0], shape[1], len(indices)), dtype=dtype)
    file_indices = indices // samples_per_file
    for file_index in np.unique(file_indices):
        output_indices = np.flatnonzero(file_indices == file_index)
        with h5py.File(files[int(file_index)], "r") as h5:
            data = h5["Data/Array Layout/2D Parameters/tec"]
            for output_index in output_indices:
                if output_index % 100 == 0:
                    print(f"loading frame {output_index + 1}/{len(indices)}", flush=True)
                global_time_index = int(indices[output_index])
                time_index = global_time_index % samples_per_file
                frame = np.asarray(data[:, :, time_index])
                if local_time_rotate:
                    frame = np.roll(frame, local_time_roll(shape[1], samples_per_file, global_time_index), axis=1)
                cube[:, :, output_index] = frame
    return cube


def project_all_times(
    files: list[Path],
    component_columns: np.ndarray,
    mean: np.ndarray,
    output_path: Path,
    total_times: int,
    shape: tuple[int, int],
    samples_per_file: int,
    local_time_rotate: bool,
    project_chunk: int,
    ridge: float,
    jobs: int,
    lat_min: float,
    lat_max: float,
    dtype: str,
) -> None:
    coefficients = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype="float64",
        shape=(component_columns.shape[1], total_times),
    )
    for start in range(0, total_times, project_chunk):
        stop = min(start + project_chunk, total_times)
        print(f"projecting frames {start}:{stop}", flush=True)
        indices = np.arange(start, stop)
        chunk = read_frame_cube(
            files,
            indices,
            shape,
            samples_per_file,
            local_time_rotate=local_time_rotate,
            dtype=dtype,
        )
        coefficients[:, start:stop] = compute_sparse_time_coefficients(
            component_columns,
            chunk,
            mean=mean,
            ridge=ridge,
            n_jobs=jobs,
            lat_min=lat_min,
            lat_max=lat_max,
        )
        coefficients.flush()


def main() -> None:
    args = parse_args()
    files = hdf5_files(args.input_dir, limit_files=args.limit_files)
    n_lat, n_lon, samples_per_file, total_times = grid_shape(files)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    local_time_rotate = not args.no_local_time_rotate

    print(f"files={len(files)} grid=({n_lat}, {n_lon}) samples_per_file={samples_per_file} total_times={total_times}")
    print(f"local_time_rotate={local_time_rotate}")

    rng = np.random.default_rng(args.random_state)
    sample_count = min(args.sample_frames, total_times)
    sample_indices = np.sort(rng.choice(total_times, size=sample_count, replace=False))
    np.save(output_dir / "global_sparse_training_indices.npy", sample_indices)
    print(f"training frames={sample_count}")

    training_cube = read_frame_cube(
        files,
        sample_indices,
        (n_lat, n_lon),
        samples_per_file,
        local_time_rotate=local_time_rotate,
        dtype=args.dtype,
    )

    component_images, component_columns, training_coefficients, mean = find_sparse_principal_components(
        training_cube,
        args.components,
        n_iterations=args.iterations,
        ridge=args.ridge,
        sample_columns=sample_count,
        random_state=args.random_state,
        n_jobs=args.jobs,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        verbose=True,
    )

    np.save(output_dir / "components_global_sparse_local_time.npy", component_images)
    np.save(output_dir / "time_series_global_sparse_training.npy", training_coefficients)
    np.save(output_dir / "mean_global_sparse_local_time.npy", mean.reshape(n_lat, n_lon))
    print(f"saved global components and mean to {output_dir}", flush=True)

    project_all_times(
        files,
        component_columns,
        mean,
        output_dir / "time_series_global_sparse_local_time.npy",
        total_times,
        (n_lat, n_lon),
        samples_per_file,
        local_time_rotate=local_time_rotate,
        project_chunk=args.project_chunk,
        ridge=args.ridge,
        jobs=args.jobs,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        dtype=args.dtype,
    )
    print("done")


if __name__ == "__main__":
    main()
