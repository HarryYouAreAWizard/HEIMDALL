#!/usr/bin/env python3
"""Train sparse PCA from raw VTEC files using on-the-fly polar MLT projection."""

from __future__ import annotations

import argparse
import datetime as dt
import multiprocessing as mp
from pathlib import Path

import h5py
import numpy as np

from build_polar_vtec_cube import (
    hdf5_files,
    polar_coordinate_grids,
    project_frame,
    source_mapping,
    total_frames,
)
from principal_component_analysis import find_sparse_principal_components


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="/data/nonie/tec_data", help="Directory containing TEC HDF5 files.")
    parser.add_argument("--output-dir", default="/home/juha/heimdall_polar_mlt_sparse_pca", help="Output directory.")
    parser.add_argument("--pixels", type=int, default=128, help="Polar image width and height.")
    parser.add_argument("--min-latitude", type=float, default=50.0, help="Magnetic latitude at disk edge.")
    parser.add_argument("--apex-height", type=float, default=350.0, help="Apex/QD conversion altitude in km.")
    parser.add_argument("--components", type=int, default=9, help="Number of sparse PCA vectors.")
    parser.add_argument("--iterations", type=int, default=8, help="Sparse ALS iterations.")
    parser.add_argument("--sample-frames", type=int, default=8192, help="Random frames used for training.")
    parser.add_argument(
        "--init-sample-columns",
        type=int,
        default=8192,
        help="Maximum number of training frames used only for initial SVD.",
    )
    parser.add_argument("--projection-jobs", type=int, default=16, help="Parallel workers for frame projection.")
    parser.add_argument("--als-jobs", type=int, default=16, help="Parallel workers for sparse ALS solves.")
    parser.add_argument("--ridge", type=float, default=1e-6, help="Small ridge term for weighted least squares.")
    parser.add_argument("--random-state", type=int, default=0, help="Random seed for training-frame selection.")
    parser.add_argument("--limit-files", type=int, default=None, help="Use only first N files for testing.")
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"], help="Training cube dtype.")
    return parser.parse_args()


def global_to_file_frame(global_indices: np.ndarray, samples_per_file: int) -> dict[int, list[tuple[int, int]]]:
    grouped: dict[int, list[tuple[int, int]]] = {}
    for output_idx, global_idx in enumerate(global_indices):
        file_idx = int(global_idx // samples_per_file)
        frame_idx = int(global_idx % samples_per_file)
        grouped.setdefault(file_idx, []).append((output_idx, frame_idx))
    return grouped


def _project_file_task(args: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    (
        file_idx,
        path,
        frame_items,
        pixels,
        min_latitude,
        apex_height,
        outside,
        dtype,
    ) = args
    with h5py.File(path, "r") as h5:
        tec = h5["Data/Array Layout/2D Parameters/tec"]
        latitudes = h5["Data/Array Layout/gdlat"][:]
        longitudes = h5["Data/Array Layout/glon"][:]
        timestamps_all = h5["Data/Array Layout/timestamps"][:]
        first_time = dt.datetime.fromtimestamp(float(timestamps_all[0]), tz=dt.timezone.utc).replace(tzinfo=None)
        flat_indices, radius, qd_longitude = source_mapping(
            latitudes,
            longitudes,
            first_time,
            pixels,
            min_latitude,
            apex_height,
        )
        frames = np.empty((len(frame_items), pixels, pixels), dtype=dtype)
        counts = np.empty((len(frame_items), pixels, pixels), dtype=np.uint16)
        output_indices = np.empty(len(frame_items), dtype=np.int64)
        timestamps = np.empty(len(frame_items), dtype=np.float64)
        for local_output_idx, (output_idx, frame_idx) in enumerate(frame_items):
            timestamp = float(timestamps_all[frame_idx])
            frame_time = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).replace(tzinfo=None)
            projected, pixel_counts = project_frame(
                tec[:, :, frame_idx],
                flat_indices,
                radius,
                qd_longitude,
                frame_time,
                pixels,
                outside,
            )
            frames[local_output_idx, :, :] = projected.astype(dtype, copy=False)
            counts[local_output_idx, :, :] = pixel_counts
            output_indices[local_output_idx] = output_idx
            timestamps[local_output_idx] = timestamp
    print(f"projected file {file_idx} with {len(frame_items)} training frames", flush=True)
    return output_indices, frames, counts, timestamps


def load_projected_training_cube(
    files: list[Path],
    sample_indices: np.ndarray,
    samples_per_file: int,
    pixels: int,
    min_latitude: float,
    apex_height: float,
    projection_jobs: int,
    dtype: str,
    outside: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped = global_to_file_frame(sample_indices, samples_per_file)
    cube = np.full((pixels, pixels, len(sample_indices)), np.nan, dtype=dtype)
    coverage = np.zeros((pixels, pixels), dtype=np.uint32)
    timestamps = np.empty(len(sample_indices), dtype=np.float64)
    tasks = [
        (file_idx, files[file_idx], items, pixels, min_latitude, apex_height, outside, dtype)
        for file_idx, items in grouped.items()
    ]
    jobs = max(1, min(int(projection_jobs), len(tasks)))
    if jobs == 1:
        results = map(_project_file_task, tasks)
        for output_indices, frames, counts, task_timestamps in results:
            for k, output_idx in enumerate(output_indices):
                cube[:, :, output_idx] = frames[k]
                coverage += (counts[k] > 0).astype(np.uint32)
                timestamps[output_idx] = task_timestamps[k]
    else:
        with mp.get_context("fork").Pool(processes=jobs) as pool:
            for output_indices, frames, counts, task_timestamps in pool.imap_unordered(_project_file_task, tasks):
                for k, output_idx in enumerate(output_indices):
                    cube[:, :, output_idx] = frames[k]
                    coverage += (counts[k] > 0).astype(np.uint32)
                    timestamps[output_idx] = task_timestamps[k]
    return cube, coverage, timestamps


def main() -> None:
    args = parse_args()
    files = hdf5_files(args.input_dir, args.limit_files)
    _, _, total_times = total_frames(files)
    with h5py.File(files[0], "r") as h5:
        samples_per_file = h5["Data/Array Layout/2D Parameters/tec"].shape[2]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    magnetic_latitude, mlt, valid_projection_mask, pixel_area, pca_area_weight_sqrt = polar_coordinate_grids(
        args.pixels,
        args.min_latitude,
    )
    outside = ~valid_projection_mask

    rng = np.random.default_rng(args.random_state)
    sample_count = min(args.sample_frames, total_times)
    sample_indices = np.sort(rng.choice(total_times, size=sample_count, replace=False))
    np.save(output_dir / "training_indices.npy", sample_indices)
    np.save(output_dir / "magnetic_latitude.npy", magnetic_latitude)
    np.save(output_dir / "mlt.npy", mlt)
    np.save(output_dir / "valid_projection_mask.npy", valid_projection_mask)
    np.save(output_dir / "pixel_area_weight.npy", pixel_area)
    np.save(output_dir / "pca_area_weight_sqrt.npy", pca_area_weight_sqrt)
    print(f"training frames={sample_count} / {total_times}", flush=True)

    training_cube, coverage, timestamps = load_projected_training_cube(
        files,
        sample_indices,
        samples_per_file,
        args.pixels,
        args.min_latitude,
        args.apex_height,
        args.projection_jobs,
        args.dtype,
        outside,
    )
    np.save(output_dir / "training_timestamps.npy", timestamps)
    np.save(output_dir / "training_observed_count.npy", coverage)
    np.save(output_dir / "training_observed_fraction.npy", coverage / float(sample_count))
    print(
        "training observed fraction: "
        f"min={np.nanmin(coverage[valid_projection_mask] / float(sample_count)):.4g} "
        f"mean={np.nanmean(coverage[valid_projection_mask] / float(sample_count)):.4g} "
        f"max={np.nanmax(coverage[valid_projection_mask] / float(sample_count)):.4g}",
        flush=True,
    )

    component_images, component_columns, coefficients, mean = find_sparse_principal_components(
        training_cube,
        args.components,
        n_iterations=args.iterations,
        ridge=args.ridge,
        sample_columns=min(args.init_sample_columns, sample_count),
        random_state=args.random_state,
        n_jobs=args.als_jobs,
        spatial_weights=pca_area_weight_sqrt,
        verbose=True,
    )
    np.save(output_dir / "components_polar_mlt.npy", component_images)
    np.save(output_dir / "component_columns_polar_mlt.npy", component_columns)
    np.save(output_dir / "time_series_training.npy", coefficients)
    np.save(output_dir / "mean_polar_mlt.npy", mean.reshape(args.pixels, args.pixels))
    print(f"saved on-the-fly polar MLT sparse PCA to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
