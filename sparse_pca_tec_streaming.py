#!/usr/bin/env python3
"""Estimate global local-time TEC sparse PCA from HDF5 files.

Training is sparse in time: only a random subset of time frames is loaded to
estimate the principal vectors.  Afterward, the final vectors are projected onto
all time frames in chunks so the full time axis is never held in memory.
"""

from __future__ import annotations

import argparse
import datetime as dt
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
        "--coordinate-system",
        choices=["geographic", "local_time", "mlt"],
        default="local_time",
        help="Coordinate grid used for PCA.",
    )
    parser.add_argument(
        "--no-local-time-rotate",
        action="store_true",
        help="Deprecated alias for --coordinate-system geographic.",
    )
    parser.add_argument("--apex-height", type=float, default=350.0, help="Apex/QD conversion altitude in km.")
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


def _metadata_value(line: object) -> str:
    text = metadata_text(line)
    return text.split()[1]


def metadata_text(line: object) -> str:
    if isinstance(line, np.void):
        line = line[0]
    if isinstance(line, bytes):
        return line.decode("utf-8", errors="ignore").strip()
    return str(line).strip()


def file_time_bounds(path: Path) -> tuple[dt.datetime, dt.datetime]:
    values = {}
    with h5py.File(path, "r") as h5:
        for line in h5["Metadata"]["Experiment Notes"]:
            text = metadata_text(line)
            for key in ["IBYRE", "IBDTE", "IBHME", "IBCSE", "IEYRE", "IEDTE", "IEHME", "IECSE"]:
                if text.startswith(key):
                    values[key] = _metadata_value(line)

    start_month_day = values["IBDTE"].zfill(4)
    start_hour_minute = values["IBHME"].zfill(4)
    end_month_day = values["IEDTE"].zfill(4)
    end_hour_minute = values["IEHME"].zfill(4)
    start = dt.datetime(
        int(values["IBYRE"]),
        int(start_month_day[:2]),
        int(start_month_day[2:4]),
        int(start_hour_minute[:2]),
        int(start_hour_minute[2:4]),
        int(float(values["IBCSE"])) // 100,
    )
    end = dt.datetime(
        int(values["IEYRE"]),
        int(end_month_day[:2]),
        int(end_month_day[2:4]),
        int(end_hour_minute[:2]),
        int(end_hour_minute[2:4]),
        int(float(values["IECSE"])) // 100,
    )
    return start, end


def frame_datetime(
    file_bounds: list[tuple[dt.datetime, dt.datetime]],
    global_time_index: int,
    samples_per_file: int,
) -> dt.datetime:
    file_index = global_time_index // samples_per_file
    time_index = global_time_index % samples_per_file
    start, end = file_bounds[file_index]
    if samples_per_file <= 1:
        return start
    return start + (end - start) * (time_index / (samples_per_file - 1))


def local_time_roll(n_lon: int, samples_per_file: int, global_time_index: int) -> int:
    """Match the legacy center_midday rotation without materializing the cube."""
    roll = int(n_lon / samples_per_file * global_time_index)
    roll = (roll + n_lon // 2) % n_lon
    return roll


def prepare_mlt_mapping(
    files: list[Path],
    shape: tuple[int, int],
    lat_min: float,
    lat_max: float,
    apex_height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[dt.datetime, dt.datetime]]]:
    from apexpy import Apex

    n_lat, n_lon = shape
    file_bounds = [file_time_bounds(path) for path in files]
    apex = Apex(date=file_bounds[0][0])
    glat = np.linspace(lat_min, lat_max, n_lat)
    glon = np.linspace(0.0, 360.0, n_lon, endpoint=False)
    glon_grid, glat_grid = np.meshgrid(glon, glat)
    qd_lat, qd_lon = apex.convert(glat_grid, glon_grid, "geo", "qd", height=apex_height)
    lat_bin = np.floor((qd_lat - lat_min) / (lat_max - lat_min) * n_lat).astype(int)
    valid = np.isfinite(qd_lat) & np.isfinite(qd_lon) & (lat_bin >= 0) & (lat_bin < n_lat)
    return qd_lon.ravel(), lat_bin.ravel(), valid.ravel(), np.asarray(file_bounds, dtype=object), file_bounds


def map_frame_to_mlt(
    frame: np.ndarray,
    qd_lon: np.ndarray,
    lat_bin: np.ndarray,
    mapping_valid: np.ndarray,
    dtime: dt.datetime,
) -> np.ndarray:
    """Map a geographic TEC frame into magnetic latitude / MLT bins.

    Bins with no finite source data are left as NaN.  Downstream sparse PCA
    interprets these NaNs as weight 0.  Bins with one or more finite source
    samples receive the average TEC value and therefore enter the PCA with
    binary weight 1, independent of how many source samples contributed.
    """
    from apexpy import Apex

    n_lat, n_lon = frame.shape
    apex = Apex(date=dtime)
    mlt = apex.mlon2mlt(qd_lon, dtime)
    mlt_bin = np.floor((mlt % 24.0) / 24.0 * n_lon).astype(int)
    flat = frame.ravel()
    valid = mapping_valid & np.isfinite(flat) & np.isfinite(mlt) & (mlt_bin >= 0) & (mlt_bin < n_lon)
    cell = lat_bin[valid] * n_lon + mlt_bin[valid]
    sums = np.bincount(cell, weights=flat[valid], minlength=n_lat * n_lon)
    counts = np.bincount(cell, minlength=n_lat * n_lon)
    out = np.full(n_lat * n_lon, np.nan, dtype=frame.dtype)
    observed = counts > 0
    out[observed] = sums[observed] / counts[observed]
    return out.reshape((n_lat, n_lon))


def read_frame_cube(
    files: list[Path],
    indices: np.ndarray,
    shape: tuple[int, int],
    samples_per_file: int,
    coordinate_system: str,
    dtype: str,
    mlt_mapping: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[dt.datetime, dt.datetime]]] | None = None,
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
                if coordinate_system == "local_time":
                    frame = np.roll(frame, local_time_roll(shape[1], samples_per_file, global_time_index), axis=1)
                elif coordinate_system == "mlt":
                    if mlt_mapping is None:
                        raise ValueError("MLT coordinate system needs an MLT mapping")
                    qd_lon, lat_bin, mapping_valid, _, file_bounds = mlt_mapping
                    frame = map_frame_to_mlt(
                        frame,
                        qd_lon,
                        lat_bin,
                        mapping_valid,
                        frame_datetime(file_bounds, global_time_index, samples_per_file),
                    )
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
    coordinate_system: str,
    project_chunk: int,
    ridge: float,
    jobs: int,
    lat_min: float,
    lat_max: float,
    dtype: str,
    mlt_mapping: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[tuple[dt.datetime, dt.datetime]]] | None = None,
    coverage_path: Path | None = None,
) -> None:
    coefficients = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype="float64",
        shape=(component_columns.shape[1], total_times),
    )
    coverage = np.zeros(shape, dtype=np.uint32) if coverage_path is not None else None
    for start in range(0, total_times, project_chunk):
        stop = min(start + project_chunk, total_times)
        print(f"projecting frames {start}:{stop}", flush=True)
        indices = np.arange(start, stop)
        chunk = read_frame_cube(
            files,
            indices,
            shape,
            samples_per_file,
            coordinate_system=coordinate_system,
            dtype=dtype,
            mlt_mapping=mlt_mapping,
        )
        if coverage is not None:
            coverage += np.isfinite(chunk).sum(axis=2, dtype=np.uint32)
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
    if coverage_path is not None and coverage is not None:
        np.save(coverage_path, coverage)


def main() -> None:
    args = parse_args()
    files = hdf5_files(args.input_dir, limit_files=args.limit_files)
    n_lat, n_lon, samples_per_file, total_times = grid_shape(files)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    coordinate_system = "geographic" if args.no_local_time_rotate else args.coordinate_system

    print(f"files={len(files)} grid=({n_lat}, {n_lon}) samples_per_file={samples_per_file} total_times={total_times}")
    print(f"coordinate_system={coordinate_system}")
    mlt_mapping = None
    if coordinate_system == "mlt":
        print("preparing quasi-dipole latitude / MLT mapping", flush=True)
        mlt_mapping = prepare_mlt_mapping(files, (n_lat, n_lon), args.lat_min, args.lat_max, args.apex_height)

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
        coordinate_system=coordinate_system,
        dtype=args.dtype,
        mlt_mapping=mlt_mapping,
    )
    training_coverage = np.isfinite(training_cube).sum(axis=2, dtype=np.uint32)
    np.save(output_dir / f"training_observed_count_global_sparse_{coordinate_system}.npy", training_coverage)
    np.save(
        output_dir / f"training_observed_fraction_global_sparse_{coordinate_system}.npy",
        training_coverage / float(training_cube.shape[2]),
    )
    print(
        "training observed fraction: "
        f"min={np.nanmin(training_coverage / float(training_cube.shape[2])):.4g} "
        f"mean={np.nanmean(training_coverage / float(training_cube.shape[2])):.4g} "
        f"max={np.nanmax(training_coverage / float(training_cube.shape[2])):.4g}",
        flush=True,
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

    np.save(output_dir / f"components_global_sparse_{coordinate_system}.npy", component_images)
    np.save(output_dir / "time_series_global_sparse_training.npy", training_coefficients)
    np.save(output_dir / f"mean_global_sparse_{coordinate_system}.npy", mean.reshape(n_lat, n_lon))
    print(f"saved global components and mean to {output_dir}", flush=True)

    project_all_times(
        files,
        component_columns,
        mean,
        output_dir / f"time_series_global_sparse_{coordinate_system}.npy",
        total_times,
        (n_lat, n_lon),
        samples_per_file,
        coordinate_system=coordinate_system,
        project_chunk=args.project_chunk,
        ridge=args.ridge,
        jobs=args.jobs,
        lat_min=args.lat_min,
        lat_max=args.lat_max,
        dtype=args.dtype,
        mlt_mapping=mlt_mapping,
        coverage_path=output_dir / f"observed_count_global_sparse_{coordinate_system}.npy",
    )
    print("done")


if __name__ == "__main__":
    main()
