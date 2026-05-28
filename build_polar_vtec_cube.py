#!/usr/bin/env python3
"""Build a north-polar magnetic-latitude/MLT VTEC data cube.

Each source VTEC frame is projected to a square polar grid.  The grid covers
the northern magnetic hemisphere down to a configurable latitude.  Pixels with
no data are NaN; pixels hit by multiple source grid cells contain their average.
The output HDF5 cube is compressed with shuffle + deflate.
"""

from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path

import h5py
import numpy as np
from apexpy import Apex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="/data/nonie/tec_data", help="Directory containing TEC HDF5 files.")
    parser.add_argument(
        "--output",
        default="/home/juha/heimdall_polar_vtec_mlt_north50_128.h5",
        help="Output HDF5 file.",
    )
    parser.add_argument("--pixels", type=int, default=128, help="Output image width and height.")
    parser.add_argument("--min-latitude", type=float, default=50.0, help="Magnetic latitude at the disk edge.")
    parser.add_argument("--apex-height", type=float, default=350.0, help="Apex/QD conversion altitude in km.")
    parser.add_argument("--compression-level", type=int, default=4, help="gzip/deflate compression level.")
    parser.add_argument("--chunk-frames", type=int, default=16, help="Time frames per HDF5 chunk.")
    parser.add_argument("--limit-files", type=int, default=None, help="Use only the first N files for testing.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing output file.")
    parser.add_argument("--log-every", type=int, default=288, help="Progress print interval in frames.")
    return parser.parse_args()


def hdf5_files(input_dir: str, limit_files: int | None) -> list[Path]:
    files = sorted(Path(input_dir).glob("*.hdf5"))
    if limit_files is not None:
        files = files[:limit_files]
    if not files:
        raise FileNotFoundError(f"No HDF5 files found in {input_dir}")
    return files


def output_geometry(pixels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.indices((pixels, pixels), dtype=float)
    x_center = (xx + 0.5) / pixels * 2.0 - 1.0
    y_center = 1.0 - (yy + 0.5) / pixels * 2.0
    outside = x_center**2 + y_center**2 > 1.0
    return x_center, y_center, outside


def polar_coordinate_grids(
    pixels: int,
    min_latitude: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return static magnetic-latitude/MLT grids and PCA area weights."""
    x_center, y_center, outside = output_geometry(pixels)
    radius = np.sqrt(x_center**2 + y_center**2)
    colat_max_rad = np.deg2rad(90.0 - min_latitude)
    colat_rad = radius * colat_max_rad
    magnetic_latitude = 90.0 - np.rad2deg(colat_rad)

    # Invert x=-r sin(theta), y=r cos(theta), where theta=(MLT-12)*15 deg.
    theta = np.arctan2(-x_center, y_center)
    mlt = (12.0 + np.rad2deg(theta) / 15.0) % 24.0

    valid_projection_mask = ~outside
    pixel_area = np.zeros((pixels, pixels), dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        # The polar image is uniform in projected disk area r dr dtheta, whereas
        # spherical cap area is sin(colat) dcolat dtheta.  This factor converts
        # projected-pixel sums to approximate equal-area spherical-cap sums.
        area_factor = np.sin(colat_rad) / np.maximum(radius, 1e-12)
    area_factor[radius < 1e-12] = colat_max_rad
    pixel_area[valid_projection_mask] = area_factor[valid_projection_mask].astype(np.float32)

    mean_area = float(np.mean(pixel_area[valid_projection_mask]))
    pca_area_weight_sqrt = np.zeros((pixels, pixels), dtype=np.float32)
    pca_area_weight_sqrt[valid_projection_mask] = np.sqrt(pixel_area[valid_projection_mask] / mean_area)

    magnetic_latitude[outside] = np.nan
    mlt[outside] = np.nan
    return (
        magnetic_latitude.astype(np.float32),
        mlt.astype(np.float32),
        valid_projection_mask,
        pixel_area,
        pca_area_weight_sqrt,
    )


def source_mapping(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    frame_time: dt.datetime,
    pixels: int,
    min_latitude: float,
    apex_height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return fixed source samples and MLT-independent coordinates for a file."""
    apex = Apex(date=frame_time)
    glon_grid, glat_grid = np.meshgrid(longitudes, latitudes)
    qd_latitude, qd_longitude = apex.convert(glat_grid, glon_grid, "geo", "qd", height=apex_height)
    valid = np.isfinite(qd_latitude) & np.isfinite(qd_longitude) & (qd_latitude >= min_latitude)
    radius = (90.0 - qd_latitude[valid]) / (90.0 - min_latitude)
    flat_indices = np.flatnonzero(valid.ravel())
    return flat_indices.astype(np.int64), radius.astype(np.float64), qd_longitude[valid].astype(np.float64)


def project_frame(
    frame: np.ndarray,
    flat_indices: np.ndarray,
    radius: np.ndarray,
    qd_longitude: np.ndarray,
    frame_time: dt.datetime,
    pixels: int,
    outside: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project one source frame to the polar MLT image."""
    apex = Apex(date=frame_time)
    values = frame.ravel()[flat_indices]
    mlt_hours = apex.mlon2mlt(qd_longitude, frame_time)
    valid = np.isfinite(values) & np.isfinite(mlt_hours)

    theta = np.deg2rad((mlt_hours[valid] - 12.0) * 15.0)
    x = -radius[valid] * np.sin(theta)
    y = radius[valid] * np.cos(theta)
    col = np.floor((x + 1.0) * 0.5 * pixels).astype(np.int64)
    row = np.floor((1.0 - (y + 1.0) * 0.5) * pixels).astype(np.int64)
    inside = (row >= 0) & (row < pixels) & (col >= 0) & (col < pixels)

    sums = np.zeros((pixels, pixels), dtype=np.float64)
    counts = np.zeros((pixels, pixels), dtype=np.uint16)
    np.add.at(sums, (row[inside], col[inside]), values[valid][inside])
    np.add.at(counts, (row[inside], col[inside]), 1)

    projected = np.full((pixels, pixels), np.nan, dtype=np.float32)
    observed = counts > 0
    projected[observed] = (sums[observed] / counts[observed]).astype(np.float32)
    projected[outside] = np.nan
    counts[outside] = 0
    return projected, counts


def total_frames(files: list[Path]) -> tuple[int, int, int]:
    with h5py.File(files[0], "r") as h5:
        n_lat, n_lon, samples_per_file = h5["Data/Array Layout/2D Parameters/tec"].shape
    return n_lat, n_lon, samples_per_file * len(files)


def main() -> None:
    args = parse_args()
    files = hdf5_files(args.input_dir, args.limit_files)
    _, _, n_total = total_frames(files)
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"{output} exists; use --overwrite to replace it")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    magnetic_latitude, mlt, valid_projection_mask, pixel_area, pca_area_weight_sqrt = polar_coordinate_grids(
        args.pixels,
        args.min_latitude,
    )
    outside = ~valid_projection_mask
    start_time = time.time()
    frame_cursor = 0

    with h5py.File(output, "w") as out:
        vtec = out.create_dataset(
            "vtec",
            shape=(n_total, args.pixels, args.pixels),
            dtype="f4",
            chunks=(max(1, args.chunk_frames), args.pixels, args.pixels),
            compression="gzip",
            compression_opts=args.compression_level,
            shuffle=True,
            fillvalue=np.nan,
        )
        counts = out.create_dataset(
            "count",
            shape=(n_total, args.pixels, args.pixels),
            dtype="u2",
            chunks=(max(1, args.chunk_frames), args.pixels, args.pixels),
            compression="gzip",
            compression_opts=args.compression_level,
            shuffle=True,
            fillvalue=0,
        )
        timestamps = out.create_dataset("timestamps", shape=(n_total,), dtype="f8")
        source_file_index = out.create_dataset("source_file_index", shape=(n_total,), dtype="u4")
        source_frame_index = out.create_dataset("source_frame_index", shape=(n_total,), dtype="u2")
        out.create_dataset("magnetic_latitude", data=magnetic_latitude, compression="gzip", shuffle=True)
        out.create_dataset("mlt", data=mlt, compression="gzip", shuffle=True)
        out.create_dataset("valid_projection_mask", data=valid_projection_mask, compression="gzip", shuffle=True)
        out.create_dataset("pixel_area_weight", data=pixel_area, compression="gzip", shuffle=True)
        out.create_dataset("pca_area_weight_sqrt", data=pca_area_weight_sqrt, compression="gzip", shuffle=True)

        out.attrs["projection"] = "north_polar_qd_mlat_mlt"
        out.attrs["pixels"] = args.pixels
        out.attrs["min_magnetic_latitude"] = args.min_latitude
        out.attrs["apex_height_km"] = args.apex_height
        out.attrs["orientation"] = "12 MLT/noon up, 18 MLT/dusk left, 06 MLT/dawn right, 00 MLT/midnight down"
        out.attrs["empty_pixel_value"] = "NaN"
        out.attrs["collision_handling"] = "average source VTEC samples per output pixel"
        out.attrs["compression"] = f"shuffle + gzip level {args.compression_level}"
        out.attrs["pca_weighting"] = (
            "For sparse PCA, use observed mask from finite vtec values and multiply "
            "inside-disk pixels by pca_area_weight_sqrt. Outside valid_projection_mask "
            "pixels should be excluded."
        )
        out.attrs["pixel_area_weight"] = (
            "Approximate spherical-cap area factor for the polar projection, normalized "
            "only through pca_area_weight_sqrt."
        )

        for file_idx, path in enumerate(files):
            with h5py.File(path, "r") as h5:
                tec = h5["Data/Array Layout/2D Parameters/tec"]
                latitudes = h5["Data/Array Layout/gdlat"][:]
                longitudes = h5["Data/Array Layout/glon"][:]
                file_timestamps = h5["Data/Array Layout/timestamps"][:]
                first_time = dt.datetime.fromtimestamp(float(file_timestamps[0]), tz=dt.timezone.utc).replace(tzinfo=None)
                flat_indices, radius, qd_longitude = source_mapping(
                    latitudes,
                    longitudes,
                    first_time,
                    args.pixels,
                    args.min_latitude,
                    args.apex_height,
                )
                for local_idx, timestamp in enumerate(file_timestamps):
                    frame_time = dt.datetime.fromtimestamp(float(timestamp), tz=dt.timezone.utc).replace(tzinfo=None)
                    projected, pixel_counts = project_frame(
                        tec[:, :, local_idx],
                        flat_indices,
                        radius,
                        qd_longitude,
                        frame_time,
                        args.pixels,
                        outside,
                    )
                    vtec[frame_cursor, :, :] = projected
                    counts[frame_cursor, :, :] = pixel_counts
                    timestamps[frame_cursor] = timestamp
                    source_file_index[frame_cursor] = file_idx
                    source_frame_index[frame_cursor] = local_idx
                    frame_cursor += 1
                    if frame_cursor % args.log_every == 0:
                        elapsed = time.time() - start_time
                        rate = frame_cursor / elapsed if elapsed > 0 else float("nan")
                        out.attrs["frames_written"] = frame_cursor
                        out.flush()
                        print(
                            f"frames {frame_cursor}/{n_total} "
                            f"({100.0 * frame_cursor / n_total:.2f}%), "
                            f"{rate:.2f} frames/s",
                            flush=True,
                        )
        out.attrs["frames_written"] = frame_cursor

    size = output.stat().st_size
    print(f"wrote {output}")
    print(f"file size bytes {size}")
    print(f"file size GiB {size / 1024**3:.3f}")


if __name__ == "__main__":
    main()
