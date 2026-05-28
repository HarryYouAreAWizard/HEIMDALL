#!/usr/bin/env python3
"""Plot polar MLT sparse PCA components as paged 4x4 grids."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", default="results/polar_mlt_sparse_pca_32_allframes")
    parser.add_argument("--output-prefix", default="figures/polar_mlt_sparse_pca_32_allframes")
    parser.add_argument("--components", default=None, help="Override component .npy path.")
    parser.add_argument("--n-components", type=int, default=32)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--color-percentile", type=float, default=98.0)
    return parser.parse_args()


def add_grid(ax: plt.Axes, magnetic_latitude: np.ndarray, valid: np.ndarray) -> None:
    min_latitude = float(np.nanmin(magnetic_latitude[valid]))
    radius_scale = 1.0 / (90.0 - min_latitude)
    for latitude in [80.0, 70.0, 60.0, 50.0]:
        if latitude < min_latitude:
            continue
        radius = (90.0 - latitude) * radius_scale
        circle = plt.Circle((0, 0), radius, fill=False, color="0.45", lw=0.8, alpha=0.72)
        ax.add_patch(circle)
        ax.text(0.0, radius, f"{latitude:.0f}", ha="center", va="bottom", fontsize=8, color="0.25")
    for mlt in range(0, 24, 3):
        theta = np.deg2rad((mlt - 12.0) * 15.0)
        ax.plot([0, -np.sin(theta)], [0, np.cos(theta)], color="0.55", lw=0.6, alpha=0.55)
    label_radius = 1.1
    for mlt, label in [(12, "12 noon"), (18, "18 dusk"), (0, "00 midnight"), (6, "06 dawn")]:
        theta = np.deg2rad((mlt - 12.0) * 15.0)
        ax.text(-label_radius * np.sin(theta), label_radius * np.cos(theta), label, ha="center", va="center", fontsize=9)
    ax.set_xlim(-1.18, 1.18)
    ax.set_ylim(-1.18, 1.18)
    ax.set_aspect("equal")
    ax.axis("off")


def plot_page(
    components: np.ndarray,
    magnetic_latitude: np.ndarray,
    valid: np.ndarray,
    indices: np.ndarray,
    output_prefix: Path,
    page_number: int,
    page_count: int,
    rows: int,
    cols: int,
    color_percentile: float,
) -> None:
    fig, axes = plt.subplots(rows, cols, figsize=(3.55 * cols, 3.35 * rows), constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()
    masked = np.ma.masked_invalid(components)
    for panel_idx, ax in enumerate(axes):
        if panel_idx >= len(indices):
            ax.axis("off")
            continue
        component_idx = int(indices[panel_idx])
        data = masked[:, :, component_idx]
        vals = np.asarray(data.compressed())
        vmax = np.nanpercentile(np.abs(vals), color_percentile) if vals.size else 1.0
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = np.nanmax(np.abs(vals)) if vals.size else 1.0
        ax.imshow(
            data,
            origin="upper",
            extent=[-1, 1, -1, 1],
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        add_grid(ax, magnetic_latitude, valid)
        ax.set_title(f"PC {component_idx + 1}", pad=3)
        ax.text(
            0.98,
            0.04,
            f"+/-{vmax:.2g}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.72),
        )
    fig.suptitle(f"Polar MLT sparse TEC PCA components (page {page_number}/{page_count})", fontsize=15)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf"]:
        output = output_prefix.parent / f"{output_prefix.name}_page{page_number:02d}.{suffix}"
        fig.savefig(output, dpi=220 if suffix == "png" else None)
        print(output)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    component_path = Path(args.components) if args.components else results_dir / "components_polar_mlt.npy"
    components = np.load(component_path)
    magnetic_latitude = np.load(results_dir / "magnetic_latitude.npy")
    valid = np.load(results_dir / "valid_projection_mask.npy")
    components = components.copy()
    components[~valid, :] = np.nan
    n_components = min(args.n_components, components.shape[2])
    indices = np.arange(n_components)
    panel_count = args.rows * args.cols
    page_count = int(np.ceil(n_components / panel_count))
    output_prefix = Path(args.output_prefix)
    for page_idx in range(page_count):
        plot_page(
            components,
            magnetic_latitude,
            valid,
            indices[page_idx * panel_count : (page_idx + 1) * panel_count],
            output_prefix,
            page_idx + 1,
            page_count,
            args.rows,
            args.cols,
            args.color_percentile,
        )


if __name__ == "__main__":
    main()
