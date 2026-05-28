#!/usr/bin/env python3
"""Plot sparse TEC principal vectors in a northern hour-angle view."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--components",
        default="sparse_results/components_sparse.npy",
        help="Sparse component array with shape (latitude, longitude/hour, component).",
    )
    parser.add_argument(
        "--output-prefix",
        default="sparse_results/sparse_tec_first8_principal_vectors_hour_angle",
        help="Output path prefix. .png and .pdf are written.",
    )
    parser.add_argument("--n-components", type=int, default=8, help="Number of components to plot.")
    parser.add_argument("--outer-latitude", type=float, default=60.0, help="Latitude of the outer circle.")
    return parser.parse_args()


def hour_angle_edges(n_hours: int) -> np.ndarray:
    """Return angular cell edges with 12 h at +y and 18 h at -x."""
    hour_edges = np.linspace(0.0, 24.0, n_hours + 1)
    return np.deg2rad((hour_edges - 12.0) * 15.0)


def northern_disk_edges(n_lat: int, outer_latitude: float) -> tuple[np.ndarray, np.ndarray]:
    """Return Cartesian cell-edge coordinates for a northern polar cap."""
    r_edges = np.linspace(0.0, 90.0 - outer_latitude, n_lat + 1)
    theta_edges = hour_angle_edges(360)
    theta_grid, r_grid = np.meshgrid(theta_edges, r_edges)
    x = -r_grid * np.sin(theta_grid)
    y = r_grid * np.cos(theta_grid)
    return x, y


def add_hour_angle_grid(ax: plt.Axes, outer_latitude: float) -> None:
    radius = 90.0 - outer_latitude
    for latitude in [80.0, 70.0, 60.0]:
        r = 90.0 - latitude
        circle = plt.Circle((0, 0), r, fill=False, color="0.45", lw=0.7, alpha=0.6)
        ax.add_patch(circle)
        ax.text(0.0, r, f"{latitude:.0f}", ha="center", va="bottom", fontsize=8, color="0.25")

    ax.plot([0, 0], [-radius, radius], color="0.45", lw=0.7, alpha=0.6)
    ax.plot([-radius, radius], [0, 0], color="0.45", lw=0.7, alpha=0.6)
    outer = plt.Circle((0, 0), radius, fill=False, color="0.1", lw=1.0)
    ax.add_patch(outer)

    label_offset = radius * 1.12
    ax.text(0, label_offset, "12 noon", ha="center", va="bottom", fontsize=9)
    ax.text(-label_offset, 0, "18", ha="right", va="center", fontsize=9)
    ax.text(0, -label_offset, "00", ha="center", va="top", fontsize=9)
    ax.text(label_offset, 0, "06", ha="left", va="center", fontsize=9)
    ax.text(0, 0, "90", ha="center", va="center", fontsize=8, color="0.25")

    ax.set_xlim(-radius * 1.2, radius * 1.2)
    ax.set_ylim(-radius * 1.2, radius * 1.2)
    ax.set_aspect("equal")
    ax.axis("off")


def main() -> None:
    args = parse_args()
    components = np.load(args.components)
    n_lat, n_hour, n_component_total = components.shape
    if n_hour != 360:
        raise ValueError(f"Expected 360 longitude/hour bins, got {n_hour}")

    n_components = min(args.n_components, n_component_total)
    x_edges, y_edges = northern_disk_edges(n_lat, args.outer_latitude)

    fig, axes = plt.subplots(2, 4, figsize=(13.5, 7.6), constrained_layout=True)
    for idx, ax in enumerate(axes.flat):
        if idx >= n_components:
            ax.axis("off")
            continue
        data = components[:, :, idx][::-1, :]
        vmax = np.nanpercentile(np.abs(data), 99.5)
        if not np.isfinite(vmax) or vmax <= 0:
            vmax = np.nanmax(np.abs(data)) or 1.0
        mesh = ax.pcolormesh(x_edges, y_edges, data, shading="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        add_hour_angle_grid(ax, args.outer_latitude)
        ax.set_title(f"Sparse PC {idx + 1}", pad=4)
        cbar = fig.colorbar(mesh, ax=ax, shrink=0.78, pad=0.02)
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle("First 8 sparse TEC principal vectors, northern hour-angle view", fontsize=15)
    output_prefix = Path(args.output_prefix)
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf"]:
        output = output_prefix.with_suffix(f".{suffix}")
        fig.savefig(output, dpi=220 if suffix == "png" else None)
        print(output)


if __name__ == "__main__":
    main()
