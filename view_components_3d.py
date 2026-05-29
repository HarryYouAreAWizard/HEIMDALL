#!/usr/bin/env python3
"""Create an interactive 3D globe viewer for TEC principal components.

The output is a self-contained HTML file. Open it in a browser, drag to rotate
the globe, and use the slider to switch between principal components.
"""

from __future__ import annotations

import argparse
import webbrowser
from pathlib import Path

import numpy as np
import plotly.graph_objects as go


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "components",
        nargs="?",
        default="components/components_geographic.npy",
        help="Component cube with shape (latitude, longitude, component).",
    )
    parser.add_argument(
        "--output",
        default="figures/components_3d_viewer.html",
        help="Output HTML file.",
    )
    parser.add_argument("--lat-min", type=float, default=None, help="Minimum latitude in degrees.")
    parser.add_argument("--lat-max", type=float, default=None, help="Maximum latitude in degrees.")
    parser.add_argument("--lon-min", type=float, default=0.0, help="Minimum longitude/hour angle in degrees.")
    parser.add_argument("--lon-max", type=float, default=360.0, help="Maximum longitude/hour angle in degrees.")
    parser.add_argument(
        "--max-components",
        type=int,
        default=None,
        help="Only include the first N components in the slider.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every Nth grid point to make the HTML lighter.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        help="Open the generated HTML in the default browser.",
    )
    parser.add_argument(
        "--no-context-globe",
        action="store_true",
        help="Do not draw the translucent full globe behind regional component data.",
    )
    return parser.parse_args()


def default_latitude_limits(n_lat: int) -> tuple[float, float]:
    """Infer common TEC grid latitude limits from the number of rows."""
    if n_lat == 30:
        return 60.0, 90.0
    if n_lat == 180:
        return -90.0, 90.0
    return -90.0, 90.0


def globe_coordinates(
    n_lat: int,
    n_lon: int,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lat = np.linspace(lat_min, lat_max, n_lat)
    lon = np.linspace(lon_min, lon_max, n_lon, endpoint=False)
    lon_grid, lat_grid = np.meshgrid(np.deg2rad(lon), np.deg2rad(lat))
    x = np.cos(lat_grid) * np.cos(lon_grid)
    y = np.cos(lat_grid) * np.sin(lon_grid)
    z = np.sin(lat_grid)
    return x, y, z, lat, lon


def add_reference_lines(fig: go.Figure) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 361)
    equator = dict(
        x=np.cos(theta),
        y=np.sin(theta),
        z=np.zeros_like(theta),
        mode="lines",
        line=dict(color="rgba(60,60,60,0.45)", width=3),
        hoverinfo="skip",
        showlegend=False,
    )
    fig.add_trace(go.Scatter3d(**equator))
    fig.add_trace(
        go.Scatter3d(
            x=[0, 0],
            y=[0, 0],
            z=[-1.08, 1.08],
            mode="lines",
            line=dict(color="rgba(60,60,60,0.35)", width=3),
            hoverinfo="skip",
            showlegend=False,
        )
    )
    for label, xyz in [
        ("N", (0, 0, 1.16)),
        ("S", (0, 0, -1.16)),
        ("0 deg", (1.12, 0, 0)),
        ("90 deg", (0, 1.12, 0)),
        ("180 deg", (-1.12, 0, 0)),
        ("270 deg", (0, -1.12, 0)),
    ]:
        fig.add_trace(
            go.Scatter3d(
                x=[xyz[0]],
                y=[xyz[1]],
                z=[xyz[2]],
                mode="text",
                text=[label],
                textfont=dict(size=13, color="black"),
                hoverinfo="skip",
                showlegend=False,
            )
        )


def add_context_globe(fig: go.Figure) -> int:
    lat = np.deg2rad(np.linspace(-90.0, 90.0, 91))
    lon = np.deg2rad(np.linspace(0.0, 360.0, 181))
    lon_grid, lat_grid = np.meshgrid(lon, lat)
    x = np.cos(lat_grid) * np.cos(lon_grid)
    y = np.cos(lat_grid) * np.sin(lon_grid)
    z = np.sin(lat_grid)
    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            surfacecolor=np.zeros_like(x),
            colorscale=[[0, "rgb(232,232,232)"], [1, "rgb(232,232,232)"]],
            showscale=False,
            opacity=0.22,
            hoverinfo="skip",
            name="Earth context",
        )
    )
    return 1


def main() -> None:
    args = parse_args()
    components = np.load(args.components)
    if components.ndim != 3:
        raise ValueError(f"Expected 3D component cube, got shape {components.shape}")

    stride = max(1, int(args.stride))
    components = components[::stride, ::stride, :]
    n_lat, n_lon, n_components_total = components.shape
    n_components = n_components_total if args.max_components is None else min(args.max_components, n_components_total)
    components = components[:, :, :n_components]

    lat_default_min, lat_default_max = default_latitude_limits(np.load(args.components, mmap_mode="r").shape[0])
    lat_min = lat_default_min if args.lat_min is None else args.lat_min
    lat_max = lat_default_max if args.lat_max is None else args.lat_max
    x, y, z, _, _ = globe_coordinates(n_lat, n_lon, lat_min, lat_max, args.lon_min, args.lon_max)

    color_abs = np.nanpercentile(np.abs(components), 99.5)
    if not np.isfinite(color_abs) or color_abs <= 0:
        color_abs = np.nanmax(np.abs(components)) or 1.0

    fig = go.Figure()
    n_context_traces = 0
    if not args.no_context_globe and (lat_min > -90.0 or lat_max < 90.0):
        n_context_traces += add_context_globe(fig)

    for idx in range(n_components):
        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=components[:, :, idx],
                colorscale="RdBu",
                reversescale=True,
                cmin=-color_abs,
                cmax=color_abs,
                colorbar=dict(title="Component value"),
                visible=(idx == 0),
                name=f"PC {idx + 1}",
                hovertemplate=(
                    f"PC {idx + 1}<br>"
                    + "x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<br>"
                    + "value=%{surfacecolor:.4g}<extra></extra>"
                ),
            )
        )

    add_reference_lines(fig)
    n_extra_traces = 8
    steps = []
    for idx in range(n_components):
        visible = [True] * n_context_traces + [False] * n_components + [True] * n_extra_traces
        visible[n_context_traces + idx] = True
        steps.append(
            dict(
                method="update",
                label=str(idx + 1),
                args=[
                    {"visible": visible},
                    {"title": f"TEC principal component {idx + 1}"},
                ],
            )
        )

    fig.update_layout(
        title="TEC principal component 1",
        width=1100,
        height=850,
        scene=dict(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            aspectmode="data",
            camera=dict(eye=dict(x=0.0, y=-1.8, z=1.25), up=dict(x=0, y=0, z=1)),
        ),
        margin=dict(l=0, r=0, t=60, b=60),
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="Component "),
                pad=dict(t=35),
                steps=steps,
            )
        ],
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(output, include_plotlyjs=True, full_html=True)
    print(output)
    if args.open:
        webbrowser.open(output.resolve().as_uri())


if __name__ == "__main__":
    main()
