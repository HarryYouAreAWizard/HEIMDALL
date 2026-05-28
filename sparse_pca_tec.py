#!/usr/bin/env python3
"""Estimate TEC principal components from sparse measurements.

Missing or non-finite TEC pixels are assigned zero weight in the PCA objective.
Observed pixels are latitude-weighted in the same way as the original PCA code.
"""

import argparse
import os

import numpy as np

from principal_component_analysis import (
    check_orthonomality,
    compute_sparse_time_coefficients,
    find_sparse_principal_components,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default="/data/nonie/masterdata/raw_northern_tec.npy",
        help="Input TEC cube, shape (lat, lon, time). Missing data should be NaN.",
    )
    parser.add_argument(
        "--components-out",
        default=os.path.expanduser("~/heimdall_sparse_results/components_sparse.npy"),
        help="Output .npy file for sparse PCA component images.",
    )
    parser.add_argument(
        "--coefficients-out",
        default=os.path.expanduser("~/heimdall_sparse_results/time_series_sparse.npy"),
        help="Output .npy file for sparse PCA time coefficients.",
    )
    parser.add_argument(
        "--mean-out",
        default=os.path.expanduser("~/heimdall_sparse_results/mean_sparse.npy"),
        help="Output .npy file for the sparse observed-sample mean image.",
    )
    parser.add_argument("--components", type=int, default=9, help="Number of principal components.")
    parser.add_argument("--iterations", type=int, default=8, help="Alternating least-squares iterations.")
    parser.add_argument("--jobs", type=int, default=1, help="Worker processes for sparse ALS solves.")
    parser.add_argument("--ridge", type=float, default=1e-6, help="Small ridge term for weighted least squares.")
    parser.add_argument(
        "--sample-columns",
        type=int,
        default=2048,
        help="Number of time columns sampled for the initial zero-filled SVD.",
    )
    parser.add_argument("--random-state", type=int, default=0, help="Random seed for initialization sampling.")
    parser.add_argument(
        "--no-orthogonality-check",
        action="store_true",
        help="Skip the component dot-product diagnostic.",
    )
    return parser.parse_args()


def ensure_parent(path: str) -> None:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)


def main() -> None:
    args = parse_args()
    print(f"Loading sparse TEC cube: {args.input}")
    tec = np.load(args.input)
    print(f"TEC shape: {tec.shape}")
    print(f"Observed samples: {np.isfinite(tec).sum()} / {tec.size}")

    components, component_columns, coefficients, mean = find_sparse_principal_components(
        tec,
        args.components,
        n_iterations=args.iterations,
        ridge=args.ridge,
        sample_columns=args.sample_columns,
        random_state=args.random_state,
        n_jobs=args.jobs,
        verbose=True,
    )

    # Recompute coefficients from final components. This is redundant after ALS,
    # but keeps this script explicit and makes it easy to project a different
    # sparse TEC cube later with compute_sparse_time_coefficients().
    coefficients = compute_sparse_time_coefficients(
        component_columns,
        tec,
        mean=mean,
        ridge=args.ridge,
        n_jobs=args.jobs,
    )

    for path, array in [
        (args.components_out, components),
        (args.coefficients_out, coefficients),
        (args.mean_out, mean.reshape(tec.shape[0], tec.shape[1])),
    ]:
        ensure_parent(path)
        np.save(path, array)
        print(f"Saved {path}: {array.shape}")

    if not args.no_orthogonality_check:
        check_orthonomality(components)


if __name__ == "__main__":
    main()
