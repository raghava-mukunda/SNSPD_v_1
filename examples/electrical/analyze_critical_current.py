#!/usr/bin/env python3
"""
FILE: examples/electrical/analyze_critical_current.py

SNSPD PHASE A + B + C
=====================

Unified critical-current analysis.

This is the ONLY critical-current example file required.

It consumes the FEM result exported by:

    examples/electrical/analyze_current_crowding.py

Expected FEM result:

    results/current_crowding_fem.npz

The script calculates:

    Phase A
    -------
    Straight-wire critical current

        Ic,straight = Jc * w * t

    Phase B
    -------
    Geometry-limited critical current from FEM current crowding

        Ic,geometry = I_FEM * Jc / J_max

    Phase C
    -------
    Element-by-element local critical-current field

        Ic_local,e = I_FEM * Jc / J_e

    and therefore

        Ic,device = min_e(Ic_local,e)

The script also creates a heatmap of Ic_local over the actual FEM mesh.

IMPORTANT
---------
Jc is intentionally an explicit input.

This script does NOT silently invent a material critical-current
density.

Temperature is currently metadata only unless the supplied Jc
already represents the chosen operating temperature.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import Normalize


def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "SNSPD Phase A+B+C critical-current analysis "
            "from FEM current-density results."
        )
    )

    parser.add_argument(
        "npz",
        nargs="?",
        default="results/current_crowding_fem.npz",
        help=(
            "FEM .npz exported by analyze_current_crowding.py. "
            "Default: results/current_crowding_fem.npz"
        ),
    )

    parser.add_argument(
        "--wire-width-nm",
        type=float,
        default=None,
        help=(
            "Nanowire width in nm. If omitted, the value stored "
            "in the FEM NPZ is used."
        ),
    )

    parser.add_argument(
        "--thickness-nm",
        type=float,
        default=None,
        help=(
            "Superconducting film thickness in nm. If omitted, "
            "the value stored in the FEM NPZ is used."
        ),
    )

    parser.add_argument(
        "--jc",
        type=float,
        required=True,
        help="Critical current density Jc in A/m^2.",
    )

    parser.add_argument(
        "--material",
        type=str,
        default="unspecified",
        help="Material name.",
    )

    parser.add_argument(
        "--temperature-k",
        type=float,
        default=None,
        help="Operating temperature in K.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="critical_current_heatmap.png",
        help="Output critical-current heatmap.",
    )

    parser.add_argument(
        "--output-npz",
        type=str,
        default="critical_current_field.npz",
        help="Output local critical-current field.",
    )

    parser.add_argument(
        "--percentile-low",
        type=float,
        default=1.0,
        help="Lower color percentile for heatmap scaling.",
    )

    parser.add_argument(
        "--percentile-high",
        type=float,
        default=99.0,
        help="Upper color percentile for heatmap scaling.",
    )

    return parser.parse_args()


def load_first(data, names, required=True):
    """Return the first available NPZ field from names."""
    for name in names:
        if name in data:
            return np.asarray(data[name])

    if required:
        raise KeyError(
            "Required FEM field not found. Tried: "
            + ", ".join(names)
        )

    return None


def scalar_from_npz(data, names, default=None):
    value = load_first(data, names, required=False)
    if value is None:
        return default
    return float(np.asarray(value).reshape(-1)[0])


def main() -> None:

    args = parse_arguments()

    npz_path = Path(args.npz).expanduser().resolve()

    if not npz_path.exists():
        raise FileNotFoundError(
            f"FEM result file not found:\n{npz_path}\n\n"
            "Run analyze_current_crowding.py first."
        )

    data = np.load(npz_path)

    # ------------------------------------------------------------
    # LOAD FEM MESH
    # ------------------------------------------------------------

    nodes = load_first(
        data,
        [
            "nodes_m",
            "nodes",
            "node_coordinates_m",
        ],
    ).astype(float)

    triangles = load_first(
        data,
        [
            "triangles",
            "elements",
            "connectivity",
        ],
    ).astype(int)

    # Current-density field.
    # Support both the names used by the FEM exporter and the
    # names used by the earlier Phase-C prototype.
    J = load_first(
        data,
        [
            "element_J_magnitude_A_per_m2",
            "J_element",
            "J_magnitude",
            "element_current_density",
            "current_density",
        ],
    ).astype(float)

    fem_current = scalar_from_npz(
        data,
        [
            "fem_current_A",
            "transport_current_A",
        ],
    )

    if fem_current is None:
        raise KeyError(
            "FEM NPZ does not contain fem_current_A."
        )

    # ------------------------------------------------------------
    # LOAD GEOMETRIC PARAMETERS
    # ------------------------------------------------------------

    width_m = scalar_from_npz(
        data,
        [
            "wire_width_m",
        ],
    )

    thickness_m = scalar_from_npz(
        data,
        [
            "film_thickness_m",
        ],
    )

    if args.wire_width_nm is not None:
        width_m = args.wire_width_nm * 1e-9

    if args.thickness_nm is not None:
        thickness_m = args.thickness_nm * 1e-9

    if width_m is None:
        raise RuntimeError(
            "Wire width unavailable. Supply --wire-width-nm."
        )

    if thickness_m is None:
        raise RuntimeError(
            "Film thickness unavailable. Supply --thickness-nm."
        )

    if width_m <= 0 or thickness_m <= 0:
        raise ValueError(
            "Wire width and film thickness must be positive."
        )

    if args.jc <= 0:
        raise ValueError(
            "Critical current density Jc must be positive."
        )

    if fem_current <= 0:
        raise ValueError(
            "FEM transport current must be positive."
        )

    # ------------------------------------------------------------
    # VALIDATE FEM ARRAYS
    # ------------------------------------------------------------

    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise RuntimeError(
            f"nodes must have shape (N,2), got {nodes.shape}"
        )

    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise RuntimeError(
            f"triangles must have shape (M,3), got {triangles.shape}"
        )

    if len(J) != len(triangles):
        raise RuntimeError(
            "Element current-density count does not match "
            "the number of FEM triangles."
        )

    finite_J = (
        np.isfinite(J)
        & (J > 0.0)
    )

    if not np.any(finite_J):
        raise RuntimeError(
            "No finite positive FEM current-density values found."
        )

    Jmax = float(np.max(J[finite_J]))

    # ------------------------------------------------------------
    # PHASE A
    # ------------------------------------------------------------

    cross_section_m2 = width_m * thickness_m

    Ic_straight = (
        args.jc * cross_section_m2
    )

    # ------------------------------------------------------------
    # PHASE B
    # ------------------------------------------------------------

    J_transport = (
        fem_current / cross_section_m2
    )

    crowding_factor = (
        Jmax / J_transport
    )

    Ic_geometry = (
        fem_current
        * args.jc
        / Jmax
    )

    geometry_reduction_percent = (
        100.0
        * (
            1.0
            - Ic_geometry / Ic_straight
        )
    )

    # ------------------------------------------------------------
    # PHASE C
    # ------------------------------------------------------------

    Ic_local = np.full_like(
        J,
        np.inf,
        dtype=float,
    )

    Ic_local[finite_J] = (
        fem_current
        * args.jc
        / J[finite_J]
    )

    finite_Ic = (
        np.isfinite(Ic_local)
        & (Ic_local > 0.0)
    )

    if not np.any(finite_Ic):
        raise RuntimeError(
            "No finite local critical-current values found."
        )

    limiting_element = int(
        np.argmin(
            np.where(
                finite_Ic,
                Ic_local,
                np.inf,
            )
        )
    )

    Ic_device = float(
        Ic_local[limiting_element]
    )

    # The local minimum must mathematically equal the Jmax-based
    # geometry-limited value.
    consistency_error = (
        abs(Ic_device - Ic_geometry)
        / max(abs(Ic_geometry), 1e-30)
    )

    # ------------------------------------------------------------
    # LIMITING ELEMENT LOCATION
    # ------------------------------------------------------------

    limiting_triangle = triangles[
        limiting_element
    ]

    limiting_xy_m = np.mean(
        nodes[limiting_triangle],
        axis=0,
    )

    x_um = limiting_xy_m[0] * 1e6
    y_um = limiting_xy_m[1] * 1e6

    # ------------------------------------------------------------
    # REPORT
    # ------------------------------------------------------------

    print()
    print("=" * 56)
    print("SNSPD CRITICAL CURRENT ANALYSIS")
    print("=" * 56)

    print()
    print("INPUT")
    print("-----")
    print(f"FEM result file           : {npz_path}")
    print(f"Material                  : {args.material}")

    if args.temperature_k is None:
        print("Temperature               : not specified")
    else:
        print(
            f"Temperature               : "
            f"{args.temperature_k:.6f} K"
        )

    print(
        f"Wire width                : "
        f"{width_m * 1e9:.6f} nm"
    )

    print(
        f"Film thickness            : "
        f"{thickness_m * 1e9:.6f} nm"
    )

    print(
        f"Cross-sectional area      : "
        f"{cross_section_m2:.6e} m²"
    )

    print(
        f"Critical current density  : "
        f"{args.jc:.6e} A/m²"
    )

    print()
    print("PHASE A — STRAIGHT WIRE")
    print("-----------------------")
    print(
        f"Ic,straight               : "
        f"{Ic_straight:.9e} A"
    )
    print(
        f"Ic,straight               : "
        f"{Ic_straight * 1e6:.6f} µA"
    )

    print()
    print("FEM CURRENT DISTRIBUTION")
    print("------------------------")
    print(
        f"FEM transport current     : "
        f"{fem_current:.9e} A"
    )
    print(
        f"Transport current density : "
        f"{J_transport:.9e} A/m²"
    )
    print(
        f"Maximum |J|               : "
        f"{Jmax:.9e} A/m²"
    )
    print(
        f"Current crowding factor   : "
        f"{crowding_factor:.9f}"
    )

    print()
    print("PHASE B — GEOMETRY-LIMITED CRITICAL CURRENT")
    print("--------------------------------------------")
    print(
        f"Ic,geometry               : "
        f"{Ic_geometry:.9e} A"
    )
    print(
        f"Ic,geometry               : "
        f"{Ic_geometry * 1e6:.6f} µA"
    )
    print(
        f"Ic,geometry / Ic,straight : "
        f"{Ic_geometry / Ic_straight:.9f}"
    )
    print(
        f"Critical-current reduction: "
        f"{geometry_reduction_percent:.6f} %"
    )

    print()
    print("PHASE C — LOCAL CRITICAL-CURRENT FIELD")
    print("--------------------------------------")
    print(
        f"Device Ic from local field : "
        f"{Ic_device:.9e} A"
    )
    print(
        f"Device Ic from local field : "
        f"{Ic_device * 1e6:.6f} µA"
    )
    print(
        f"Limiting FEM element       : "
        f"{limiting_element}"
    )
    print(
        f"Limiting x                 : "
        f"{x_um:.9f} µm"
    )
    print(
        f"Limiting y                 : "
        f"{y_um:.9f} µm"
    )
    print(
        f"Limiting local J           : "
        f"{J[limiting_element]:.9e} A/m²"
    )
    print(
        f"Local Ic at limiter        : "
        f"{Ic_local[limiting_element] * 1e6:.9f} µA"
    )

    print()
    print("CONSISTENCY CHECK")
    print("-----------------")
    print(
        f"Phase-B Ic                 : "
        f"{Ic_geometry:.9e} A"
    )
    print(
        f"Phase-C minimum Ic         : "
        f"{Ic_device:.9e} A"
    )
    print(
        f"Relative difference        : "
        f"{consistency_error:.6e}"
    )

    # ------------------------------------------------------------
    # SAVE LOCAL FIELD
    # ------------------------------------------------------------

    output_npz = (
        Path(args.output_npz)
        .expanduser()
        .resolve()
    )

    output_npz.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_npz,
        nodes_m=nodes,
        triangles=triangles,
        J_element=J,
        Ic_local_A=Ic_local,
        fem_current_A=fem_current,
        Jc_A_m2=args.jc,
        Ic_straight_A=Ic_straight,
        Ic_geometry_A=Ic_geometry,
        Ic_device_A=Ic_device,
        crowding_factor=crowding_factor,
        limiting_element=limiting_element,
        limiting_x_m=limiting_xy_m[0],
        limiting_y_m=limiting_xy_m[1],
        wire_width_m=width_m,
        film_thickness_m=thickness_m,
    )

    # ------------------------------------------------------------
    # CRITICAL-CURRENT HEATMAP
    # ------------------------------------------------------------

    triangulation = mtri.Triangulation(
        nodes[:, 0] * 1e6,
        nodes[:, 1] * 1e6,
        triangles,
    )

    finite_values = Ic_local[
        np.isfinite(Ic_local)
        & (Ic_local > 0.0)
    ]

    vmin = float(
        np.percentile(
            finite_values,
            args.percentile_low,
        )
    )

    vmax = float(
        np.percentile(
            finite_values,
            args.percentile_high,
        )
    )

    if vmax <= vmin:
        vmin = float(np.min(finite_values))
        vmax = float(np.max(finite_values))

    if vmax <= vmin:
        vmax = vmin * 1.01

    plot_values = np.ma.masked_where(
        ~np.isfinite(Ic_local),
        Ic_local * 1e6,
    )

    fig, ax = plt.subplots(
        figsize=(11, 9)
    )

    collection = ax.tripcolor(
        triangulation,
        plot_values,
        shading="flat",
        cmap="turbo",
        norm=Normalize(
            vmin=vmin * 1e6,
            vmax=vmax * 1e6,
        ),
    )

    cbar = fig.colorbar(
        collection,
        ax=ax,
        pad=0.02,
    )

    cbar.set_label(
        "Local critical current $I_{c,local}$ (µA)"
    )

    ax.scatter(
        x_um,
        y_um,
        s=100,
        facecolors="none",
        edgecolors="white",
        linewidths=2.0,
        zorder=10,
    )

    ax.annotate(
        (
            f"Limiting element\n"
            f"x = {x_um:.3f} µm\n"
            f"y = {y_um:.3f} µm\n"
            f"Ic = {Ic_device * 1e6:.3f} µA"
        ),
        xy=(x_um, y_um),
        xytext=(12, 12),
        textcoords="offset points",
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="black",
            alpha=0.75,
        ),
        color="white",
        fontsize=9,
        zorder=11,
    )

    title = (
        "SNSPD Local Critical-Current Map\n"
        f"{args.material}, "
        f"$J_c={args.jc:.3e}$ A/m²"
    )

    if args.temperature_k is not None:
        title += (
            f", T={args.temperature_k:.2f} K"
        )

    ax.set_title(title)
    ax.set_xlabel("x (µm)")
    ax.set_ylabel("y (µm)")
    ax.set_aspect("equal")

    ax.text(
        0.01,
        0.01,
        (
            f"Device Ic = {Ic_device * 1e6:.3f} µA\n"
            f"Jmax = {Jmax:.3e} A/m²\n"
            f"CJ = {crowding_factor:.3f}"
        ),
        transform=ax.transAxes,
        va="bottom",
        ha="left",
        fontsize=9,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="white",
            alpha=0.85,
        ),
    )

    fig.tight_layout()

    output_png = (
        Path(args.output)
        .expanduser()
        .resolve()
    )

    output_png.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_png,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    print()
    print(
        f"Critical-current field saved : "
        f"{output_npz}"
    )
    print(
        f"Critical-current heatmap saved: "
        f"{output_png}"
    )

    if consistency_error > 1e-10:
        raise RuntimeError(
            "Critical-current consistency FAILED."
        )

    print()
    print(
        "Critical-current consistency : PASS"
    )


if __name__ == "__main__":
    main()
