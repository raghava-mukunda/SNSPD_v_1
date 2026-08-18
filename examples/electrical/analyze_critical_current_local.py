#!/usr/bin/env python3
"""
SNSPD PHASE C
=============
Local critical-current field + visualization.

INPUT
-----
An NPZ file exported by analyze_current_crowding.py containing:

    nodes_m       : FEM node coordinates, shape (N, 2), metres
    triangles     : triangle connectivity, shape (M, 3)
    J_element     : element current-density magnitude, shape (M,), A/m^2
    fem_current_A : transport current represented by FEM, A

OPTIONAL
--------
The script also accepts J_c, wire width and thickness directly.

PHYSICS
-------
For each FEM element e:

    J_e(I) = (I / I_FEM) J_e,FEM

The local current reaches the material critical current density when

    J_e(I_c,e) = J_c

therefore

    I_c,e = I_FEM * J_c / J_e,FEM

The device critical current in this model is

    I_c,device = min_e(I_c,e)

which must equal

    I_FEM * J_c / J_max,FEM.

The visualization shows local I_c,e over the FEM mesh.
Low-Ic regions are the most vulnerable regions of the geometry.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import Normalize


def parse_args():
    p = argparse.ArgumentParser(
        description="SNSPD local critical-current heatmap."
    )

    p.add_argument(
        "npz",
        help="NPZ file exported by current-crowding analysis."
    )

    p.add_argument(
        "--jc",
        type=float,
        required=True,
        help="Material critical current density Jc [A/m^2]."
    )

    p.add_argument(
        "--wire-width-nm",
        type=float,
        default=None,
        help="Wire width [nm], used for straight-wire reference."
    )

    p.add_argument(
        "--thickness-nm",
        type=float,
        default=None,
        help="Film thickness [nm], used for straight-wire reference."
    )

    p.add_argument(
        "--material",
        default="unspecified",
    )

    p.add_argument(
        "--temperature-k",
        type=float,
        default=None,
    )

    p.add_argument(
        "--output",
        default="critical_current_heatmap.png",
    )

    p.add_argument(
        "--output-npz",
        default="critical_current_field.npz",
    )

    p.add_argument(
        "--percentile-low",
        type=float,
        default=1.0,
        help="Lower color percentile."
    )

    p.add_argument(
        "--percentile-high",
        type=float,
        default=99.0,
        help="Upper color percentile."
    )

    return p.parse_args()


def get_array(data, *names):
    for name in names:
        if name in data:
            return np.asarray(data[name])
    raise KeyError(
        "Could not find any of these arrays in NPZ: "
        + ", ".join(names)
    )


def main():
    args = parse_args()

    data = np.load(args.npz)

    nodes = get_array(
        data,
        "nodes_m",
        "nodes",
        "node_coordinates_m",
    ).astype(float)

    triangles = get_array(
        data,
        "triangles",
        "elements",
        "connectivity",
    ).astype(int)

    J = get_array(
        data,
        "J_element",
        "J_magnitude",
        "element_current_density",
        "current_density",
    ).astype(float)

    if "fem_current_A" in data:
        fem_current = float(data["fem_current_A"])
    elif "transport_current_A" in data:
        fem_current = float(data["transport_current_A"])
    else:
        raise KeyError(
            "NPZ must contain fem_current_A or transport_current_A."
        )

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
            "Number of element current-density values does not match "
            "number of FEM triangles."
        )

    if args.jc <= 0:
        raise ValueError("Jc must be positive.")

    if fem_current <= 0:
        raise ValueError("FEM transport current must be positive.")

    finite = (
        np.isfinite(J)
        & (J > 0.0)
    )

    if not np.any(finite):
        raise RuntimeError(
            "No finite positive FEM current-density values found."
        )

    Jmax = float(np.max(J[finite]))

    # ------------------------------------------------------------
    # LOCAL CRITICAL CURRENT
    # ------------------------------------------------------------
    #
    # Ic_local(e) = I_FEM * Jc / J_e
    #
    # Elements with J=0 do not become critical at finite current
    # under this simple local model, so assign +infinity.
    # ------------------------------------------------------------

    Ic_local = np.full_like(
        J,
        np.inf,
        dtype=float,
    )

    Ic_local[finite] = (
        fem_current
        * args.jc
        / J[finite]
    )

    finite_ic = (
        np.isfinite(Ic_local)
        & (Ic_local > 0.0)
    )

    Ic_device = float(
        np.min(Ic_local[finite_ic])
    )

    limiting_element = int(
        np.argmin(
            np.where(
                finite_ic,
                Ic_local,
                np.inf,
            )
        )
    )

    # Direct analytical equivalent from FEM Jmax.
    Ic_from_Jmax = (
        fem_current
        * args.jc
        / Jmax
    )

    relative_error = abs(
        Ic_device - Ic_from_Jmax
    ) / max(
        abs(Ic_from_Jmax),
        1e-30,
    )

    # ------------------------------------------------------------
    # STRAIGHT-WIRE REFERENCE
    # ------------------------------------------------------------

    Ic_straight = None
    crowding_factor = None

    if (
        args.wire_width_nm is not None
        and args.thickness_nm is not None
    ):
        width_m = args.wire_width_nm * 1e-9
        thickness_m = args.thickness_nm * 1e-9

        area_m2 = (
            width_m * thickness_m
        )

        Ic_straight = (
            args.jc * area_m2
        )

        J_transport = (
            fem_current / area_m2
        )

        crowding_factor = (
            Jmax / J_transport
        )

    # ------------------------------------------------------------
    # LIMITING ELEMENT LOCATION
    # ------------------------------------------------------------

    tri = triangles[limiting_element]

    hotspot_xy = np.mean(
        nodes[tri],
        axis=0,
    )

    x_um = hotspot_xy[0] * 1e6
    y_um = hotspot_xy[1] * 1e6

    # ------------------------------------------------------------
    # PRINT REPORT
    # ------------------------------------------------------------

    print()
    print("=" * 52)
    print("SNSPD LOCAL CRITICAL CURRENT ANALYSIS")
    print("=" * 52)

    print()
    print("MATERIAL")
    print("--------")
    print(f"Material                  : {args.material}")

    if args.temperature_k is None:
        print("Temperature               : not specified")
    else:
        print(
            f"Temperature               : "
            f"{args.temperature_k:.6f} K"
        )

    print(
        f"Critical current density : "
        f"{args.jc:.6e} A/m²"
    )

    print()
    print("FEM CURRENT FIELD")
    print("-----------------")
    print(
        f"FEM transport current    : "
        f"{fem_current:.9e} A"
    )
    print(
        f"Maximum |J|              : "
        f"{Jmax:.9e} A/m²"
    )

    if crowding_factor is not None:
        print(
            f"Current crowding C_J     : "
            f"{crowding_factor:.9f}"
        )

    print()
    print("CRITICAL CURRENT")
    print("----------------")
    print(
        f"Ic,device                : "
        f"{Ic_device:.9e} A"
    )
    print(
        f"Ic,device                : "
        f"{Ic_device * 1e6:.6f} µA"
    )

    print(
        f"Ic from Jmax             : "
        f"{Ic_from_Jmax:.9e} A"
    )

    print(
        f"Verification error       : "
        f"{relative_error:.6e}"
    )

    if Ic_straight is not None:
        reduction = (
            100.0
            * (1.0 - Ic_device / Ic_straight)
        )

        print()
        print("STRAIGHT-WIRE COMPARISON")
        print("------------------------")
        print(
            f"Ic,straight              : "
            f"{Ic_straight:.9e} A"
        )
        print(
            f"Ic,straight              : "
            f"{Ic_straight * 1e6:.6f} µA"
        )
        print(
            f"Geometry reduction       : "
            f"{reduction:.6f} %"
        )

    print()
    print("LIMITING LOCATION")
    print("-----------------")
    print(
        f"Element index             : "
        f"{limiting_element}"
    )
    print(
        f"x                         : "
        f"{x_um:.9f} µm"
    )
    print(
        f"y                         : "
        f"{y_um:.9f} µm"
    )
    print(
        f"Local J                   : "
        f"{J[limiting_element]:.9e} A/m²"
    )
    print(
        f"Local Ic                  : "
        f"{Ic_local[limiting_element] * 1e6:.9f} µA"
    )

    # ------------------------------------------------------------
    # SAVE FIELD
    # ------------------------------------------------------------

    np.savez(
        args.output_npz,
        nodes_m=nodes,
        triangles=triangles,
        J_element=J,
        Ic_local_A=Ic_local,
        fem_current_A=fem_current,
        Jc_A_m2=args.jc,
        Ic_device_A=Ic_device,
        Ic_straight_A=(
            np.nan
            if Ic_straight is None
            else Ic_straight
        ),
        limiting_element=limiting_element,
    )

    # ------------------------------------------------------------
    # HEATMAP
    # ------------------------------------------------------------

    triangulation = mtri.Triangulation(
        nodes[:, 0] * 1e6,
        nodes[:, 1] * 1e6,
        triangles,
    )

    values = Ic_local.copy()

    finite_values = values[
        np.isfinite(values)
        & (values > 0.0)
    ]

    # Robust color limits prevent one extreme region from
    # making the whole device visually uninformative.
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

    masked_values = np.ma.masked_invalid(
        np.where(
            np.isfinite(values),
            values,
            np.nan,
        )
    )

    fig, ax = plt.subplots(
        figsize=(11, 9)
    )

    collection = ax.tripcolor(
        triangulation,
        masked_values,
        shading="flat",
        cmap="turbo",
        norm=Normalize(
            vmin=vmin,
            vmax=vmax,
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

    # Mark the limiting FEM element.
    ax.scatter(
        x_um,
        y_um,
        s=90,
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
            f"Color scale = "
            f"{vmin * 1e6:.3f}–{vmax * 1e6:.3f} µA"
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

    fig.savefig(
        args.output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.show()

    print()
    print(
        f"Saved critical-current field : "
        f"{Path(args.output_npz).resolve()}"
    )
    print(
        f"Saved heatmap                 : "
        f"{Path(args.output).resolve()}"
    )

    if relative_error > 1e-10:
        raise RuntimeError(
            "Critical-current verification FAILED: "
            "local field minimum does not agree with "
            "the Jmax-based calculation."
        )

    print()
    print(
        "Local critical-current verification : PASS"
    )


if __name__ == "__main__":
    main()
