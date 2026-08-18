#!/usr/bin/env python3
"""
FILE: examples/electrical/analyze_room_temperature_resistance.py

PURPOSE
-------
Calculate the normal-state / room-temperature resistance of the
SNSPD meander from the already-solved stationary electrical FEM.

The current-crowding FEM solves:

    div(sigma * grad(V)) = 0

with a 1 V terminal difference and the specified normal-state
resistivity.

The FEM reaction current is therefore used directly:

    R_RT = V / I_FEM

This is preferable to estimating resistance from the bounding-box
dimensions because the FEM current is obtained from the actual
processed meander geometry and terminal definition.

IMPORTANT
---------
This script does NOT solve a second FEM problem.

It consumes:

    results/current_crowding_fem.npz

which was produced by analyze_current_crowding.py.

Therefore the reported resistance corresponds to the SAME normal-state
resistivity, geometry, film thickness, nanowire width, and terminal
definition used by the validated electrical FEM.

The resistance should be interpreted as:

    "normal-state resistance using the supplied room-temperature
     resistivity model"

rather than a universal experimental NbTiN room-temperature value.

Example
-------
    python3 examples/electrical/analyze_room_temperature_resistance.py \
        results/current_crowding_fem.npz

Override the voltage used for R = V/I:

    --voltage 1.0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate room-temperature/normal-state resistance "
            "from the validated electrical FEM result."
        )
    )

    parser.add_argument(
        "fem_npz",
        help="FEM result produced by analyze_current_crowding.py.",
    )

    parser.add_argument(
        "--voltage",
        type=float,
        default=1.0,
        help="FEM voltage difference represented by the result (default: 1 V).",
    )

    parser.add_argument(
        "--rho-rt-ohm-m",
        type=float,
        default=None,
        help=(
            "Optional room-temperature resistivity in ohm m. "
            "If omitted, the resistivity stored in the FEM result is used."
        ),
    )

    parser.add_argument(
        "--output",
        default=None,
        help="Optional output text file.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_arguments()

    fem_path = Path(args.fem_npz).expanduser().resolve()

    if not fem_path.exists():
        raise FileNotFoundError(
            f"FEM result file not found:\n{fem_path}"
        )

    if args.voltage <= 0:
        raise ValueError("--voltage must be positive.")

    data = np.load(fem_path, allow_pickle=False)

    required = [
        "fem_current_A",
        "wire_width_m",
        "film_thickness_m",
        "normal_resistivity_ohm_m",
        "conductivity_S_per_m",
        "solve_voltage_V",
    ]

    missing = [
        key for key in required
        if key not in data.files
    ]

    if missing:
        raise RuntimeError(
            "FEM result is missing required fields:\n"
            + "\n".join(f"  - {key}" for key in missing)
            + "\n\nRun the updated analyze_current_crowding.py first."
        )

    fem_current_A = float(data["fem_current_A"])
    wire_width_m = float(data["wire_width_m"])
    thickness_m = float(data["film_thickness_m"])

    stored_rho = float(data["normal_resistivity_ohm_m"])
    stored_sigma = float(data["conductivity_S_per_m"])
    stored_voltage = float(data["solve_voltage_V"])

    rho_rt = (
        float(args.rho_rt_ohm_m)
        if args.rho_rt_ohm_m is not None
        else stored_rho
    )

    if rho_rt <= 0:
        raise ValueError("Room-temperature resistivity must be positive.")

    if fem_current_A <= 0 or not np.isfinite(fem_current_A):
        raise RuntimeError(
            f"FEM terminal current is not finite/positive: {fem_current_A}"
        )

    # The stored FEM solution is linear in voltage.
    # If the user requests a different voltage, scale the FEM current.
    current_at_requested_voltage_A = (
        fem_current_A
        * args.voltage
        / stored_voltage
    )

    if current_at_requested_voltage_A <= 0:
        raise RuntimeError("Calculated terminal current is not positive.")

    resistance_ohm = (
        args.voltage
        / current_at_requested_voltage_A
    )

    conductivity_rt = 1.0 / rho_rt

    # Geometry-derived active area from the FEM mesh.
    area_m2 = None
    if "triangle_areas_m2" in data.files:
        triangle_areas = np.asarray(
            data["triangle_areas_m2"],
            dtype=float,
        )
        finite_positive = (
            np.isfinite(triangle_areas)
            & (triangle_areas > 0.0)
        )
        if np.any(finite_positive):
            area_m2 = float(
                np.sum(triangle_areas[finite_positive])
            )

    effective_length_m = None
    simple_geometry_resistance_ohm = None

    if area_m2 is not None and wire_width_m > 0:
        # For a nominally uniform-width wire:
        #
        #     A_active ~= width * length
        #
        # therefore:
        #
        #     L_eff ~= A_active / width
        #
        effective_length_m = area_m2 / wire_width_m

        simple_geometry_resistance_ohm = (
            rho_rt
            * effective_length_m
            / (wire_width_m * thickness_m)
        )

    # Terminal currents for conservation checking.
    positive_current = (
        float(data["positive_terminal_current_A"])
        if "positive_terminal_current_A" in data.files
        else np.nan
    )

    negative_current = (
        float(data["negative_terminal_current_A"])
        if "negative_terminal_current_A" in data.files
        else np.nan
    )

    relative_conservation = (
        float(data["current_conservation_relative"])
        if "current_conservation_relative" in data.files
        else np.nan
    )

    # ------------------------------------------------------------
    # REPORT
    # ------------------------------------------------------------

    lines = []

    def p(line: str = "") -> None:
        print(line)
        lines.append(line)

    p()
    p("=" * 64)
    p("SNSPD ROOM-TEMPERATURE RESISTANCE")
    p("=" * 64)
    p()
    p("INPUT FEM RESULT")
    p("----------------")
    p(f"FEM result file             : {fem_path}")
    p(f"FEM solve voltage           : {stored_voltage:.9e} V")
    p(f"FEM terminal current        : {fem_current_A:.9e} A")
    p()
    p("NANOWIRE PARAMETERS")
    p("-------------------")
    p(f"Nanowire width              : {wire_width_m * 1e9:.6f} nm")
    p(f"Film thickness              : {thickness_m * 1e9:.6f} nm")
    p(f"Normal resistivity           : {rho_rt:.6e} ohm m")
    p(f"Normal conductivity          : {conductivity_rt:.6e} S/m")
    p()
    p("ROOM-TEMPERATURE / NORMAL-STATE RESULT")
    p("---------------------------------------")
    p(f"Applied voltage              : {args.voltage:.9e} V")
    p(
        f"Terminal current             : "
        f"{current_at_requested_voltage_A:.9e} A"
    )
    p(
        f"Resistance                   : "
        f"{resistance_ohm:.9e} ohm"
    )
    p(
        f"Resistance                   : "
        f"{resistance_ohm / 1e3:.6f} kOhm"
    )
    p(
        f"Resistance                   : "
        f"{resistance_ohm / 1e6:.6f} MOhm"
    )

    if area_m2 is not None:
        p()
        p("GEOMETRY SANITY CHECK")
        p("---------------------")
        p(f"FEM active area             : {area_m2 * 1e12:.6f} um²")

        if effective_length_m is not None:
            p(
                f"Area-derived effective length: "
                f"{effective_length_m * 1e6:.6f} um"
            )

        if simple_geometry_resistance_ohm is not None:
            p(
                f"rho*L/(w*t) estimate        : "
                f"{simple_geometry_resistance_ohm:.9e} ohm"
            )

            ratio = (
                resistance_ohm
                / simple_geometry_resistance_ohm
            )

            p(
                f"FEM / simple-area estimate  : "
                f"{ratio:.6f}"
            )

    p()
    p("TERMINAL CONSERVATION")
    p("---------------------")
    if np.isfinite(positive_current):
        p(
            f"Positive terminal current    : "
            f"{positive_current:.9e} A"
        )
    if np.isfinite(negative_current):
        p(
            f"Negative terminal current    : "
            f"{negative_current:.9e} A"
        )
    if np.isfinite(relative_conservation):
        p(
            f"Relative conservation error : "
            f"{relative_conservation:.9e}"
        )

    p()
    p("INTERPRETATION")
    p("--------------")
    p(
        "R_RT is obtained from the validated normal-state FEM "
        "terminal reaction current:"
    )
    p("    R_RT = V / I")
    p()
    p(
        "The result is therefore geometry-specific and includes "
        "the actual FEM terminal definition."
    )
    p()
    p(
        "The supplied normal resistivity is a material-model input. "
        "It is NOT automatically a universal experimental NbTiN "
        "room-temperature resistivity."
    )
    p()
    p("SNSPD room-temperature resistance analysis : PASS")

    if args.output is not None:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text("\n".join(lines) + "\n")
        print()
        print(f"Resistance report saved      : {output_path}")


if __name__ == "__main__":
    main()
