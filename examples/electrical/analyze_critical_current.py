"""
SNSPD Clem-Berggren critical-current analysis.

Input
-----
Validated FEM result:

    results/current_crowding_fem.npz

Required material quantities at the operating temperature:

    lambda
    xi

The FEM supplies:

    J(x,y)

which is converted to:

    K(x,y) = d J(x,y)

The Clem-Berggren vortex-entry model then determines
the geometry-limited critical current.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from snspd.physics.clem_berggren import (
    ClemBerggrenParameters,
    analyze_clem_berggren,
    format_result,
)


def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "SNSPD Clem-Berggren vortex-entry "
            "critical-current analysis."
        )
    )

    parser.add_argument(
        "fem_file",
        type=str,
        help=(
            "Validated FEM .npz result file."
        ),
    )

    parser.add_argument(
        "--lambda-nm",
        type=float,
        required=True,
        help=(
            "London penetration depth at the "
            "operating temperature [nm]."
        ),
    )

    parser.add_argument(
        "--xi-nm",
        type=float,
        required=True,
        help=(
            "Ginzburg-Landau coherence length "
            "at the operating temperature [nm]."
        ),
    )

    parser.add_argument(
        "--temperature-k",
        type=float,
        required=True,
        help="Operating temperature [K].",
    )

    parser.add_argument(
        "--material",
        type=str,
        default="unspecified",
        help="Material name.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "results/"
            "critical_current_clem_berggren.npz"
        ),
        help="Output .npz file.",
    )

    return parser.parse_args()


def main():

    args = parse_arguments()

    fem_path = Path(
        args.fem_file
    )

    if not fem_path.exists():

        raise FileNotFoundError(
            f"FEM result file not found: "
            f"{fem_path}"
        )

    print(
        "\n"
        "Loading validated FEM result..."
    )

    data = np.load(
        fem_path
    )

    required = [
        "nodes_m",
        "triangles",
        "triangle_centers_m",
        "element_J_magnitude_A_per_m2",
        "fem_current_A",
        "wire_width_m",
        "film_thickness_m",
    ]

    missing = [
        name
        for name in required
        if name not in data
    ]

    if missing:

        raise RuntimeError(
            "FEM result is missing required fields:\n"
            + "\n".join(
                f"    {name}"
                for name in missing
            )
        )

    nodes = data[
        "nodes_m"
    ]

    triangles = data[
        "triangles"
    ]

    triangle_centers = data[
        "triangle_centers_m"
    ]

    element_J = data[
        "element_J_magnitude_A_per_m2"
    ]

    fem_current = float(
        data[
            "fem_current_A"
        ]
    )

    wire_width = float(
        data[
            "wire_width_m"
        ]
    )

    film_thickness = float(
        data[
            "film_thickness_m"
        ]
    )

    # --------------------------------------------------------
    # Material parameters at operating temperature.
    #
    # IMPORTANT:
    # These are NOT silently inferred.
    # They must be supplied explicitly.
    # --------------------------------------------------------

    lambda_m = (
        args.lambda_nm
        * 1e-9
    )

    xi_m = (
        args.xi_nm
        * 1e-9
    )

    params = ClemBerggrenParameters(
        wire_width_m=wire_width,
        film_thickness_m=film_thickness,
        penetration_depth_m=lambda_m,
        coherence_length_m=xi_m,
        temperature_k=args.temperature_k,
        material=args.material,
    )

    print(
        "\n"
        "========================================================"
    )

    print(
        "\n"
        "INPUT"
    )

    print(
        "-----"
    )

    print(
        f"FEM result file          : "
        f"{fem_path.resolve()}"
    )

    print(
        f"Material                 : "
        f"{args.material}"
    )

    print(
        f"Temperature              : "
        f"{args.temperature_k:.6f} K"
    )

    print(
        f"Wire width               : "
        f"{wire_width * 1e9:.6f} nm"
    )

    print(
        f"Film thickness           : "
        f"{film_thickness * 1e9:.6f} nm"
    )

    print(
        f"Lambda                   : "
        f"{lambda_m * 1e9:.6f} nm"
    )

    print(
        f"Xi                       : "
        f"{xi_m * 1e9:.6f} nm"
    )

    print(
        f"FEM transport current    : "
        f"{fem_current:.12e} A"
    )

    print(
        f"Pearl length             : "
        f"{params.pearl_length_m * 1e6:.6f} um"
    )

    result = analyze_clem_berggren(
        nodes_m=nodes,
        triangles=triangles,
        triangle_centers_m=triangle_centers,
        element_J_magnitude_A_per_m2=element_J,
        fem_current_A=fem_current,
        params=params,
    )

    print(
        format_result(
            result
        )
    )

    # --------------------------------------------------------
    # Save numerical result.
    # --------------------------------------------------------

    output_path = Path(
        args.output
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if result.corners:

        corner_indices = np.array(
            [
                c.vertex_index
                for c in result.corners
            ],
            dtype=np.int64,
        )

        corner_x = np.array(
            [
                c.x_m
                for c in result.corners
            ],
            dtype=float,
        )

        corner_y = np.array(
            [
                c.y_m
                for c in result.corners
            ],
            dtype=float,
        )

        corner_angles = np.array(
            [
                c.interior_angle_rad
                for c in result.corners
            ],
            dtype=float,
        )

        corner_Ic = np.array(
            [
                c.critical_current_A
                for c in result.corners
            ],
            dtype=float,
        )

        corner_r2 = np.array(
            [
                c.fit_r2
                for c in result.corners
            ],
            dtype=float,
        )

    else:

        corner_indices = np.empty(
            0,
            dtype=np.int64,
        )

        corner_x = np.empty(
            0,
            dtype=float,
        )

        corner_y = np.empty(
            0,
            dtype=float,
        )

        corner_angles = np.empty(
            0,
            dtype=float,
        )

        corner_Ic = np.empty(
            0,
            dtype=float,
        )

        corner_r2 = np.empty(
            0,
            dtype=float,
        )

    np.savez_compressed(
        output_path,

        critical_current_A=float(
            result.critical_current_A
        ),

        straight_strip_critical_current_A=float(
            result.straight_strip_critical_current_A
        ),

        straight_strip_critical_sheet_current_A_per_m=float(
            result.straight_strip_critical_sheet_current_A_per_m
        ),

        pearl_length_m=float(
            result.pearl_length_m
        ),

        penetration_depth_m=float(
            result.penetration_depth_m
        ),

        coherence_length_m=float(
            result.coherence_length_m
        ),

        wire_width_m=float(
            wire_width
        ),

        film_thickness_m=float(
            film_thickness
        ),

        temperature_k=float(
            args.temperature_k
        ),

        fem_current_A=float(
            fem_current
        ),

        limiting_x_m=float(
            result.limiting_x_m
        ),

        limiting_y_m=float(
            result.limiting_y_m
        ),

        limiting_angle_deg=float(
            result.limiting_angle_deg
        ),

        limiting_K0_reference=float(
            result.limiting_K0_reference_A_per_m_power
        ),

        limiting_K0_critical=float(
            result.limiting_K0_critical_A_per_m_power
        ),

        corner_vertex_indices=corner_indices,

        corner_x_m=corner_x,

        corner_y_m=corner_y,

        corner_angles_rad=corner_angles,

        corner_critical_currents_A=corner_Ic,

        corner_fit_R2=corner_r2,
    )

    print(
        "\n"
        "Clem-Berggren result saved to:"
    )

    print(
        output_path.resolve()
    )


if __name__ == "__main__":

    main()