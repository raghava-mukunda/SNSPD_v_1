# FILE: examples/electrical/analyze_current_crowding.py
# PURPOSE:
# Performs a stationary electrical FEM analysis of an SNSPD
# meander and calculates current crowding directly from the
# FEM current-density field.
#
# Physical formulation:
#
#     div(sigma * grad(V)) = 0
#
#     E = -grad(V)
#
#     J = sigma E
#       = -sigma grad(V)
#
# Terminal current:
#
#     I = t * integral_Gamma (J . n) ds
#
# Current-crowding factor:
#
#     C_J = J_max / J_avg,terminal
#
# where:
#
#     J_avg,terminal = I / (t * L_terminal)
#
# This analysis is still a NORMAL-STATE electrical model.
#
# It does not yet calculate superconducting critical current.
#
# The resulting J(x,y) distribution will later be coupled
# to the superconducting constitutive and critical-current
# model.


from __future__ import annotations


# ============================================================
# IMPORTS
# ============================================================

from pathlib import Path

import numpy as np


from snspd.geometry.svg_importer import (
    import_svg,
)

from snspd.geometry.analyzer import (
    analyze_geometry,
    format_metrics,
)

from snspd.mesh.gmsh_mesher import (
    GmshMesher,
)

from snspd.mesh.quality import (
    analyze_mesh_quality,
    format_mesh_quality,
)

from snspd.fem.electrical.current_distribution import (
    CurrentDistributionSolver,
)

from snspd.fem.electrical.terminal import (
    build_boundary_edges,
    select_terminal_boundary,
)


# ============================================================
# PHYSICAL PARAMETERS
# ============================================================

FILM_THICKNESS = 5.0e-9

NORMAL_RESISTIVITY = 1.0e-6

CONDUCTIVITY = (
    1.0
    / NORMAL_RESISTIVITY
)

SOLVE_VOLTAGE = 1.0

MESH_SIZE = 0.25e-6

TERMINAL_TOLERANCE = 1.0e-12


# ============================================================
# TARGET BIAS
# ============================================================

TARGET_BIAS_CURRENT = 10.0e-6


# ============================================================
# GEOMETRY
# ============================================================


def load_geometry():

    svg_file = (
        Path(__file__).resolve().parents[1]
        / "simple_meander.svg"
    )

    if not svg_file.exists():

        raise FileNotFoundError(
            f"Geometry file not found:\n"
            f"{svg_file}"
        )

    print(
        f"Geometry file : {svg_file}"
    )

    return import_svg(
        svg_file
    )


# ============================================================
# NANOWIRE WIDTH
# ============================================================


def get_nanowire_width(
    geometry,
) -> float:
    """
    Retrieve nanowire width from geometry metadata.

    This quantity is reported as a geometry/material property.
    It is NOT used to calculate the terminal current.
    """

    widths = []

    for region in geometry.regions:

        width = region.metadata.get(
            "width_m"
        )

        if width is not None:

            widths.append(
                float(width)
            )

    if not widths:

        raise RuntimeError(
            "No nanowire width metadata found."
        )

    width = widths[0]

    if width <= 0.0:

        raise RuntimeError(
            "Nanowire width must be positive."
        )

    if not np.allclose(
        widths,
        width,
        rtol=1.0e-6,
        atol=1.0e-15,
    ):

        raise RuntimeError(
            "Inconsistent nanowire widths found."
        )

    return width


# ============================================================
# MAIN
# ============================================================


def main():

    print(
        "\n"
        "====================================================\n"
        "SNSPD CURRENT CROWDING ANALYSIS\n"
        "====================================================\n"
    )

    # ========================================================
    # GEOMETRY
    # ========================================================

    geometry = load_geometry()

    print(
        geometry.summary()
    )

    geometry_errors = (
        geometry.validate()
    )

    if geometry_errors:

        print(
            "\nGEOMETRY VALIDATION FAILED"
        )

        for error in geometry_errors:

            print(
                f"ERROR: {error}"
            )

        raise RuntimeError(
            "Invalid SNSPD geometry."
        )

    print(
        "Geometry validation : PASS"
    )

    # ========================================================
    # GEOMETRY METRICS
    # ========================================================

    metrics = (
        analyze_geometry(
            geometry
        )
    )

    print(
        format_metrics(
            metrics
        )
    )

    # ========================================================
    # WIRE WIDTH
    # ========================================================

    wire_width = (
        get_nanowire_width(
            geometry
        )
    )

    print(
        "\n"
        "Nanowire electrical parameters"
    )

    print(
        "-------------------------------"
    )

    print(
        f"Nanowire width          : "
        f"{wire_width * 1e9:.6f} nm"
    )

    print(
        f"Film thickness          : "
        f"{FILM_THICKNESS * 1e9:.6f} nm"
    )

    print(
        f"Normal resistivity      : "
        f"{NORMAL_RESISTIVITY:.6e} ohm m"
    )

    print(
        f"Normal conductivity     : "
        f"{CONDUCTIVITY:.6e} S/m"
    )

    # ========================================================
    # MESH
    # ========================================================

    print(
        "\nGenerating FEM mesh..."
    )

    mesher = GmshMesher(
        characteristic_length=MESH_SIZE
    )

    mesh = mesher.generate(
        geometry
    )

    print(
        mesh.summary()
    )

    mesh_errors = (
        mesh.validate()
    )

    if mesh_errors:

        print(
            "\nMESH VALIDATION FAILED"
        )

        for error in mesh_errors:

            print(
                f"ERROR: {error}"
            )

        raise RuntimeError(
            "Invalid FEM mesh."
        )

    print(
        "Mesh validation : PASS"
    )

    # ========================================================
    # MESH QUALITY
    # ========================================================

    quality = (
        analyze_mesh_quality(
            mesh
        )
    )

    print(
        format_mesh_quality(
            quality
        )
    )

    # ========================================================
    # GLOBAL BOUNDARY
    # ========================================================

    boundary_edges = (
        build_boundary_edges(
            mesh.nodes,
            mesh.triangles,
        )
    )

    print(
        "\n"
        "FEM boundary"
    )

    print(
        "------------"
    )

    print(
        f"Total boundary edges : "
        f"{len(boundary_edges)}"
    )

    # ========================================================
    # TERMINAL LOCATIONS
    # ========================================================

    xmin = float(
        np.min(
            mesh.nodes[:, 0]
        )
    )

    xmax = float(
        np.max(
            mesh.nodes[:, 0]
        )
    )

    positive_terminal = (
        select_terminal_boundary(
            boundary_edges,
            axis=0,
            coordinate=xmax,
            tolerance=TERMINAL_TOLERANCE,
            name="POSITIVE",
        )
    )

    negative_terminal = (
        select_terminal_boundary(
            boundary_edges,
            axis=0,
            coordinate=xmin,
            tolerance=TERMINAL_TOLERANCE,
            name="NEGATIVE",
        )
    )

    print(
        "\n"
        "Electrical terminals"
    )

    print(
        "--------------------"
    )

    print(
        f"Positive terminal edges : "
        f"{positive_terminal.edge_count}"
    )

    print(
        f"Negative terminal edges : "
        f"{negative_terminal.edge_count}"
    )

    print(
        f"Positive terminal length : "
        f"{positive_terminal.length_m * 1e6:.6f} um"
    )

    print(
        f"Negative terminal length : "
        f"{negative_terminal.length_m * 1e6:.6f} um"
    )

    # ========================================================
    # FEM ELECTRICAL SOLUTION
    # ========================================================

    print(
        "\nSolving stationary electrical FEM..."
    )

    solver = CurrentDistributionSolver(
        nodes=mesh.nodes,
        triangles=mesh.triangles,
        conductivity=CONDUCTIVITY,
        thickness=FILM_THICKNESS,
    )

    # --------------------------------------------------------
    # Terminal DOFs
    # --------------------------------------------------------

    # Remove duplicate nodes because adjacent boundary edges
    # share vertices.
    positive_nodes = np.unique(
        np.array(
            [
                node
                for edge in positive_terminal.edges
                for node in (
                    edge.node_a,
                    edge.node_b,
                )
            ],
            dtype=int,
        )
    )

    negative_nodes = np.unique(
        np.array(
            [
                node
                for edge in negative_terminal.edges
                for node in (
                    edge.node_a,
                    edge.node_b,
                )
            ],
            dtype=int,
        )
    )

    print(
        f"\nPositive terminal nodes : "
        f"{len(positive_nodes)}"
    )

    print(
        f"Negative terminal nodes : "
        f"{len(negative_nodes)}"
    )

    # --------------------------------------------------------
    # Solve
    # --------------------------------------------------------

    result = solver.solve(
        positive_terminal_nodes=positive_nodes,
        negative_terminal_nodes=negative_nodes,
        voltage_difference=SOLVE_VOLTAGE,
    )

    # --------------------------------------------------------
    # Terminal currents
    # --------------------------------------------------------
    #
    # IMPORTANT:
    #
    # Terminal currents are taken from the FEM reaction vector
    # returned by CurrentDistributionSolver.
    #
    # We do NOT independently integrate the element J field
    # along terminal edges here.
    #
    # For the weak formulation
    #
    #     ∫ sigma grad(V) · grad(w) dΩ = 0
    #
    # with Dirichlet terminal constraints, the reaction vector
    # gives the discrete boundary current associated with the
    # prescribed potential. This is the authoritative global
    # transport-current calculation.
    #
    # The independent edge integration was removed because it
    # introduces a separate approximation and was previously
    # producing a small artificial terminal-current imbalance.

    required_result_fields = (
        "positive_terminal_current",
        "negative_terminal_current",
        "total_current",
    )

    missing_fields = [
        name
        for name in required_result_fields
        if not hasattr(result, name)
    ]

    if missing_fields:
        raise RuntimeError(
            "CurrentDistributionResult is missing the FEM "
            "reaction-current fields: "
            + ", ".join(missing_fields)
            + ". Update "
            "src/snspd/fem/electrical/current_distribution.py "
            "to return FEM terminal reaction currents."
        )

    positive_terminal_current = float(
        result.positive_terminal_current
    )

    negative_terminal_current = float(
        result.negative_terminal_current
    )

    terminal_current = float(
        result.total_current
    )

    # --------------------------------------------------------
    # Current conservation
    # --------------------------------------------------------

    terminal_current_difference = abs(
        positive_terminal_current
        - negative_terminal_current
    )

    terminal_current_scale = max(
        positive_terminal_current,
        negative_terminal_current,
    )

    if terminal_current_scale > 0.0:
        current_difference = (
            terminal_current_difference
            / terminal_current_scale
        )
    else:
        current_difference = 0.0

    print(
        "\n"
        "POSITIVE TERMINAL\n"
        "-----------------"
    )

    print(
        f"FEM reaction current : "
        f"{positive_terminal_current:.12e} A"
    )

    print(
        "\n"
        "NEGATIVE TERMINAL\n"
        "-----------------"
    )

    print(
        f"FEM reaction current : "
        f"{negative_terminal_current:.12e} A"
    )

    print(
        "\n"
        "TERMINAL CURRENT CONSERVATION\n"
        "-----------------------------"
    )

    print(
        f"Positive |I| : "
        f"{positive_terminal_current:.12e} A"
    )

    print(
        f"Negative |I| : "
        f"{negative_terminal_current:.12e} A"
    )

    print(
        f"Absolute difference : "
        f"{terminal_current_difference:.12e} A"
    )

    print(
        f"Relative difference : "
        f"{current_difference:.12e}"
    )

    # ========================================================
    # CURRENT-DENSITY METRICS
    # ========================================================

    element_J = np.asarray(
        result.element_current_density,
        dtype=float,
    )

    element_J_magnitude = np.linalg.norm(
        element_J,
        axis=1,
    )

    element_areas = (
        mesh.triangle_areas()
    )

    total_area = float(
        np.sum(
            element_areas
        )
    )

    maximum_J_index = int(
        np.argmax(
            element_J_magnitude
        )
    )

    maximum_J = float(
        element_J_magnitude[
            maximum_J_index
        ]
    )

    area_average_J = float(
        np.sum(
            element_J_magnitude
            * element_areas
        )
        / total_area
    )

    # ========================================================
    # TERMINAL-AVERAGE CURRENT DENSITY
    # ========================================================

    terminal_length = (
        0.5
        * (
            positive_terminal.length_m
            +
            negative_terminal.length_m
        )
    )

    if terminal_length <= 0.0:

        raise RuntimeError(
            "Terminal length is zero."
        )

    terminal_average_J = (
        terminal_current
        / (
            FILM_THICKNESS
            * terminal_length
        )
    )

    # ========================================================
    # CURRENT CROWDING
    # ========================================================

    crowding_factor = (
        maximum_J
        / terminal_average_J
    )

    # ========================================================
    # MAXIMUM-J LOCATION
    # ========================================================

    triangle_nodes = (
        mesh.triangles[
            maximum_J_index
        ]
    )

    triangle_points = (
        mesh.nodes[
            triangle_nodes
        ]
    )

    hotspot = np.mean(
        triangle_points,
        axis=0,
    )

    # ========================================================
    # REPORT
    # ========================================================

    print(
        "\n"
        "===================================================="
    )

    print(
        "\nCURRENT DISTRIBUTION"
    )

    print(
        "--------------------"
    )

    print(
        f"Applied FEM voltage       : "
        f"{SOLVE_VOLTAGE:.6e} V"
    )

    print(
        f"Computed terminal current : "
        f"{terminal_current:.9e} A"
    )

    print(
        "Terminal current source    : FEM reaction vector"
    )

    print(
        f"Maximum |J|               : "
        f"{maximum_J:.9e} A/m²"
    )

    print(
        f"Area-weighted <|J|>       : "
        f"{area_average_J:.9e} A/m²"
    )

    print(
        f"Terminal-average |J|      : "
        f"{terminal_average_J:.9e} A/m²"
    )

    print(
        f"Current crowding factor   : "
        f"{crowding_factor:.9f}"
    )

    # ========================================================
    # HOTSPOT
    # ========================================================

    print(
        "\n"
        "CURRENT-DENSITY HOTSPOT"
    )

    print(
        "-----------------------"
    )

    print(
        f"Element index             : "
        f"{maximum_J_index}"
    )

    print(
        f"x                         : "
        f"{hotspot[0] * 1e6:.9f} um"
    )

    print(
        f"y                         : "
        f"{hotspot[1] * 1e6:.9f} um"
    )

    # ========================================================
    # BIAS SCALING
    # ========================================================

    scale = (
        TARGET_BIAS_CURRENT
        / terminal_current
    )

    bias_max_J = (
        maximum_J
        * scale
    )

    print(
        "\n"
        "SCALED BIAS-CURRENT RESULT"
    )

    print(
        "---------------------------"
    )

    print(
        f"Target bias current       : "
        f"{TARGET_BIAS_CURRENT:.9e} A"
    )

    print(
        f"Scaling factor            : "
        f"{scale:.9e}"
    )

    print(
        f"Maximum |J| at bias       : "
        f"{bias_max_J:.9e} A/m²"
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    if not np.isfinite(
        terminal_current
    ):

        raise RuntimeError(
            "Terminal current is not finite."
        )

    if terminal_current <= 0.0:

        raise RuntimeError(
            "Terminal current must be positive."
        )

    if not np.isfinite(
        crowding_factor
    ):

        raise RuntimeError(
            "Current crowding factor is not finite."
        )

    if crowding_factor <= 0.0:

        raise RuntimeError(
            "Current crowding factor must be positive."
        )

    # For the discrete source-free FEM system, the terminal
    # reaction currents must balance up to numerical roundoff.
    #
    # Do not loosen this tolerance to hide a solver or boundary
    # selection error.

    if current_difference > 1.0e-10:

        raise RuntimeError(
            "Terminal current conservation FAILED."
        )

    print(
        "\n"
        "INTERPRETATION"
    )

    print(
        "--------------"
    )

    print(
        "Terminal current:"
    )

    print(
        "    I = t * integral(J.n ds)"
    )

    print(
        "\nTerminal-average current density:"
    )

    print(
        "    J_avg = I / (t * L_terminal)"
    )

    print(
        "\nCurrent-crowding factor:"
    )

    print(
        "    C_J = J_max / J_avg"
    )

    print(
        "\nC_J approximately 1:"
    )

    print(
        "    approximately uniform current distribution"
    )

    print(
        "\nC_J greater than 1:"
    )

    print(
        "    current crowding is present"
    )

    print(
        "\n"
        "SNSPD current-crowding analysis : PASS"
    )


# ============================================================
# ENTRY POINT
# ============================================================


if __name__ == "__main__":

    main()