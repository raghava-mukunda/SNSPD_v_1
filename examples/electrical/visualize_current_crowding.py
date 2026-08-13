# FILE: examples/electrical/visualize_current_crowding.py
#
# PURPOSE:
# Run the verified SNSPD electrical FEM solver and visualize
# the resulting element-wise current-density distribution.
#
# Pipeline:
#
#     SVG
#       ↓
#     DeviceGeometry
#       ↓
#     FEM mesh
#       ↓
#     CurrentDistributionSolver
#       ↓
#     Element-wise J
#       ↓
#     Current-density heat map
#       ↓
#     Local crowding-factor map


from __future__ import annotations

from pathlib import Path

import numpy as np

from snspd.geometry.svg_importer import (
    import_svg,
)

from snspd.mesh.gmsh_mesher import (
    GmshMesher,
)

from snspd.fem.electrical.current_distribution import (
    CurrentDistributionSolver,
)

from snspd.visualization.current_density_plot import (
    plot_current_density,
    plot_crowding_factor,
)


# ============================================================
# CONFIGURATION
# ============================================================


NANOWIRE_WIDTH = 1.0e-6

FILM_THICKNESS = 5.0e-9

NORMAL_RESISTIVITY = 1.0e-6

APPLIED_VOLTAGE = 1.0

CHARACTERISTIC_LENGTH = 0.25e-6


# ============================================================
# TERMINAL DETECTION
# ============================================================


def find_terminal_nodes(
    nodes: np.ndarray,
    tolerance: float = 1.0e-12,
) -> tuple[np.ndarray, np.ndarray]:

    x = nodes[:, 0]

    xmin = np.min(
        x
    )

    xmax = np.max(
        x
    )

    negative_nodes = np.where(
        np.abs(
            x - xmin
        )
        <= tolerance
    )[0]

    positive_nodes = np.where(
        np.abs(
            x - xmax
        )
        <= tolerance
    )[0]

    if len(negative_nodes) == 0:

        raise RuntimeError(
            "No negative-terminal nodes detected."
        )

    if len(positive_nodes) == 0:

        raise RuntimeError(
            "No positive-terminal nodes detected."
        )

    return (
        positive_nodes,
        negative_nodes,
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print(
        "\n"
        "====================================================\n"
        "SNSPD FEM CURRENT-DENSITY VISUALIZATION\n"
        "====================================================\n"
    )

    # --------------------------------------------------------
    # Geometry
    # --------------------------------------------------------

    geometry_file = (
        Path(__file__).resolve().parents[1]
        / "simple_meander.svg"
    )

    print(
        f"\nGeometry file : {geometry_file}"
    )

    geometry = import_svg(
        geometry_file
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
            "Cannot continue with invalid geometry."
        )

    print(
        geometry.summary()
    )

    print(
        "\nGeometry validation : PASS"
    )

    # --------------------------------------------------------
    # Material / electrical parameters
    # --------------------------------------------------------

    conductivity = (
        1.0
        / NORMAL_RESISTIVITY
    )

    print(
        "\n"
        "Nanowire electrical parameters\n"
        "-------------------------------"
    )

    print(
        f"Nanowire width          : "
        f"{NANOWIRE_WIDTH * 1e9:.6f} nm"
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
        f"{conductivity:.6e} S/m"
    )

    # --------------------------------------------------------
    # Mesh
    # --------------------------------------------------------

    print(
        "\nGenerating FEM mesh..."
    )

    mesher = GmshMesher(
        characteristic_length=(
            CHARACTERISTIC_LENGTH
        )
    )

    mesh = mesher.generate(
        geometry
    )

    mesh_errors = (
        mesh.validate()
    )

    if mesh_errors:

        for error in mesh_errors:

            print(
                f"ERROR: {error}"
            )

        raise RuntimeError(
            "Generated FEM mesh is invalid."
        )

    print(
        mesh.summary()
    )

    print(
        "\nMesh validation : PASS"
    )

    # --------------------------------------------------------
    # Electrical terminals
    # --------------------------------------------------------

    positive_nodes, negative_nodes = (
        find_terminal_nodes(
            mesh.nodes
        )
    )

    print(
        "\nElectrical terminals"
        "\n--------------------"
    )

    print(
        f"Positive terminal nodes : "
        f"{len(positive_nodes)}"
    )

    print(
        f"Negative terminal nodes : "
        f"{len(negative_nodes)}"
    )

    # --------------------------------------------------------
    # FEM electrical solve
    # --------------------------------------------------------

    print(
        "\nSolving stationary electrical FEM..."
    )

    solver = CurrentDistributionSolver(
        nodes=mesh.nodes,
        triangles=mesh.triangles,
        conductivity=conductivity,
        thickness=FILM_THICKNESS,
    )

    result = solver.solve(
        positive_terminal_nodes=positive_nodes,
        negative_terminal_nodes=negative_nodes,
        voltage_difference=APPLIED_VOLTAGE,
    )

    # --------------------------------------------------------
    # FEM result
    # --------------------------------------------------------

    terminal_current = abs(
        result.total_current
    )

    J = (
        result.element_current_density
    )

    J_magnitude = np.linalg.norm(
        J,
        axis=1,
    )

    hotspot_element = int(
        np.argmax(
            J_magnitude
        )
    )

    hotspot_nodes = (
        mesh.triangles[
            hotspot_element
        ]
    )

    hotspot_position = np.mean(
        mesh.nodes[
            hotspot_nodes
        ],
        axis=0,
    )

    J_transport = (
        terminal_current
        / (
            NANOWIRE_WIDTH
            * FILM_THICKNESS
        )
    )

    maximum_crowding = (
        np.max(
            J_magnitude
        )
        / J_transport
    )

    # --------------------------------------------------------
    # REPORT
    # --------------------------------------------------------

    print(
        "\n"
        "===================================================="
    )

    print(
        "\nCURRENT-DENSITY RESULT"
        "\n----------------------"
    )

    print(
        f"Terminal current       : "
        f"{terminal_current:.9e} A"
    )

    print(
        f"Maximum |J|            : "
        f"{np.max(J_magnitude):.9e} A/m²"
    )

    print(
        f"Transport J            : "
        f"{J_transport:.9e} A/m²"
    )

    print(
        f"Maximum local C_J      : "
        f"{maximum_crowding:.9f}"
    )

    print(
        f"Hotspot element        : "
        f"{hotspot_element}"
    )

    print(
        f"Hotspot x              : "
        f"{hotspot_position[0] * 1e6:.9f} µm"
    )

    print(
        f"Hotspot y              : "
        f"{hotspot_position[1] * 1e6:.9f} µm"
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    output_directory = (
        Path(__file__).resolve().parents[1]
        / "results"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Physical current-density heat map
    # --------------------------------------------------------

    plot_current_density(
        mesh,
        J,
        terminal_current=terminal_current,
        nanowire_width=NANOWIRE_WIDTH,
        film_thickness=FILM_THICKNESS,
        positive_terminal_nodes=positive_nodes,
        negative_terminal_nodes=negative_nodes,
        title=(
            "SNSPD FEM CURRENT DENSITY |J|"
        ),
        save_path=(
            output_directory
            / "current_density_heatmap.png"
        ),
        show=True,
    )

    # --------------------------------------------------------
    # Normalized crowding map
    # --------------------------------------------------------

    plot_crowding_factor(
        mesh,
        J,
        terminal_current=terminal_current,
        nanowire_width=NANOWIRE_WIDTH,
        film_thickness=FILM_THICKNESS,
        save_path=(
            output_directory
            / "current_crowding_factor.png"
        ),
        show=True,
    )

    print(
        "\n"
        "===================================================="
    )

    print(
        "\nCurrent-density visualization : PASS"
    )

    print(
        "Current-crowding visualization : PASS"
    )


if __name__ == "__main__":

    main()