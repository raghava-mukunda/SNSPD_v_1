# FILE: examples/electrical/verify_current_distribution.py
# PURPOSE:
# Analytical verification of the SNSPD current-distribution
# FEM solver.
#
# Verification geometry:
#
#       0 <= x <= L
#       0 <= y <= W
#
# Exact solution:
#
#       V(x,y) = V0 * x / L
#
# Therefore:
#
#       E = -grad(V)
#
#       Ex = -V0/L
#       Ey = 0
#
# and:
#
#       Jx = sigma * V0/L
#       Jy = 0
#
# The current density must therefore be spatially uniform.
#
# This is a verification problem, not an SNSPD model.
#
# We must pass this test before using the solver on arbitrary
# SNSPD geometries.


from __future__ import annotations

import numpy as np

from shapely.geometry import Polygon

from snspd.geometry.geometry import (
    DeviceGeometry,
    GeometryRegion,
)

from snspd.mesh.gmsh_mesher import (
    GmshMesher,
)

from snspd.fem.electrical.current_distribution import (
    CurrentDistributionSolver,
)


# ============================================================
# ANALYTICAL SOLUTION
# ============================================================


def exact_potential(
    x: np.ndarray,
    y: np.ndarray,
    length: float,
    voltage: float,
) -> np.ndarray:

    return (
        voltage
        * x
        / length
    )


# ============================================================
# EXACT CURRENT DENSITY
# ============================================================


# FILE: examples/electrical/verify_current_distribution.py
# PURPOSE:
# Exact analytical current-density solution used to verify
# the stationary current-continuity FEM solver.
#
# For:
#
#     V(x,y) = V0*x/L
#
# we have:
#
#     E = -grad(V)
#
# and:
#
#     J = sigma*E
#       = -sigma*grad(V)
#
# therefore:
#
#     Jx = -sigma*V0/L
#     Jy = 0


def exact_current_density(
    conductivity: float,
    voltage: float,
    length: float,
) -> np.ndarray:

    Jx = (
        -conductivity
        * voltage
        / length
    )

    return np.array(
        [
            Jx,
            0.0,
        ],
        dtype=float,
    )

# ============================================================
# GEOMETRY
# ============================================================


def create_rectangle() -> DeviceGeometry:

    length = 1.0

    width = 0.5

    polygon = Polygon(
        [
            (0.0, 0.0),
            (length, 0.0),
            (length, width),
            (0.0, width),
        ]
    )

    geometry = DeviceGeometry(
        source_format="analytical",
        source_file=None,
    )

    geometry.add_region(
        GeometryRegion(
            polygon=polygon,
            name="verification_rectangle",
            material="verification",
        )
    )

    return geometry


# ============================================================
# TERMINAL DETECTION
# ============================================================


def find_terminal_nodes(
    nodes: np.ndarray,
    length: float,
    tolerance: float = 1.0e-10,
) -> tuple[np.ndarray, np.ndarray]:

    x = nodes[:, 0]

    negative = np.where(
        np.abs(x - 0.0)
        <= tolerance
    )[0]

    positive = np.where(
        np.abs(x - length)
        <= tolerance
    )[0]

    if len(negative) == 0:

        raise RuntimeError(
            "Negative terminal has no mesh nodes."
        )

    if len(positive) == 0:

        raise RuntimeError(
            "Positive terminal has no mesh nodes."
        )

    return (
        positive,
        negative,
    )


# ============================================================
# MAIN
# ============================================================


def main() -> None:

    print(
        "\n"
        "============================================\n"
        "SNSPD CURRENT DISTRIBUTION FEM VERIFICATION\n"
        "============================================\n"
    )

    length = 1.0

    width = 0.5

    conductivity = 1.0e6

    thickness = 5.0e-9

    voltage = 1.0

    geometry = (
        create_rectangle()
    )

    errors = geometry.validate()

    if errors:

        for error in errors:

            print(
                f"ERROR: {error}"
            )

        raise RuntimeError(
            "Verification geometry is invalid."
        )

    print(
        "Geometry validation : PASS"
    )

    mesher = GmshMesher(
        characteristic_length=0.05
    )

    mesh = mesher.generate(
        geometry
    )

    print(
        f"Nodes     : {mesh.node_count}"
    )

    print(
        f"Triangles : {mesh.element_count}"
    )

    positive_nodes, negative_nodes = (
        find_terminal_nodes(
            mesh.nodes,
            length,
        )
    )

    print(
        f"Positive terminal nodes : "
        f"{len(positive_nodes)}"
    )

    print(
        f"Negative terminal nodes : "
        f"{len(negative_nodes)}"
    )

    solver = CurrentDistributionSolver(
        nodes=mesh.nodes,
        triangles=mesh.triangles,
        conductivity=conductivity,
        thickness=thickness,
    )

    result = solver.solve(
        positive_terminal_nodes=positive_nodes,
        negative_terminal_nodes=negative_nodes,
        voltage_difference=voltage,
    )

    # ========================================================
    # POTENTIAL ERROR
    # ========================================================

    exact_V = exact_potential(
        mesh.nodes[:, 0],
        mesh.nodes[:, 1],
        length,
        voltage,
    )

    potential_error = (
        result.potential
        - exact_V
    )

    potential_rms = float(
        np.sqrt(
            np.mean(
                potential_error**2
            )
        )
    )

    # ========================================================
    # CURRENT DENSITY ERROR
    # ========================================================

    exact_J = exact_current_density(
        conductivity,
        voltage,
        length,
    )

    numerical_J = (
        result.element_current_density
    )

    J_error = (
        numerical_J
        - exact_J[None, :]
    )

    J_rms = float(
        np.sqrt(
            np.mean(
                np.sum(
                    J_error**2,
                    axis=1,
                )
            )
        )
    )

    # ========================================================
    # RELATIVE CURRENT-DENSITY ERROR
    # ========================================================

    exact_J_magnitude = np.linalg.norm(
        exact_J
    )

    relative_J_error = (
        J_rms
        / exact_J_magnitude
    )

    # ========================================================
    # EXPECTED TOTAL CURRENT
    # ========================================================

    expected_current = (
        conductivity
        * thickness
        * width
        * voltage
        / length
    )

    current_error = abs(
        result.total_current
        - expected_current
    )

    relative_current_error = (
        current_error
        / expected_current
    )

    # ========================================================
    # OUTPUT
    # ========================================================

    print(
        "\n"
        "# VERIFICATION RESULT"
    )

    print(
        "\n"
        f"Potential RMS error       : "
        f"{potential_rms:.8e}"
    )

    print(
        f"Current-density RMS error : "
        f"{J_rms:.8e}"
    )

    print(
        f"Relative J error          : "
        f"{relative_J_error:.8e}"
    )

    print(
        f"\nExpected current         : "
        f"{expected_current:.8e} A"
    )

    print(
        f"Computed current         : "
        f"{result.total_current:.8e} A"
    )

    print(
        f"Relative current error   : "
        f"{relative_current_error:.8e}"
    )

    print(
        f"\nMaximum |J|              : "
        f"{result.maximum_current_density:.8e} A/m^2"
    )

    print(
        f"Expected |J|             : "
        f"{exact_J_magnitude:.8e} A/m^2"
    )

    # ========================================================
    # PASS/FAIL
    # ========================================================

    tolerance = 5.0e-3

    if relative_J_error > tolerance:

        raise RuntimeError(
            "Current-density verification FAILED."
        )

    if relative_current_error > tolerance:

        raise RuntimeError(
            "Terminal-current verification FAILED."
        )

    print(
        "\n"
        "Potential solution        : PASS"
    )

    print(
        "Current-density solution   : PASS"
    )

    print(
        "Terminal-current solution  : PASS"
    )

    print(
        "\n"
        "Current distribution FEM verification : PASS"
    )


if __name__ == "__main__":

    main()