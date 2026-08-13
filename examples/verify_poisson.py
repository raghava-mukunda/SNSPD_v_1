# FILE: examples/verify_poisson.py
# PURPOSE:
# Verifies the first FEM solver against an analytical solution.
#
# PDE:
#
#     -∇²u = f
#
# Domain:
#
#     0 <= x <= 1
#     0 <= y <= 1
#
# Analytical solution:
#
#     u(x,y) = sin(pi*x) sin(pi*y)
#
# Therefore:
#
#     f(x,y)
#       = 2*pi²*sin(pi*x)*sin(pi*y)
#
# Boundary condition:
#
#     u = 0
#
# on the complete boundary.
#
# This test is NOT SNSPD physics.
#
# It verifies that our numerical FEM infrastructure is mathematically
# capable of solving a known PDE before we introduce electromagnetic
# and superconducting equations.


from __future__ import annotations


import numpy as np


from shapely.geometry import (
    Polygon,
)


from snspd.geometry.geometry import (
    DeviceGeometry,
    GeometryRegion,
)


from snspd.mesh.gmsh_mesher import (
    GmshMesher,
)


from snspd.fem.assembly import (
    assemble_poisson_system,
)


from snspd.fem.boundary import (
    find_boundary_nodes,
    apply_dirichlet_zero,
)


from snspd.fem.solver import (
    solve_linear_system,
)


# ============================================================
# ANALYTICAL SOLUTION
# ============================================================


def analytical_solution(
    x,
    y,
):
    """
    Exact solution:

        u = sin(pi*x) sin(pi*y)
    """

    return (
        np.sin(
            np.pi * x
        )
        * np.sin(
            np.pi * y
        )
    )


# ============================================================
# SOURCE TERM
# ============================================================


def source_function(
    x,
    y,
):
    """
    Source term corresponding to:

        -∇²u = f

    for the analytical solution.
    """

    return (
        2.0
        * np.pi**2
        * np.sin(
            np.pi * x
        )
        * np.sin(
            np.pi * y
        )
    )


# ============================================================
# ERROR
# ============================================================


def calculate_l2_node_error(
    nodes,
    numerical_solution,
):
    """
    Calculate the RMS nodal error.

    This is a simple verification metric.

    A later version will implement the mathematically proper
    element-integrated L2 norm.
    """

    exact = analytical_solution(
        nodes[:, 0],
        nodes[:, 1],
    )

    error = (
        numerical_solution
        - exact
    )

    return float(
        np.sqrt(
            np.mean(
                error**2
            )
        )
    )


# ============================================================
# CREATE UNIT SQUARE
# ============================================================


def create_unit_square_geometry():

    polygon = Polygon(
        [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
        ]
    )

    geometry = DeviceGeometry(
        source_format="analytical",
        source_file=None,
    )

    geometry.add_region(
        GeometryRegion(
            polygon=polygon,
            name="unit_square",
            material="verification",
        )
    )

    return geometry


# ============================================================
# MAIN
# ============================================================


def main():

    print(
        "\n"
        "========================================\n"
        "SNSPD FEM ANALYTICAL VERIFICATION\n"
        "========================================\n"
    )

    geometry = (
        create_unit_square_geometry()
    )

    errors = geometry.validate()

    if errors:

        raise RuntimeError(
            "Verification geometry invalid:\n"
            + "\n".join(errors)
        )

    # --------------------------------------------------------
    # Mesh
    # --------------------------------------------------------

    mesh_size = 0.05

    print(
        f"Generating mesh with h = "
        f"{mesh_size}"
    )

    mesher = GmshMesher(
        characteristic_length=mesh_size
    )

    mesh = mesher.generate(
        geometry
    )

    print(
        f"Nodes     : "
        f"{mesh.node_count}"
    )

    print(
        f"Triangles : "
        f"{mesh.element_count}"
    )

    # --------------------------------------------------------
    # Assemble
    # --------------------------------------------------------

    print(
        "Assembling FEM system..."
    )

    K, F = (
        assemble_poisson_system(
            mesh,
            source_function,
        )
    )

    # --------------------------------------------------------
    # Boundary conditions
    # --------------------------------------------------------

    boundary_nodes = (
        find_boundary_nodes(
            mesh.nodes,
            tolerance=1e-10,
        )
    )

    print(
        f"Boundary nodes : "
        f"{len(boundary_nodes)}"
    )

    K_bc, F_bc = (
        apply_dirichlet_zero(
            K,
            F,
            boundary_nodes,
        )
    )

    # --------------------------------------------------------
    # Solve
    # --------------------------------------------------------

    print(
        "Solving FEM system..."
    )

    solution = (
        solve_linear_system(
            K_bc,
            F_bc,
        )
    )

    # --------------------------------------------------------
    # Error
    # --------------------------------------------------------

    error = (
        calculate_l2_node_error(
            mesh.nodes,
            solution,
        )
    )

    print(
        "\n"
        "FEM VERIFICATION RESULT"
    )

    print(
        "======================="
    )

    print(
        f"RMS nodal error : "
        f"{error:.6e}"
    )

    # --------------------------------------------------------
    # Acceptance criterion
    # --------------------------------------------------------

    if error > 1e-3:

        raise RuntimeError(
            "FEM analytical verification FAILED."
        )

    print(
        "Verification : PASS"
    )


if __name__ == "__main__":

    main()