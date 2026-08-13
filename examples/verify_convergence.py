# FILE: examples/verify_convergence.py
# PURPOSE:
# Performs rigorous mesh-convergence verification of the P1 FEM
# Poisson solver.
#
# The same analytical PDE is solved on progressively finer meshes.
#
# PDE:
#
#     -∇²u = 2π² sin(πx) sin(πy)
#
# Exact solution:
#
#     u(x,y) = sin(πx) sin(πy)
#
# Boundary condition:
#
#     u = 0
#
# on the complete boundary.
#
# The verification uses two mathematically defined FEM error norms:
#
#     L2 error:
#
#         ||u - uh||_L2
#
#     H1 seminorm error:
#
#         |u - uh|_H1
#
# The observed convergence order is calculated for both norms.
#
# This test is independent of SNSPD-specific physics and exists to
# verify the numerical FEM foundation before electromagnetic,
# superconducting, thermal, and detection physics are introduced.


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


from snspd.fem.error import (
    calculate_l2_error,
    calculate_h1_seminorm_error,
)


# ============================================================
# EXACT SOLUTION
# ============================================================

def analytical_solution(
    x,
    y,
):
    """
    Exact analytical solution:

        u(x,y) = sin(πx) sin(πy)
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
# EXACT GRADIENT
# ============================================================

def analytical_gradient(
    x,
    y,
):
    """
    Exact gradient of:

        u(x,y) = sin(πx) sin(πy)

    Therefore:

        du/dx =
            π cos(πx) sin(πy)

        du/dy =
            π sin(πx) cos(πy)

    Returns
    -------
    ndarray
        [du/dx, du/dy]
    """

    du_dx = (
        np.pi
        * np.cos(
            np.pi * x
        )
        * np.sin(
            np.pi * y
        )
    )

    du_dy = (
        np.pi
        * np.sin(
            np.pi * x
        )
        * np.cos(
            np.pi * y
        )
    )

    return np.array(
        [
            du_dx,
            du_dy,
        ]
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

    for:

        u(x,y) = sin(πx) sin(πy)

    Since:

        ∇²u
        =
        -2π² sin(πx) sin(πy)

    we have:

        f
        =
        2π² sin(πx) sin(πy)
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
# GEOMETRY
# ============================================================

def create_unit_square_geometry():
    """
    Create the analytical verification domain:

        0 <= x <= 1
        0 <= y <= 1
    """

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
# SINGLE CONVERGENCE RUN
# ============================================================

def solve_for_mesh_size(
    geometry,
    mesh_size,
):
    """
    Solve the analytical Poisson problem for one mesh size.

    Returns
    -------
    mesh
        Generated FEM mesh.

    solution
        Numerical FEM solution.

    l2_error
        Proper domain-integrated L2 error.

    h1_error
        Proper H1 seminorm error.
    """

    # --------------------------------------------------------
    # Generate FEM mesh
    # --------------------------------------------------------

    mesher = GmshMesher(
        characteristic_length=mesh_size
    )

    mesh = mesher.generate(
        geometry
    )

    # --------------------------------------------------------
    # Assemble global FEM system
    #
    #     K u = F
    # --------------------------------------------------------

    K, F = (
        assemble_poisson_system(
            mesh,
            source_function,
        )
    )

    # --------------------------------------------------------
    # Identify boundary nodes
    # --------------------------------------------------------

    boundary_nodes = (
        find_boundary_nodes(
            mesh.nodes,
            tolerance=1e-10,
        )
    )

    # --------------------------------------------------------
    # Apply:
    #
    #     u = 0
    #
    # on the complete boundary.
    # --------------------------------------------------------

    K_bc, F_bc = (
        apply_dirichlet_zero(
            K,
            F,
            boundary_nodes,
        )
    )

    # --------------------------------------------------------
    # Solve linear FEM system
    # --------------------------------------------------------

    solution = (
        solve_linear_system(
            K_bc,
            F_bc,
        )
    )

    # --------------------------------------------------------
    # Proper L2 error
    #
    #     ||u - uh||_L2
    #
    # --------------------------------------------------------

    l2_error = (
        calculate_l2_error(
            mesh,
            solution,
            analytical_solution,
        )
    )

    # --------------------------------------------------------
    # Proper H1 seminorm error
    #
    #     |u - uh|_H1
    #
    # --------------------------------------------------------

    h1_error = (
        calculate_h1_seminorm_error(
            mesh,
            solution,
            analytical_gradient,
        )
    )

    return (
        mesh,
        solution,
        l2_error,
        h1_error,
    )


# ============================================================
# CONVERGENCE ORDER
# ============================================================

def calculate_convergence_order(
    previous_error,
    current_error,
    previous_h,
    current_h,
):
    """
    Calculate observed convergence order:

        p =
            log(E_h / E_h2)
            ----------------
            log(h / h2)

    """

    if (
        previous_error <= 0
        or current_error <= 0
    ):

        return np.nan

    return (
        np.log(
            previous_error
            / current_error
        )
        /
        np.log(
            previous_h
            / current_h
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        "====================================================\n"
        "SNSPD FEM MESH-CONVERGENCE VERIFICATION\n"
        "====================================================\n"
    )

    print(
        "Verification PDE:\n"
        "\n"
        "    -∇²u = 2π² sin(πx) sin(πy)\n"
        "\n"
        "Exact solution:\n"
        "\n"
        "    u(x,y) = sin(πx) sin(πy)\n"
    )

    # ========================================================
    # CREATE ANALYTICAL DOMAIN
    # ========================================================

    geometry = (
        create_unit_square_geometry()
    )

    geometry_errors = (
        geometry.validate()
    )

    if geometry_errors:

        raise RuntimeError(
            "Verification geometry is invalid:\n"
            + "\n".join(
                geometry_errors
            )
        )

    # ========================================================
    # MESH SIZES
    # ========================================================

    mesh_sizes = [
        0.20,
        0.10,
        0.05,
        0.025,
        0.0125,
    ]

    results = []

    # ========================================================
    # SOLVE EACH MESH
    # ========================================================

    for h in mesh_sizes:

        print(
            f"\n"
            f"Solving h = {h:.6f}"
        )

        (
            mesh,
            solution,
            l2_error,
            h1_error,
        ) = solve_for_mesh_size(
            geometry,
            h,
        )

        print(
            f"    Nodes     : "
            f"{mesh.node_count}"
        )

        print(
            f"    Triangles : "
            f"{mesh.element_count}"
        )

        print(
            f"    L2 error  : "
            f"{l2_error:.8e}"
        )

        print(
            f"    H1 error  : "
            f"{h1_error:.8e}"
        )

        results.append(
            {
                "h": h,
                "nodes": mesh.node_count,
                "triangles": mesh.element_count,
                "l2_error": l2_error,
                "h1_error": h1_error,
            }
        )

    # ========================================================
    # CONVERGENCE RESULTS
    # ========================================================

    print(
        "\n"
        "===================================================="
    )

    print(
        "CONVERGENCE RESULTS"
    )

    print(
        "===================================================="
    )

    print(
        f"{'h':>10}"
        f"{'Nodes':>10}"
        f"{'Triangles':>12}"
        f"{'L2 Error':>18}"
        f"{'L2 Order':>12}"
        f"{'H1 Error':>18}"
        f"{'H1 Order':>12}"
    )

    # ========================================================
    # PREVIOUS VALUES
    # ========================================================

    previous_h = None

    previous_l2_error = None

    previous_h1_error = None

    # ========================================================
    # PRINT TABLE
    # ========================================================

    for result in results:

        h = result["h"]

        l2_error = (
            result["l2_error"]
        )

        h1_error = (
            result["h1_error"]
        )

        # ----------------------------------------------------
        # First mesh has no previous result.
        # ----------------------------------------------------

        if previous_h is None:

            l2_order_text = "---"

            h1_order_text = "---"

        else:

            l2_order = (
                calculate_convergence_order(
                    previous_l2_error,
                    l2_error,
                    previous_h,
                    h,
                )
            )

            h1_order = (
                calculate_convergence_order(
                    previous_h1_error,
                    h1_error,
                    previous_h,
                    h,
                )
            )

            l2_order_text = (
                f"{l2_order:.6f}"
            )

            h1_order_text = (
                f"{h1_order:.6f}"
            )

        print(
            f"{h:10.6f}"
            f"{result['nodes']:10d}"
            f"{result['triangles']:12d}"
            f"{l2_error:18.8e}"
            f"{l2_order_text:>12}"
            f"{h1_error:18.8e}"
            f"{h1_order_text:>12}"
        )

        previous_h = h

        previous_l2_error = (
            l2_error
        )

        previous_h1_error = (
            h1_error
        )

    # ========================================================
    # EXTRACT ERROR ARRAYS
    # ========================================================

    l2_errors = np.array(
        [
            result["l2_error"]
            for result in results
        ]
    )

    h1_errors = np.array(
        [
            result["h1_error"]
            for result in results
        ]
    )

    # ========================================================
    # MONOTONICITY CHECK
    # ========================================================
    #
    # A correctly converging solution should show decreasing
    # error as the mesh is refined.
    #
    # ========================================================

    l2_monotonic = np.all(
        np.diff(
            l2_errors
        ) < 0
    )

    h1_monotonic = np.all(
        np.diff(
            h1_errors
        ) < 0
    )

    print(
        "\n"
        "===================================================="
    )

    print(
        "ERROR MONOTONICITY"
    )

    print(
        "===================================================="
    )

    print(
        "L2 error decreasing : "
        f"{'PASS' if l2_monotonic else 'FAIL'}"
    )

    print(
        "H1 error decreasing : "
        f"{'PASS' if h1_monotonic else 'FAIL'}"
    )

    if not l2_monotonic:

        raise RuntimeError(
            "FEM convergence verification FAILED: "
            "L2 error did not decrease monotonically."
        )

    if not h1_monotonic:

        raise RuntimeError(
            "FEM convergence verification FAILED: "
            "H1 error did not decrease monotonically."
        )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print(
        "\n"
        "===================================================="
    )

    print(
        "FEM CONVERGENCE VERIFICATION : PASS"
    )

    print(
        "===================================================="
    )

    print(
        "\n"
        "Both the L2 and H1 error norms decrease "
        "monotonically under mesh refinement."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()