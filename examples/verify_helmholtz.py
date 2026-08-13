# FILE: examples/verify_helmholtz.py
# PURPOSE:
# Verifies the complex-valued frequency-domain FEM implementation.
#
# PDE:
#
#     -∇²u - k²u = f
#
# Exact solution:
#
#     u(x,y)
#       =
#       (1 + 0.5j)
#       sin(πx)
#       sin(πy)
#
# Source:
#
#     f
#       =
#       (2π² - k²)
#       (1 + 0.5j)
#       sin(πx)
#       sin(πy)
#
# Boundary:
#
#     u = 0
#
# This is a numerical benchmark for complex-valued
# frequency-domain FEM.
#
# It is NOT yet the full Maxwell equations.


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


from snspd.fem.helmholtz import (
    assemble_helmholtz_system,
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
# WAVE NUMBER
# ============================================================

WAVE_NUMBER = 4.0


# ============================================================
# EXACT SOLUTION
# ============================================================

def analytical_solution(
    x,
    y,
):
    """
    Exact complex-valued solution.
    """

    return (
        (1.0 + 0.5j)
        * np.sin(
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
    Exact complex gradient.
    """

    amplitude = (
        1.0 + 0.5j
    )

    du_dx = (
        amplitude
        * np.pi
        * np.cos(
            np.pi * x
        )
        * np.sin(
            np.pi * y
        )
    )

    du_dy = (
        amplitude
        * np.pi
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
        ],
        dtype=complex,
    )


# ============================================================
# SOURCE
# ============================================================

def source_function(
    x,
    y,
):
    """
    Manufactured complex source.

    For:

        u = A sin(πx) sin(πy)

    where:

        A = 1 + 0.5j

    and:

        -∇²u = 2π²u,

    the Helmholtz source is:

        f = (2π² - k²)u.
    """

    amplitude = (
        1.0 + 0.5j
    )

    return (
        (
            2.0 * np.pi**2
            -
            WAVE_NUMBER**2
        )
        * amplitude
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
# SINGLE SOLUTION
# ============================================================

def solve_for_mesh_size(
    geometry,
    mesh_size,
):

    mesher = GmshMesher(
        characteristic_length=mesh_size
    )

    mesh = mesher.generate(
        geometry
    )

    # --------------------------------------------------------
    # Assemble:
    #
    #     K - k²M
    #
    # --------------------------------------------------------

    A, F = (
        assemble_helmholtz_system(
            mesh,
            source_function,
            WAVE_NUMBER,
        )
    )

    # --------------------------------------------------------
    # Boundary
    # --------------------------------------------------------

    boundary_nodes = (
        find_boundary_nodes(
            mesh.nodes,
            tolerance=1e-10,
        )
    )

    A_bc, F_bc = (
        apply_dirichlet_zero(
            A,
            F,
            boundary_nodes,
        )
    )

    # --------------------------------------------------------
    # Solve
    # --------------------------------------------------------

    solution = (
        solve_linear_system(
            A_bc,
            F_bc,
        )
    )

    # --------------------------------------------------------
    # Errors
    # --------------------------------------------------------

    l2_error = (
        calculate_l2_error(
            mesh,
            solution,
            analytical_solution,
        )
    )

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
# MAIN
# ============================================================

def main():

    print(
        "\n"
        "====================================================\n"
        "COMPLEX HELMHOLTZ FEM VERIFICATION\n"
        "====================================================\n"
    )

    print(
        f"Wave number k = "
        f"{WAVE_NUMBER:.6f}"
    )

    print(
        "\n"
        "Exact solution:\n"
        "\n"
        "    u(x,y) = "
        "(1 + 0.5j) sin(πx) sin(πy)\n"
    )

    geometry = (
        create_unit_square_geometry()
    )

    mesh_sizes = [
        0.20,
        0.10,
        0.05,
        0.025,
        0.0125,
    ]

    results = []

    # ========================================================
    # MESH SWEEP
    # ========================================================

    for h in mesh_sizes:

        print(
            f"\nSolving h = "
            f"{h:.6f}"
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
    # CONVERGENCE TABLE
    # ========================================================

    print(
        "\n"
        "===================================================="
    )

    print(
        "COMPLEX HELMHOLTZ CONVERGENCE"
    )

    print(
        "===================================================="
    )

    print(
        f"{'h':>10}"
        f"{'Nodes':>10}"
        f"{'L2 Error':>18}"
        f"{'L2 Order':>12}"
        f"{'H1 Error':>18}"
        f"{'H1 Order':>12}"
    )

    previous_h = None
    previous_l2 = None
    previous_h1 = None

    for result in results:

        h = result["h"]

        l2 = result["l2_error"]

        h1 = result["h1_error"]

        if previous_h is None:

            l2_order = "---"
            h1_order = "---"

        else:

            l2_order = (
                np.log(
                    previous_l2
                    / l2
                )
                /
                np.log(
                    previous_h
                    / h
                )
            )

            h1_order = (
                np.log(
                    previous_h1
                    / h1
                )
                /
                np.log(
                    previous_h
                    / h
                )
            )

            l2_order = (
                f"{l2_order:.6f}"
            )

            h1_order = (
                f"{h1_order:.6f}"
            )

        print(
            f"{h:10.6f}"
            f"{result['nodes']:10d}"
            f"{l2:18.8e}"
            f"{l2_order:>12}"
            f"{h1:18.8e}"
            f"{h1_order:>12}"
        )

        previous_h = h
        previous_l2 = l2
        previous_h1 = h1

    # ========================================================
    # MONOTONICITY
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

    l2_pass = np.all(
        np.diff(
            l2_errors
        ) < 0
    )

    h1_pass = np.all(
        np.diff(
            h1_errors
        ) < 0
    )

    print(
        "\n"
        "L2 error decreasing : "
        f"{'PASS' if l2_pass else 'FAIL'}"
    )

    print(
        "H1 error decreasing : "
        f"{'PASS' if h1_pass else 'FAIL'}"
    )

    # ========================================================
    # COMPLEXITY CHECK
    # ========================================================

    # Solve the finest mesh again to inspect the imaginary
    # component explicitly.

    (
        mesh,
        solution,
        _,
        _,
    ) = solve_for_mesh_size(
        geometry,
        mesh_sizes[-1],
    )

    max_real = np.max(
        np.abs(
            np.real(solution)
        )
    )

    max_imag = np.max(
        np.abs(
            np.imag(solution)
        )
    )

    print(
        "\n"
        "COMPLEX FIELD CHECK"
    )

    print(
        "==================="
    )

    print(
        f"Maximum |Re(u)| : "
        f"{max_real:.8e}"
    )

    print(
        f"Maximum |Im(u)| : "
        f"{max_imag:.8e}"
    )

    if max_imag <= 1e-12:

        raise RuntimeError(
            "Complex FEM verification FAILED: "
            "imaginary field component vanished."
        )

    if not l2_pass:

        raise RuntimeError(
            "Complex Helmholtz verification FAILED: "
            "L2 error is not monotonically convergent."
        )

    if not h1_pass:

        raise RuntimeError(
            "Complex Helmholtz verification FAILED: "
            "H1 error is not monotonically convergent."
        )

    print(
        "\n"
        "===================================================="
    )

    print(
        "COMPLEX HELMHOLTZ FEM VERIFICATION : PASS"
    )

    print(
        "===================================================="
    )


if __name__ == "__main__":

    main()