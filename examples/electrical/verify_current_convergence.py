# FILE: examples/electrical/verify_current_convergence.py
# PURPOSE:
# Manufactured-solution mesh-convergence verification of the
# stationary electrical FEM solver.
#
# PDE:
#
#     -div(sigma * grad(V)) = f
#
# Domain:
#
#     0 <= x <= 1
#     0 <= y <= 1
#
# Exact solution:
#
#     V(x,y) = sin(pi*x) sin(pi*y)
#
# Since:
#
#     Laplacian(V)
#       =
#       -2*pi^2*sin(pi*x)*sin(pi*y)
#
# the manufactured source is:
#
#     f =
#       2*sigma*pi^2*sin(pi*x)*sin(pi*y)
#
# Exact current density:
#
#     J = -sigma*grad(V)
#
#     Jx =
#       -sigma*pi*cos(pi*x)*sin(pi*y)
#
#     Jy =
#       -sigma*pi*sin(pi*x)*cos(pi*y)
#
# This solution is NOT exactly representable by P1 elements,
# so genuine discretization error is produced.
#
# Expected asymptotic behavior:
#
#     L2(V)  ~ O(h^2)
#     L2(J)  ~ O(h)
#
# for smooth solutions with P1 elements.


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
# CONSTANTS
# ============================================================


SIGMA = 1.0e6

THICKNESS = 5.0e-9

PI = np.pi


# ============================================================
# EXACT POTENTIAL
# ============================================================


def exact_potential(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:

    return (
        np.sin(PI * x)
        * np.sin(PI * y)
    )


# ============================================================
# EXACT CURRENT
# ============================================================


def exact_current_density(
    x: float,
    y: float,
) -> np.ndarray:

    return np.array(
        [
            -SIGMA
            * PI
            * np.cos(PI * x)
            * np.sin(PI * y),

            -SIGMA
            * PI
            * np.sin(PI * x)
            * np.cos(PI * y),
        ],
        dtype=float,
    )


# ============================================================
# MANUFACTURED SOURCE
# ============================================================


def manufactured_source(
    x: float,
    y: float,
) -> float:

    return (
        2.0
        * SIGMA
        * PI**2
        * np.sin(PI * x)
        * np.sin(PI * y)
    )


# ============================================================
# GEOMETRY
# ============================================================


def create_unit_square() -> DeviceGeometry:

    polygon = Polygon(
        [
            (0.0, 0.0),
            (1.0, 0.0),
            (1.0, 1.0),
            (0.0, 1.0),
        ]
    )

    geometry = DeviceGeometry(
        source_format="manufactured",
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
# BOUNDARY NODES
# ============================================================


def find_boundary_nodes(
    nodes: np.ndarray,
    tolerance: float = 1.0e-10,
) -> np.ndarray:

    x = nodes[:, 0]

    y = nodes[:, 1]

    mask = (
        (np.abs(x - 0.0) <= tolerance)
        |
        (np.abs(x - 1.0) <= tolerance)
        |
        (np.abs(y - 0.0) <= tolerance)
        |
        (np.abs(y - 1.0) <= tolerance)
    )

    return np.where(mask)[0]


# ============================================================
# SINGLE SOLVE
# ============================================================


def solve_for_mesh_size(
    geometry: DeviceGeometry,
    mesh_size: float,
) -> dict:

    mesher = GmshMesher(
        characteristic_length=mesh_size
    )

    mesh = mesher.generate(
        geometry
    )

    boundary_nodes = (
        find_boundary_nodes(
            mesh.nodes
        )
    )

    # Exact solution is zero on the entire boundary.

    prescribed_values = {
        int(node): 0.0
        for node in boundary_nodes
    }

    solver = CurrentDistributionSolver(
        nodes=mesh.nodes,
        triangles=mesh.triangles,
        conductivity=SIGMA,
        thickness=THICKNESS,
        source=manufactured_source,
    )

    # --------------------------------------------------------
    # Directly assemble and solve with all boundary nodes
    # prescribed to zero.
    # --------------------------------------------------------

    K, F = (
        solver.assemble_system()
    )

    K_bc, F_bc = (
        solver.apply_dirichlet(
            K,
            F,
            prescribed_values,
        )
    )

    from scipy.sparse.linalg import spsolve

    potential = spsolve(
        K_bc,
        F_bc,
    )

    # --------------------------------------------------------
    # Current density
    # --------------------------------------------------------

    element_current = (
        solver.calculate_element_current_density(
            potential
        )
    )

    # --------------------------------------------------------
    # Potential L2 error
    #
    # Exact integration of the numerical error is not
    # possible from nodal values alone. We therefore evaluate
    # the FEM interpolant at the same three-point quadrature
    # points used by the source integration.
    # --------------------------------------------------------

    potential_error_integral = 0.0

    current_error_integral = 0.0

    for element_index, triangle in enumerate(
        mesh.triangles
    ):

        indices = np.asarray(
            triangle,
            dtype=int,
        )

        p1 = mesh.nodes[
            indices[0]
        ]

        p2 = mesh.nodes[
            indices[1]
        ]

        p3 = mesh.nodes[
            indices[2]
        ]

        area = (
            0.5
            * abs(
                (
                    (p2[0] - p1[0])
                    * (p3[1] - p1[1])
                )
                -
                (
                    (p3[0] - p1[0])
                    * (p2[1] - p1[1])
                )
            )
        )

        element_nodes = potential[
            indices
        ]

        B = (
            solver
            .triangle_gradient_matrix
            if hasattr(
                solver,
                "triangle_gradient_matrix",
            )
            else None
        )

        # P1 gradient is constant over the element.

        from snspd.fem.electrical.current_distribution import (
            triangle_gradient_matrix,
            triangle_quadrature,
        )

        gradient_matrix = (
            triangle_gradient_matrix(
                p1,
                p2,
                p3,
            )
        )

        numerical_gradient = (
            gradient_matrix
            @ element_nodes
        )

        numerical_J = (
            -SIGMA
            * numerical_gradient
        )

        for point, weight in (
            triangle_quadrature(
                p1,
                p2,
                p3,
            )
        ):

            x = point[0]

            y = point[1]

            # Barycentric coordinates.

            denominator = (
                (
                    p2[1]
                    - p3[1]
                )
                * (
                    p1[0]
                    - p3[0]
                )
                +
                (
                    p3[0]
                    - p2[0]
                )
                * (
                    p1[1]
                    - p3[1]
                )
            )

            N1 = (
                (
                    p2[1]
                    - p3[1]
                )
                * (
                    x
                    - p3[0]
                )
                +
                (
                    p3[0]
                    - p2[0]
                )
                * (
                    y
                    - p3[1]
                )
            ) / denominator

            N2 = (
                (
                    p3[1]
                    - p1[1]
                )
                * (
                    x
                    - p3[0]
                )
                +
                (
                    p1[0]
                    - p3[0]
                )
                * (
                    y
                    - p3[1]
                )
            ) / denominator

            N3 = (
                1.0
                - N1
                - N2
            )

            numerical_V = (
                N1 * element_nodes[0]
                + N2 * element_nodes[1]
                + N3 * element_nodes[2]
            )

            exact_V = exact_potential(
                x,
                y,
            )

            exact_J = exact_current_density(
                x,
                y,
            )

            potential_difference = (
                numerical_V
                - exact_V
            )

            current_difference = (
                numerical_J
                - exact_J
            )

            potential_error_integral += (
                weight
                * potential_difference**2
            )

            current_error_integral += (
                weight
                * np.dot(
                    current_difference,
                    current_difference,
                )
            )

    potential_l2 = float(
        np.sqrt(
            potential_error_integral
        )
    )

    current_l2 = float(
        np.sqrt(
            current_error_integral
        )
    )

    # --------------------------------------------------------
    # Exact norms for relative errors
    # --------------------------------------------------------

    # Numerical quadrature is used for the exact norm too,
    # ensuring consistent integration treatment.

    exact_potential_norm_integral = 0.0

    exact_current_norm_integral = 0.0

    for element_index, triangle in enumerate(
        mesh.triangles
    ):

        indices = np.asarray(
            triangle,
            dtype=int,
        )

        p1 = mesh.nodes[
            indices[0]
        ]

        p2 = mesh.nodes[
            indices[1]
        ]

        p3 = mesh.nodes[
            indices[2]
        ]

        from snspd.fem.electrical.current_distribution import (
            triangle_quadrature,
        )

        for point, weight in (
            triangle_quadrature(
                p1,
                p2,
                p3,
            )
        ):

            x = point[0]

            y = point[1]

            V_exact = exact_potential(
                x,
                y,
            )

            J_exact = exact_current_density(
                x,
                y,
            )

            exact_potential_norm_integral += (
                weight
                * V_exact**2
            )

            exact_current_norm_integral += (
                weight
                * np.dot(
                    J_exact,
                    J_exact,
                )
            )

    relative_potential_l2 = (
        potential_l2
        /
        np.sqrt(
            exact_potential_norm_integral
        )
    )

    relative_current_l2 = (
        current_l2
        /
        np.sqrt(
            exact_current_norm_integral
        )
    )

    return {
        "h": mesh_size,
        "nodes": mesh.node_count,
        "triangles": mesh.element_count,
        "potential_l2": potential_l2,
        "relative_potential_l2": relative_potential_l2,
        "current_l2": current_l2,
        "relative_current_l2": relative_current_l2,
    }


# ============================================================
# CONVERGENCE ORDER
# ============================================================


def convergence_order(
    previous_error: float,
    current_error: float,
    previous_h: float,
    current_h: float,
) -> float:

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


def main() -> None:

    print(
        "\n"
        "====================================================\n"
        "SNSPD ELECTRICAL FEM MANUFACTURED-SOLUTION TEST\n"
        "====================================================\n"
    )

    print(
        "\n"
        "PDE:\n\n"
        "    -div(sigma * grad(V)) = f\n\n"
        "Exact solution:\n\n"
        "    V = sin(pi*x) sin(pi*y)\n\n"
        "Manufactured source:\n\n"
        "    f = 2*sigma*pi^2*sin(pi*x) sin(pi*y)\n"
    )

    geometry = (
        create_unit_square()
    )

    errors = geometry.validate()

    if errors:

        raise RuntimeError(
            "Manufactured-solution geometry is invalid."
        )

    print(
        "Geometry validation : PASS"
    )

    mesh_sizes = [
        0.20,
        0.10,
        0.05,
        0.025,
        0.0125,
    ]

    results = []

    for h in mesh_sizes:

        print(
            f"\nSolving h = {h:.6f}"
        )

        result = (
            solve_for_mesh_size(
                geometry,
                h,
            )
        )

        results.append(
            result
        )

        print(
            f"Nodes          : "
            f"{result['nodes']}"
        )

        print(
            f"Triangles      : "
            f"{result['triangles']}"
        )

        print(
            f"L2(V)          : "
            f"{result['potential_l2']:.8e}"
        )

        print(
            f"Relative L2(V) : "
            f"{result['relative_potential_l2']:.8e}"
        )

        print(
            f"L2(J)          : "
            f"{result['current_l2']:.8e}"
        )

        print(
            f"Relative L2(J) : "
            f"{result['relative_current_l2']:.8e}"
        )

    # ========================================================
    # TABLE
    # ========================================================

    print(
        "\n"
        "====================================================\n"
        "CONVERGENCE RESULTS\n"
        "===================================================="
    )

    print(
        f"\n"
        f"{'h':>10}"
        f"{'Nodes':>10}"
        f"{'L2(V)':>16}"
        f"{'V Order':>12}"
        f"{'L2(J)':>16}"
        f"{'J Order':>12}"
    )

    previous = None

    for result in results:

        h = result["h"]

        V_error = (
            result[
                "relative_potential_l2"
            ]
        )

        J_error = (
            result[
                "relative_current_l2"
            ]
        )

        if previous is None:

            V_order_text = "---"

            J_order_text = "---"

        else:

            V_order = convergence_order(
                previous[
                    "relative_potential_l2"
                ],
                V_error,
                previous["h"],
                h,
            )

            J_order = convergence_order(
                previous[
                    "relative_current_l2"
                ],
                J_error,
                previous["h"],
                h,
            )

            V_order_text = (
                f"{V_order:.6f}"
            )

            J_order_text = (
                f"{J_order:.6f}"
            )

        print(
            f"{h:10.6f}"
            f"{result['nodes']:10d}"
            f"{V_error:16.8e}"
            f"{V_order_text:>12}"
            f"{J_error:16.8e}"
            f"{J_order_text:>12}"
        )

        previous = result

    # ========================================================
    # VALIDATION
    # ========================================================

    V_errors = np.array(
        [
            result[
                "relative_potential_l2"
            ]

            for result in results
        ]
    )

    J_errors = np.array(
        [
            result[
                "relative_current_l2"
            ]

            for result in results
        ]
    )

    if not np.all(
        np.diff(V_errors) < 0.0
    ):

        raise RuntimeError(
            "Potential L2 error did not decrease monotonically."
        )

    if not np.all(
        np.diff(J_errors) < 0.0
    ):

        raise RuntimeError(
            "Current-density L2 error did not decrease monotonically."
        )

    # --------------------------------------------------------
    # Determine asymptotic orders using the final refinements.
    # --------------------------------------------------------

    V_orders = []

    J_orders = []

    for i in range(
        1,
        len(results),
    ):

        V_orders.append(
            convergence_order(
                V_errors[i - 1],
                V_errors[i],
                results[i - 1]["h"],
                results[i]["h"],
            )
        )

        J_orders.append(
            convergence_order(
                J_errors[i - 1],
                J_errors[i],
                results[i - 1]["h"],
                results[i]["h"],
            )
        )

    # The final three refinements should approach the expected
    # asymptotic behavior.

    final_V_order = np.mean(
        V_orders[-3:]
    )

    final_J_order = np.mean(
        J_orders[-3:]
    )

    print(
        "\n"
        "===================================================="
    )

    print(
        f"\nFinal asymptotic V order : "
        f"{final_V_order:.6f}"
    )

    print(
        f"Final asymptotic J order : "
        f"{final_J_order:.6f}"
    )

    print(
        "\n"
        "Potential L2 monotonicity      : PASS"
    )

    print(
        "Current-density L2 monotonicity: PASS"
    )

    # Broad theoretical checks rather than demanding an
    # exact order at every individual mesh.

    if not (
        1.7
        <= final_V_order
        <= 2.3
    ):

        raise RuntimeError(
            "Potential convergence order is outside "
            "the expected P1 FEM asymptotic range."
        )

    if not (
        0.7
        <= final_J_order
        <= 1.3
    ):

        raise RuntimeError(
            "Current-density convergence order is outside "
            "the expected P1 FEM asymptotic range."
        )

    print(
        "Potential convergence order   : PASS"
    )

    print(
        "Current convergence order     : PASS"
    )

    print(
        "\n"
        "Electrical FEM manufactured-solution "
        "verification : PASS"
    )


if __name__ == "__main__":

    main()