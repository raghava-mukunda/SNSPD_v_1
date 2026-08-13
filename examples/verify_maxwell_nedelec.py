# FILE: examples/verify_maxwell_nedelec.py
# PURPOSE:
# Verifies the first-order Nedelec H(curl) finite-element
# implementation against a manufactured analytical Maxwell solution.
#
# Governing equation:
#
#     curl(mu^-1 curl(E))
#     -
#     omega² epsilon E
#     =
#     J
#
# Parameters:
#
#     mu = 1
#     epsilon = 1
#
# Exact electric field:
#
#     E_x = sin(πx) sin(πy)
#     E_y = 0
#
# The source is analytically derived from the exact field.
#
# PEC boundary conditions are imposed on all four boundaries.
#
# The test measures:
#
#     - vector L2 error
#     - curl error
#     - mesh convergence
#
# This is the first actual H(curl)-conforming electromagnetic
# verification of the simulator.


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


from snspd.fem.solver import (
    solve_linear_system,
)


from snspd.fem.electromagnetics.maxwell import (
    assemble_maxwell_system,
)


from snspd.fem.electromagnetics.boundary import (
    apply_pec_boundary,
)


# ============================================================
# PHYSICAL / MATHEMATICAL PARAMETERS
# ============================================================

MU = 1.0

EPSILON = 1.0

OMEGA = 2.0


# ============================================================
# EXACT ELECTRIC FIELD
# ============================================================

def exact_electric_field(
    x,
    y,
):
    """
    Exact field:

        E =
        [ sin(πx) sin(πy) ]
        [        0        ]
    """

    return np.array(
        [
            np.sin(
                np.pi * x
            )
            * np.sin(
                np.pi * y
            ),
            0.0,
        ],
        dtype=complex,
    )


# ============================================================
# EXACT CURL
# ============================================================

def exact_curl(
    x,
    y,
):
    """
    curl(E):

        dEy/dx - dEx/dy

        =
        -π sin(πx) cos(πy)
    """

    return (
        -np.pi
        * np.sin(
            np.pi * x
        )
        * np.cos(
            np.pi * y
        )
    )


# ============================================================
# MANUFACTURED SOURCE
# ============================================================

def source_function(
    x,
    y,
):
    """
    Exact source:

        J =
        curl(mu^-1 curl(E))
        -
        omega² epsilon E

    with:

        mu = 1
        epsilon = 1.
    """

    ex = (
        np.sin(
            np.pi * x
        )
        * np.sin(
            np.pi * y
        )
    )

    jx = (
        (
            np.pi**2
            -
            OMEGA**2
        )
        * ex
    )

    jy = (
        np.pi**2
        * np.cos(
            np.pi * x
        )
        * np.cos(
            np.pi * y
        )
    )

    return np.array(
        [
            jx,
            jy,
        ],
        dtype=complex,
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
            name="maxwell_verification_domain",
            material="vacuum",
        )
    )

    return geometry


# ============================================================
# VECTOR FIELD RECONSTRUCTION
# ============================================================

def reconstruct_field_at_point(
    mesh,
    topology,
    solution,
    element_index,
    point,
):
    """
    Reconstruct the Nedelec finite-element electric field
    inside one triangle.
    """

    from snspd.fem.electromagnetics.nedelec import (
        NedelecTriangle,
    )

    coordinates = (
        mesh.nodes[
            mesh.triangles[
                element_index
            ]
        ]
    )

    element = NedelecTriangle(
        coordinates
    )

    local_edges = (
        topology.triangle_edges[
            element_index
        ]
    )

    signs = (
        topology.triangle_edge_signs[
            element_index
        ]
    )

    field = np.zeros(
        2,
        dtype=complex,
    )

    for local_edge in range(3):

        global_edge = int(
            local_edges[
                local_edge
            ]
        )

        sign = (
            signs[
                local_edge
            ]
        )

        basis = (
            element.basis_function(
                local_edge,
                point,
            )
        )

        field += (
            sign
            * solution[
                global_edge
            ]
            * basis
        )

    return field


# ============================================================
# FIELD L2 ERROR
# ============================================================

def calculate_field_l2_error(
    mesh,
    topology,
    solution,
):
    """
    Calculate:

        ||E - Eh||_L2

        =
        sqrt(
            integral |E-Eh|² dA
        )
    """

    total = 0.0

    quadrature = np.array(
        [
            [
                1.0 / 6.0,
                1.0 / 6.0,
                2.0 / 3.0,
            ],
            [
                1.0 / 6.0,
                2.0 / 3.0,
                1.0 / 6.0,
            ],
            [
                2.0 / 3.0,
                1.0 / 6.0,
                1.0 / 6.0,
            ],
        ]
    )

    for element_index in range(
        mesh.element_count
    ):

        coordinates = (
            mesh.nodes[
                mesh.triangles[
                    element_index
                ]
            ]
        )

        x1, y1 = coordinates[0]
        x2, y2 = coordinates[1]
        x3, y3 = coordinates[2]

        area = abs(
            0.5
            * (
                (x2 - x1)
                * (y3 - y1)
                -
                (x3 - x1)
                * (y2 - y1)
            )
        )

        for barycentric in quadrature:

            point = (
                barycentric
                @ coordinates
            )

            numerical = (
                reconstruct_field_at_point(
                    mesh,
                    topology,
                    solution,
                    element_index,
                    point,
                )
            )

            exact = (
                exact_electric_field(
                    point[0],
                    point[1],
                )
            )

            error = (
                exact
                -
                numerical
            )

            total += (
                np.vdot(
                    error,
                    error,
                ).real
                * area
                / 3.0
            )

    return float(
        np.sqrt(
            max(
                total,
                0.0,
            )
        )
    )


# ============================================================
# CURL ERROR
# ============================================================

def calculate_curl_l2_error(
    mesh,
    topology,
    solution,
):
    """
    Calculate:

        ||curl(E) - curl(Eh)||_L2.
    """

    from snspd.fem.electromagnetics.nedelec import (
        NedelecTriangle,
    )

    total = 0.0

    for element_index in range(
        mesh.element_count
    ):

        coordinates = (
            mesh.nodes[
                mesh.triangles[
                    element_index
                ]
            ]
        )

        x1, y1 = coordinates[0]
        x2, y2 = coordinates[1]
        x3, y3 = coordinates[2]

        area = abs(
            0.5
            * (
                (x2 - x1)
                * (y3 - y1)
                -
                (x3 - x1)
                * (y2 - y1)
            )
        )

        element = NedelecTriangle(
            coordinates
        )

        local_edges = (
            topology.triangle_edges[
                element_index
            ]
        )

        signs = (
            topology.triangle_edge_signs[
                element_index
            ]
        )

        numerical_curl = 0.0

        for local_edge in range(3):

            global_edge = int(
                local_edges[
                    local_edge
                ]
            )

            numerical_curl += (
                signs[
                    local_edge
                ]
                * solution[
                    global_edge
                ]
                * element.basis_curl(
                    local_edge
                )
            )

        # The numerical curl is constant within
        # a first-order Nedelec triangle.
        #
        # The exact curl is not constant, so use
        # three-point quadrature.

        quadrature = np.array(
            [
                [
                    1.0 / 6.0,
                    1.0 / 6.0,
                    2.0 / 3.0,
                ],
                [
                    1.0 / 6.0,
                    2.0 / 3.0,
                    1.0 / 6.0,
                ],
                [
                    2.0 / 3.0,
                    1.0 / 6.0,
                    1.0 / 6.0,
                ],
            ]
        )

        for barycentric in quadrature:

            point = (
                barycentric
                @ coordinates
            )

            exact = (
                exact_curl(
                    point[0],
                    point[1],
                )
            )

            error = (
                exact
                -
                numerical_curl
            )

            total += (
                abs(error)**2
                * area
                / 3.0
            )

    return float(
        np.sqrt(
            max(
                total,
                0.0,
            )
        )
    )


# ============================================================
# SINGLE SOLVE
# ============================================================

def solve_mesh(
    geometry,
    mesh_size,
):

    mesher = GmshMesher(
        characteristic_length=mesh_size
    )

    mesh = mesher.generate(
        geometry
    )

    A, F, topology = (
        assemble_maxwell_system(
            mesh,
            omega=OMEGA,
            epsilon=EPSILON,
            mu=MU,
            source_function=source_function,
        )
    )

    A_bc, F_bc = (
        apply_pec_boundary(
            A,
            F,
            topology,
        )
    )

    solution = (
        solve_linear_system(
            A_bc,
            F_bc,
        )
    )

    field_error = (
        calculate_field_l2_error(
            mesh,
            topology,
            solution,
        )
    )

    curl_error = (
        calculate_curl_l2_error(
            mesh,
            topology,
            solution,
        )
    )

    return (
        mesh,
        topology,
        solution,
        field_error,
        curl_error,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n"
        "====================================================\n"
        "NÉDÉLEC H(CURL) MAXWELL VERIFICATION\n"
        "====================================================\n"
    )

    print(
        "Equation:\n"
        "\n"
        "    curl(mu^-1 curl(E))\n"
        "    - omega² epsilon E\n"
        "    = J\n"
    )

    print(
        f"mu       = {MU}"
    )

    print(
        f"epsilon  = {EPSILON}"
    )

    print(
        f"omega    = {OMEGA}"
    )

    print(
        "\n"
        "Exact field:\n"
        "\n"
        "    E = [ sin(πx) sin(πy), 0 ]\n"
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

    for h in mesh_sizes:

        print(
            f"\nSolving h = "
            f"{h:.6f}"
        )

        (
            mesh,
            topology,
            solution,
            field_error,
            curl_error,
        ) = solve_mesh(
            geometry,
            h,
        )

        print(
            f"    Nodes          : "
            f"{mesh.node_count}"
        )

        print(
            f"    Triangles      : "
            f"{mesh.element_count}"
        )

        print(
            f"    Global edges   : "
            f"{topology.edge_count}"
        )

        print(
            f"    Boundary edges : "
            f"{len(topology.boundary_edges)}"
        )

        print(
            f"    Field L2 error : "
            f"{field_error:.8e}"
        )

        print(
            f"    Curl L2 error  : "
            f"{curl_error:.8e}"
        )

        results.append(
            {
                "h": h,
                "nodes": mesh.node_count,
                "edges": topology.edge_count,
                "field_error": field_error,
                "curl_error": curl_error,
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
        "NÉDELEC MAXWELL CONVERGENCE"
    )

    print(
        "===================================================="
    )

    print(
        f"{'h':>10}"
        f"{'Nodes':>10}"
        f"{'Edges':>10}"
        f"{'Field L2':>18}"
        f"{'Order':>12}"
        f"{'Curl L2':>18}"
        f"{'Order':>12}"
    )

    previous_h = None
    previous_field = None
    previous_curl = None

    for result in results:

        h = result["h"]

        field_error = (
            result["field_error"]
        )

        curl_error = (
            result["curl_error"]
        )

        if previous_h is None:

            field_order = "---"
            curl_order = "---"

        else:

            field_order = (
                np.log(
                    previous_field
                    / field_error
                )
                /
                np.log(
                    previous_h
                    / h
                )
            )

            curl_order = (
                np.log(
                    previous_curl
                    / curl_error
                )
                /
                np.log(
                    previous_h
                    / h
                )
            )

            field_order = (
                f"{field_order:.6f}"
            )

            curl_order = (
                f"{curl_order:.6f}"
            )

        print(
            f"{h:10.6f}"
            f"{result['nodes']:10d}"
            f"{result['edges']:10d}"
            f"{field_error:18.8e}"
            f"{field_order:>12}"
            f"{curl_error:18.8e}"
            f"{curl_order:>12}"
        )

        previous_h = h
        previous_field = field_error
        previous_curl = curl_error

    # ========================================================
    # MONOTONICITY
    # ========================================================

    field_errors = np.array(
        [
            result["field_error"]
            for result in results
        ]
    )

    curl_errors = np.array(
        [
            result["curl_error"]
            for result in results
        ]
    )

    field_pass = np.all(
        np.diff(
            field_errors
        ) < 0
    )

    curl_pass = np.all(
        np.diff(
            curl_errors
        ) < 0
    )

    print(
        "\n"
        "Field L2 error decreasing : "
        f"{'PASS' if field_pass else 'FAIL'}"
    )

    print(
        "Curl L2 error decreasing  : "
        f"{'PASS' if curl_pass else 'FAIL'}"
    )

    if not field_pass:

        raise RuntimeError(
            "Nedelec Maxwell verification FAILED: "
            "field error is not monotonically convergent."
        )

    if not curl_pass:

        raise RuntimeError(
            "Nedelec Maxwell verification FAILED: "
            "curl error is not monotonically convergent."
        )

    print(
        "\n"
        "===================================================="
    )

    print(
        "NÉDELEC H(CURL) MAXWELL VERIFICATION : PASS"
    )

    print(
        "===================================================="
    )


if __name__ == "__main__":

    main()