# FILE: src/snspd/fem/electromagnetics/maxwell.py
# PURPOSE:
# Assembles the two-dimensional frequency-domain Maxwell
# curl-curl formulation using lowest-order Nedelec edge elements.
#
# Governing equation:
#
#     curl(mu^{-1} curl(E))
#     -
#     omega² epsilon E
#     =
#     J
#
# Weak form:
#
#     integral:
#
#         mu^{-1}
#         curl(E) curl(v)
#
#         -
#
#         omega² epsilon
#         E · v
#
#     dA
#
# The resulting global matrix is:
#
#     A = K_curl - omega² M_epsilon
#
# The electric field degrees of freedom are edge circulations.


from __future__ import annotations


import numpy as np


from scipy.sparse import (
    lil_matrix,
)


from snspd.fem.electromagnetics.nedelec import (
    NedelecTriangle,
)


from snspd.fem.electromagnetics.topology import (
    EdgeTopology,
    build_edge_topology,
)


def assemble_maxwell_system(
    mesh,
    omega,
    epsilon,
    mu,
    source_function=None,
):
    """
    Assemble:

        curl(mu^-1 curl(E))
        -
        omega² epsilon E
        =
        J

    Parameters
    ----------
    mesh:
        Canonical triangular FEM mesh.

    omega:
        Angular frequency [rad/s].

    epsilon:
        Electric permittivity.

    mu:
        Magnetic permeability.

    source_function:
        Callable:

            J(x,y)

        returning a two-component vector.

    Returns
    -------
    A:
        Global complex Maxwell matrix.

    F:
        Global complex source vector.

    topology:
        Global edge topology.
    """

    if omega < 0:

        raise ValueError(
            "Angular frequency must be non-negative."
        )

    if epsilon == 0:

        raise ValueError(
            "Permittivity must be non-zero."
        )

    if mu == 0:

        raise ValueError(
            "Permeability must be non-zero."
        )

    topology = (
        build_edge_topology(
            mesh
        )
    )

    edge_count = (
        topology.edge_count
    )

    A = lil_matrix(
        (
            edge_count,
            edge_count,
        ),
        dtype=complex,
    )

    F = np.zeros(
        edge_count,
        dtype=complex,
    )

    # ========================================================
    # ELEMENT ASSEMBLY
    # ========================================================

    for element_index in range(
        mesh.element_count
    ):

        node_indices = (
            mesh.triangles[
                element_index
            ]
        )

        coordinates = (
            mesh.nodes[
                node_indices
            ]
        )

        element = NedelecTriangle(
            coordinates
        )

        # ----------------------------------------------------
        # Local curl-curl operator
        # ----------------------------------------------------

        K_local = (
            element.curl_curl_matrix(
                inverse_mu=1.0 / mu
            )
        )

        # ----------------------------------------------------
        # Local electric-field mass operator
        # ----------------------------------------------------

        M_local = (
            element.mass_matrix(
                coefficient=epsilon
            )
        )

        # ----------------------------------------------------
        # Frequency-domain Maxwell matrix
        # ----------------------------------------------------

        A_local = (
            K_local
            -
            omega**2
            * M_local
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

        # ====================================================
        # MATRIX ASSEMBLY
        # ====================================================

        for local_i in range(3):

            global_i = int(
                local_edges[
                    local_i
                ]
            )

            sign_i = (
                signs[
                    local_i
                ]
            )

            for local_j in range(3):

                global_j = int(
                    local_edges[
                        local_j
                    ]
                )

                sign_j = (
                    signs[
                        local_j
                    ]
                )

                A[
                    global_i,
                    global_j,
                ] += (
                    sign_i
                    * sign_j
                    * A_local[
                        local_i,
                        local_j
                    ]
                )

        # ====================================================
        # SOURCE ASSEMBLY
        # ====================================================

        if source_function is not None:

            # ------------------------------------------------
            # Three-point quadrature.
            # ------------------------------------------------

            barycentric_points = np.array(
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

            for barycentric in (
                barycentric_points
            ):

                point = (
                    barycentric
                    @ coordinates
                )

                source = np.asarray(
                    source_function(
                        point[0],
                        point[1],
                    ),
                    dtype=complex,
                )

                if source.shape != (2,):

                    raise ValueError(
                        "Maxwell source must return "
                        "a two-component vector."
                    )

                for local_i in range(3):

                    basis = (
                        element.basis_function(
                            local_i,
                            point,
                        )
                    )

                    local_source = (
                        np.vdot(
                            basis,
                            source,
                        )
                    )

                    global_i = int(
                        local_edges[
                            local_i
                        ]
                    )

                    sign_i = (
                        signs[
                            local_i
                        ]
                    )

                    F[global_i] += (
                        sign_i
                        * element.area
                        / 3.0
                        * local_source
                    )

    return (
        A.tocsr(),
        F,
        topology,
    )