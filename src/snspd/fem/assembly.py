# FILE: src/snspd/fem/assembly.py
# PURPOSE:
# Assembles element-level FEM matrices into the global system.
#
# The global problem has the form:
#
#     K u = F
#
# where K is the assembled stiffness matrix and F is the assembled
# load vector.
#
# This module operates entirely on the canonical Mesh representation.


from __future__ import annotations


import numpy as np


from scipy.sparse import (
    lil_matrix,
)


from snspd.mesh.mesh import Mesh


def assemble_poisson_system(
    mesh: Mesh,
    source_function,
    coefficient: float = 1.0,
):
    """
    Assemble the global FEM system for:

        -∇ · (k ∇u) = f

    Parameters
    ----------
    mesh:
        Canonical triangular FEM mesh.

    source_function:
        Callable:

            f(x,y)

    coefficient:
        Scalar PDE coefficient k.

    Returns
    -------
    K:
        Global sparse stiffness matrix.

    F:
        Global load vector.
    """

    if mesh.element_count == 0:

        raise ValueError(
            "Cannot assemble FEM system "
            "from an empty mesh."
        )

    node_count = (
        mesh.node_count
    )

    K = lil_matrix(
        (
            node_count,
            node_count,
        ),
        dtype=float,
    )

    F = np.zeros(
        node_count,
        dtype=float,
    )

    # =========================================================
    # ELEMENT LOOP
    # =========================================================

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

        # -----------------------------------------------------
        # Local element mathematics
        # -----------------------------------------------------

        from snspd.fem.elements import (
            TriangleElement,
        )

        element = TriangleElement(
            coordinates
        )

        Ke = (
            element.stiffness_matrix(
                coefficient=coefficient
            )
        )

        Fe = (
            element.load_vector(
                source_function
            )
        )

        # =====================================================
        # GLOBAL ASSEMBLY
        # =====================================================

        for local_i in range(3):

            global_i = int(
                node_indices[
                    local_i
                ]
            )

            F[global_i] += (
                Fe[local_i]
            )

            for local_j in range(3):

                global_j = int(
                    node_indices[
                        local_j
                    ]
                )

                K[
                    global_i,
                    global_j,
                ] += (
                    Ke[
                        local_i,
                        local_j
                    ]
                )

    return K.tocsr(), F