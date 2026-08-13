# FILE: src/snspd/fem/helmholtz.py
# PURPOSE:
# Assembles the scalar frequency-domain Helmholtz FEM system:
#
#     -∇²u - k²u = f
#
# The weak-form matrix is:
#
#     K - k² M
#
# where:
#
#     K = stiffness matrix
#     M = mass matrix
#
# The implementation supports complex-valued fields and sources.
#
# This is a numerical verification layer preceding the full vector
# Maxwell FEM formulation.


from __future__ import annotations


import numpy as np


from scipy.sparse import (
    lil_matrix,
)


from snspd.mesh.mesh import Mesh


from snspd.fem.elements import (
    TriangleElement,
)


def assemble_helmholtz_system(
    mesh: Mesh,
    source_function,
    wave_number: float,
):
    """
    Assemble:

        -∇²u - k²u = f

    Parameters
    ----------
    mesh:
        Canonical FEM mesh.

    source_function:
        Complex-capable callable:

            f(x,y)

    wave_number:
        Scalar wavenumber k.

    Returns
    -------
    A:
        Sparse complex FEM system matrix.

    F:
        Complex FEM load vector.
    """

    if wave_number < 0:

        raise ValueError(
            "Wave number must be non-negative."
        )

    node_count = (
        mesh.node_count
    )

    # --------------------------------------------------------
    # The Helmholtz system may be complex.
    # --------------------------------------------------------

    A = lil_matrix(
        (
            node_count,
            node_count,
        ),
        dtype=complex,
    )

    F = np.zeros(
        node_count,
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

        element = TriangleElement(
            coordinates
        )

        # ----------------------------------------------------
        # Weak form:
        #
        #     ∫ ∇Nᵀ∇u dΩ
        #     -
        #     k² ∫ Nᵀu dΩ
        #
        # Therefore:
        #
        #     Ae = Ke - k² Me
        # ----------------------------------------------------

        Ke = (
            element.stiffness_matrix()
        )

        Me = (
            element.mass_matrix()
        )

        Ae = (
            Ke
            -
            wave_number**2
            * Me
        )

        Fe = (
            element.load_vector(
                source_function
            )
        )

        # ----------------------------------------------------
        # Global assembly
        # ----------------------------------------------------

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

                A[
                    global_i,
                    global_j,
                ] += (
                    Ae[
                        local_i,
                        local_j
                    ]
                )

    return (
        A.tocsr(),
        F,
    )