# FILE: src/snspd/fem/electromagnetics/boundary.py
# PURPOSE:
# Applies electromagnetic boundary conditions to the Nedelec
# edge-element system.
#
# For a Perfect Electric Conductor (PEC):
#
#     n × E = 0
#
# which is equivalent to zero tangential electric field.
#
# Since the Nedelec degrees of freedom are edge circulations,
# PEC is imposed directly on boundary-edge DOFs.


from __future__ import annotations


import numpy as np


def apply_pec_boundary(
    A,
    F,
    topology,
):
    """
    Apply homogeneous PEC boundary conditions.

    Boundary-edge electric-field DOFs are constrained to zero.

    Parameters
    ----------
    A:
        Global complex Maxwell matrix.

    F:
        Global complex RHS.

    topology:
        EdgeTopology object.

    Returns
    -------
    A_bc:
        Boundary-conditioned matrix.

    F_bc:
        Boundary-conditioned RHS.
    """

    A_bc = A.tolil(
        copy=True
    )

    F_bc = np.asarray(
        F
    ).copy()

    boundary_set = set(
        int(edge)
        for edge in topology.boundary_edges
    )

    # ========================================================
    # ZERO COLUMNS
    # ========================================================

    for row in range(
        A_bc.shape[0]
    ):

        if row in boundary_set:

            continue

        columns = (
            A_bc.rows[row]
        )

        values = (
            A_bc.data[row]
        )

        for index in range(
            len(columns)
        ):

            if (
                columns[index]
                in boundary_set
            ):

                values[index] = 0.0

    # ========================================================
    # REPLACE BOUNDARY ROWS WITH IDENTITY
    # ========================================================

    for edge in topology.boundary_edges:

        edge = int(edge)

        A_bc.rows[edge] = [
            edge
        ]

        A_bc.data[edge] = [
            1.0
        ]

        F_bc[edge] = 0.0

    return (
        A_bc.tocsr(),
        F_bc,
    )