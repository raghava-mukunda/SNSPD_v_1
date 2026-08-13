# FILE: src/snspd/fem/boundary.py
# PURPOSE:
# Provides Dirichlet boundary-condition handling for FEM systems.
#
# Supports both real-valued and complex-valued FEM matrices.
#
# For prescribed:
#
#     u_i = g_i
#
# the corresponding degrees of freedom are constrained.


from __future__ import annotations


import numpy as np


def find_boundary_nodes(
    nodes: np.ndarray,
    tolerance: float = 1e-12,
) -> np.ndarray:
    """
    Find nodes lying on the outer rectangular boundary.

    This routine is intended for analytical verification domains.
    Geometry-aware boundary tagging will replace it later.
    """

    nodes = np.asarray(
        nodes,
        dtype=float,
    )

    xmin = np.min(
        nodes[:, 0]
    )

    xmax = np.max(
        nodes[:, 0]
    )

    ymin = np.min(
        nodes[:, 1]
    )

    ymax = np.max(
        nodes[:, 1]
    )

    on_left = np.isclose(
        nodes[:, 0],
        xmin,
        atol=tolerance,
    )

    on_right = np.isclose(
        nodes[:, 0],
        xmax,
        atol=tolerance,
    )

    on_bottom = np.isclose(
        nodes[:, 1],
        ymin,
        atol=tolerance,
    )

    on_top = np.isclose(
        nodes[:, 1],
        ymax,
        atol=tolerance,
    )

    boundary = (
        on_left
        | on_right
        | on_bottom
        | on_top
    )

    return np.flatnonzero(
        boundary
    )


def apply_dirichlet_zero(
    K,
    F,
    boundary_nodes: np.ndarray,
):
    """
    Apply:

        u = 0

    on specified nodes.

    Works for real and complex FEM systems.
    """

    K_bc = K.tolil(
        copy=True
    )

    F_bc = np.asarray(
        F
    ).copy()

    boundary_set = set(
        int(node)
        for node in boundary_nodes
    )

    # ========================================================
    # ZERO COLUMNS FIRST
    # ========================================================

    for row in range(
        K_bc.shape[0]
    ):

        if row in boundary_set:

            continue

        columns = (
            K_bc.rows[row]
        )

        values = (
            K_bc.data[row]
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

    for node in boundary_nodes:

        node = int(node)

        K_bc.rows[node] = [
            node
        ]

        K_bc.data[node] = [
            1.0
        ]

        F_bc[node] = 0.0

    return (
        K_bc.tocsr(),
        F_bc,
    )