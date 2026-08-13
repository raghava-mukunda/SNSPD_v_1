# FILE: src/snspd/fem/electromagnetics/topology.py
# PURPOSE:
# Constructs the global oriented edge topology required by
# Nedelec H(curl)-conforming finite elements.
#
# A first-order Nedelec triangle has one degree of freedom per edge:
#
#     DOF_e = integral_e E · dl
#
# A global orientation is assigned to every mesh edge.
#
# We use the deterministic convention:
#
#     lower global node index -> higher global node index
#
# Local element edges are:
#
#     (0,1)
#     (1,2)
#     (2,0)
#
# If the local orientation disagrees with the global orientation,
# the local Nedelec basis function receives a sign of -1.
#
# This orientation handling is mathematically essential because
# reversing an edge orientation reverses its circulation DOF.


from __future__ import annotations


from dataclasses import dataclass


import numpy as np


@dataclass
class EdgeTopology:
    """
    Global edge topology for a triangular mesh.
    """

    edges: np.ndarray

    triangle_edges: np.ndarray

    triangle_edge_signs: np.ndarray

    boundary_edges: np.ndarray

    @property
    def edge_count(self) -> int:
        return int(
            len(self.edges)
        )


def build_edge_topology(mesh):
    """
    Construct the global edge topology.

    Parameters
    ----------
    mesh:
        Canonical triangular FEM mesh.

    Returns
    -------
    EdgeTopology
        Global edge list, element-edge connectivity,
        orientation signs, and boundary edges.
    """

    triangles = np.asarray(
        mesh.triangles,
        dtype=int,
    )

    edge_map = {}

    edge_list = []

    triangle_edges = np.zeros(
        (
            len(triangles),
            3,
        ),
        dtype=int,
    )

    triangle_signs = np.ones(
        (
            len(triangles),
            3,
        ),
        dtype=int,
    )

    # --------------------------------------------------------
    # Local edge ordering
    #
    # edge 0: vertex 0 -> vertex 1
    # edge 1: vertex 1 -> vertex 2
    # edge 2: vertex 2 -> vertex 0
    # --------------------------------------------------------

    local_edges = (
        (0, 1),
        (1, 2),
        (2, 0),
    )

    for element_index, triangle in enumerate(
        triangles
    ):

        for local_edge_index, (
            local_i,
            local_j,
        ) in enumerate(local_edges):

            node_i = int(
                triangle[local_i]
            )

            node_j = int(
                triangle[local_j]
            )

            # ------------------------------------------------
            # Global orientation:
            #
            # min(node_i,node_j)
            # ->
            # max(node_i,node_j)
            # ------------------------------------------------

            global_edge = (
                min(node_i, node_j),
                max(node_i, node_j),
            )

            if global_edge not in edge_map:

                edge_map[
                    global_edge
                ] = len(edge_list)

                edge_list.append(
                    global_edge
                )

            edge_index = edge_map[
                global_edge
            ]

            triangle_edges[
                element_index,
                local_edge_index,
            ] = edge_index

            # ------------------------------------------------
            # Orientation sign.
            #
            # +1:
            #
            # local orientation =
            # global orientation
            #
            # -1:
            #
            # local orientation =
            # reverse(global orientation)
            # ------------------------------------------------

            if (
                node_i,
                node_j
            ) == global_edge:

                triangle_signs[
                    element_index,
                    local_edge_index,
                ] = 1

            else:

                triangle_signs[
                    element_index,
                    local_edge_index,
                ] = -1

    edges = np.asarray(
        edge_list,
        dtype=int,
    )

    # ========================================================
    # FIND BOUNDARY EDGES
    # ========================================================
    #
    # A mesh edge is on the external boundary if it belongs
    # to exactly one triangle.
    # ========================================================

    edge_usage = np.zeros(
        len(edges),
        dtype=int,
    )

    for element_edges in (
        triangle_edges
    ):

        for edge_index in (
            element_edges
        ):

            edge_usage[
                edge_index
            ] += 1

    boundary_edges = np.flatnonzero(
        edge_usage == 1
    )

    return EdgeTopology(
        edges=edges,
        triangle_edges=triangle_edges,
        triangle_edge_signs=triangle_signs,
        boundary_edges=boundary_edges,
    )