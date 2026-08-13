# FILE: src/snspd/fem/electrical/terminal.py
# PURPOSE:
# Provides FEM boundary-edge analysis for SNSPD electrical terminals.
#
# The terminal current is calculated from the continuum definition:
#
#     I = t * integral_Gamma (J . n) ds
#
# where:
#
#     J = current density [A/m^2]
#     n = outward unit normal
#     t = film thickness [m]
#     Gamma = electrical terminal boundary
#
# This module does NOT assume:
#
#     I = J * w * t
#
# and does NOT use the overall 2D footprint area.
#
# The purpose is to obtain terminal current directly from the
# FEM current-density field and the actual terminal boundary.
#
# Boundary edges are reconstructed from the triangular mesh.
#
# A boundary edge is an edge belonging to exactly one triangle.
#
# The outward normal is determined geometrically from the
# orientation of the edge relative to the centroid of its
# adjacent triangle.
#
# Current density is assumed piecewise constant over each P1
# triangular element because it is obtained from the gradient
# of the linear FEM potential.


from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# ============================================================
# DATA STRUCTURES
# ============================================================


@dataclass(frozen=True)
class BoundaryEdge:
    """
    Represents one boundary edge of the FEM mesh.

    Parameters
    ----------
    node_a:
        First mesh-node index.

    node_b:
        Second mesh-node index.

    element:
        Index of the unique adjacent triangle.

    length_m:
        Edge length [m].

    midpoint:
        Edge midpoint [m].

    normal:
        Outward unit normal.

    """

    node_a: int

    node_b: int

    element: int

    length_m: float

    midpoint: np.ndarray

    normal: np.ndarray


@dataclass(frozen=True)
class TerminalBoundary:
    """
    Collection of FEM boundary edges belonging to one terminal.

    Parameters
    ----------
    name:
        Terminal name.

    edges:
        Boundary edges belonging to this terminal.

    """

    name: str

    edges: tuple[BoundaryEdge, ...]

    @property
    def edge_count(self) -> int:
        """
        Number of boundary edges.
        """

        return len(self.edges)

    @property
    def length_m(self) -> float:
        """
        Total terminal boundary length [m].
        """

        return float(
            sum(
                edge.length_m
                for edge in self.edges
            )
        )


@dataclass(frozen=True)
class TerminalCurrentResult:
    """
    Result of terminal-current integration.

    Parameters
    ----------
    signed_current_A:
        Signed current obtained from:

            t * integral(J.n ds)

    current_A:
        Absolute transported current [A].

    average_normal_current_density_A_m2:
        Magnitude of terminal-average normal current density:

            |I| / (t * L_terminal)

    flux_A_m:
        Integrated J.n ds before multiplying by thickness.
    """

    signed_current_A: float

    current_A: float

    average_normal_current_density_A_m2: float

    flux_A_m: float


# ============================================================
# BOUNDARY EDGE CONSTRUCTION
# ============================================================


def _edge_key(
    node_a: int,
    node_b: int,
) -> tuple[int, int]:
    """
    Return an orientation-independent edge key.
    """

    if node_a < node_b:

        return (
            node_a,
            node_b,
        )

    return (
        node_b,
        node_a,
    )


def _build_edge_adjacency(
    triangles: np.ndarray,
) -> dict[
    tuple[int, int],
    list[tuple[int, int, int]],
]:
    """
    Construct edge-to-element adjacency.

    Returns
    -------
    dict
        Maps:

            (node_a, node_b)

        to a list of:

            (element_index, local_node_a, local_node_b)

    Boundary edges occur when the list has exactly one entry.
    """

    adjacency = {}

    for element_index, triangle in enumerate(
        triangles
    ):

        n0, n1, n2 = (
            int(value)
            for value in triangle
        )

        local_edges = (
            (n0, n1),
            (n1, n2),
            (n2, n0),
        )

        for local_a, local_b in local_edges:

            key = _edge_key(
                local_a,
                local_b,
            )

            if key not in adjacency:

                adjacency[key] = []

            adjacency[key].append(
                (
                    element_index,
                    local_a,
                    local_b,
                )
            )

    return adjacency


# ============================================================
# OUTWARD NORMAL
# ============================================================


def _outward_normal_for_edge(
    nodes: np.ndarray,
    node_a: int,
    node_b: int,
    element: int,
    triangles: np.ndarray,
) -> np.ndarray:
    """
    Determine the outward normal of a boundary edge.

    The edge has two possible unit normals.

    We determine which one points away from the centroid of
    the adjacent triangle.

    This works independently of the orientation of the triangle
    node ordering.
    """

    point_a = (
        nodes[node_a]
    )

    point_b = (
        nodes[node_b]
    )

    edge_vector = (
        point_b
        - point_a
    )

    length = float(
        np.linalg.norm(
            edge_vector
        )
    )

    if length <= 0.0:

        raise ValueError(
            "Boundary edge has zero length."
        )

    # Rotate edge vector by +90 degrees.
    normal_a = np.array(
        [
            -edge_vector[1],
            edge_vector[0],
        ],
        dtype=float,
    )

    normal_a /= (
        np.linalg.norm(
            normal_a
        )
    )

    normal_b = (
        -normal_a
    )

    triangle_nodes = (
        triangles[element]
    )

    centroid = np.mean(
        nodes[
            triangle_nodes
        ],
        axis=0,
    )

    midpoint = (
        0.5
        * (
            point_a
            + point_b
        )
    )

    # Vector from edge midpoint into the element.
    inward_vector = (
        centroid
        - midpoint
    )

    # The outward normal must point opposite to the vector
    # pointing from the boundary into the element.
    if (
        np.dot(
            normal_a,
            inward_vector,
        )
        < 0.0
    ):

        return normal_a

    return normal_b


def build_boundary_edges(
    nodes: np.ndarray,
    triangles: np.ndarray,
) -> tuple[BoundaryEdge, ...]:
    """
    Construct all external FEM boundary edges.

    Returns
    -------
    tuple[BoundaryEdge, ...]
        All edges belonging to exactly one triangle.
    """

    nodes = np.asarray(
        nodes,
        dtype=float,
    )

    triangles = np.asarray(
        triangles,
        dtype=int,
    )

    adjacency = (
        _build_edge_adjacency(
            triangles
        )
    )

    boundary_edges = []

    for key, entries in adjacency.items():

        if len(entries) != 1:

            continue

        element, node_a, node_b = (
            entries[0]
        )

        point_a = (
            nodes[node_a]
        )

        point_b = (
            nodes[node_b]
        )

        edge_vector = (
            point_b
            - point_a
        )

        length = float(
            np.linalg.norm(
                edge_vector
            )
        )

        if length <= 0.0:

            raise ValueError(
                "Mesh contains a zero-length boundary edge."
            )

        midpoint = (
            0.5
            * (
                point_a
                + point_b
            )
        )

        normal = (
            _outward_normal_for_edge(
                nodes=nodes,
                node_a=node_a,
                node_b=node_b,
                element=element,
                triangles=triangles,
            )
        )

        boundary_edges.append(
            BoundaryEdge(
                node_a=node_a,
                node_b=node_b,
                element=element,
                length_m=length,
                midpoint=midpoint,
                normal=normal,
            )
        )

    return tuple(
        boundary_edges
    )


# ============================================================
# TERMINAL SELECTION
# ============================================================


def select_terminal_boundary(
    boundary_edges: tuple[BoundaryEdge, ...],
    *,
    axis: int,
    coordinate: float,
    tolerance: float,
    name: str,
) -> TerminalBoundary:
    """
    Select boundary edges whose midpoint lies on a terminal
    coordinate.

    Parameters
    ----------
    boundary_edges:
        Complete external FEM boundary.

    axis:
        Coordinate axis:

            0 -> x
            1 -> y

    coordinate:
        Terminal coordinate [m].

    tolerance:
        Positional tolerance [m].

    name:
        Terminal name.

    Notes
    -----
    This function still uses the present geometry's known
    terminal locations.

    The important distinction is that the terminal is now
    represented as an actual set of FEM boundary edges rather
    than merely a set of boundary nodes.

    Later, explicit contact regions from GDS/SVG metadata can
    replace coordinate-based selection without changing the
    current-integration machinery.
    """

    if axis not in (
        0,
        1,
    ):

        raise ValueError(
            "Terminal axis must be 0 or 1."
        )

    selected = []

    for edge in boundary_edges:

        if (
            abs(
                edge.midpoint[axis]
                - coordinate
            )
            <= tolerance
        ):

            selected.append(
                edge
            )

    if not selected:

        raise RuntimeError(
            f"No FEM boundary edges found for "
            f"{name} terminal at coordinate "
            f"{coordinate:.6e} m."
        )

    return TerminalBoundary(
        name=name,
        edges=tuple(
            selected
        ),
    )


# ============================================================
# TERMINAL CURRENT INTEGRATION
# ============================================================


def integrate_terminal_current(
    terminal: TerminalBoundary,
    element_current_density: np.ndarray,
    film_thickness: float,
) -> TerminalCurrentResult:
    """
    Integrate current density over a terminal boundary.

    Fundamental equation:

        I = t * integral_Gamma (J . n) ds

    Since J is constant inside each linear triangular element:

        integral_edge (J . n) ds

        =
        (J_element . n_edge) * L_edge

    Therefore:

        I =
        t * sum[
            (J_element . n_edge) L_edge
        ]

    Parameters
    ----------
    terminal:
        TerminalBoundary.

    element_current_density:
        Array with shape:

            (number_of_elements, 2)

        containing J [A/m^2].

    film_thickness:
        Film thickness [m].

    Returns
    -------
    TerminalCurrentResult
    """

    J = np.asarray(
        element_current_density,
        dtype=float,
    )

    if J.ndim != 2:

        raise ValueError(
            "element_current_density must be a 2D array."
        )

    if J.shape[1] != 2:

        raise ValueError(
            "element_current_density must have shape (N, 2)."
        )

    if film_thickness <= 0.0:

        raise ValueError(
            "Film thickness must be positive."
        )

    flux = 0.0

    for edge in terminal.edges:

        if (
            edge.element < 0
            or edge.element >= len(J)
        ):

            raise IndexError(
                "Boundary edge references an invalid FEM element."
            )

        J_element = (
            J[edge.element]
        )

        normal_component = float(
            np.dot(
                J_element,
                edge.normal,
            )
        )

        flux += (
            normal_component
            * edge.length_m
        )

    signed_current = (
        film_thickness
        * flux
    )

    current = abs(
        signed_current
    )

    terminal_area = (
        terminal.length_m
        * film_thickness
    )

    if terminal_area <= 0.0:

        raise RuntimeError(
            "Terminal cross-sectional area is zero."
        )

    average_normal_J = (
        current
        / terminal_area
    )

    return TerminalCurrentResult(
        signed_current_A=float(
            signed_current
        ),
        current_A=float(
            current
        ),
        average_normal_current_density_A_m2=float(
            average_normal_J
        ),
        flux_A_m=float(
            flux
        ),
    )


# ============================================================
# TERMINAL REPORT
# ============================================================


def format_terminal_report(
    terminal: TerminalBoundary,
    result: TerminalCurrentResult,
) -> str:
    """
    Generate a human-readable terminal report.
    """

    return (
        "\n"
        f"{terminal.name} TERMINAL\n"
        f"{'-' * (len(terminal.name) + 9)}\n"
        f"Boundary edges             : "
        f"{terminal.edge_count}\n"
        f"Boundary length            : "
        f"{terminal.length_m * 1e6:.9f} um\n"
        f"Signed current              : "
        f"{result.signed_current_A:.9e} A\n"
        f"Current magnitude           : "
        f"{result.current_A:.9e} A\n"
        f"Average normal |J|          : "
        f"{result.average_normal_current_density_A_m2:.9e} A/m²\n"
    )


# ============================================================
# TERMINAL CURRENT CONSISTENCY
# ============================================================


def terminal_current_relative_difference(
    current_a: float,
    current_b: float,
) -> float:
    """
    Calculate relative disagreement between two terminal
    current magnitudes.

    For a charge-conserving stationary solution:

        I_positive = I_negative

    in magnitude.

    The result should therefore approach zero as the FEM
    solution and terminal integration become accurate.
    """

    scale = max(
        abs(current_a),
        abs(current_b),
        1.0e-30,
    )

    return abs(
        current_a
        - current_b
    ) / scale