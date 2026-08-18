# FILE:
# examples/electrical/analyze_current_crowding.py
#
# PURPOSE:
# Perform a stationary electrical FEM analysis of an SNSPD
# nanowire and calculate the spatial current-density distribution
# and current-crowding factor.
#
# Physical formulation:
#
#     div(sigma * grad(V)) = 0
#
#     E = -grad(V)
#
#     J = sigma E
#       = -sigma grad(V)
#
# Terminal current:
#
#     I = t * integral_Gamma (J . n) ds
#
# In the discrete FEM system, the terminal transport current
# is obtained from the Dirichlet reaction vector.
#
# Current-crowding factor:
#
#     C_J = J_max / J_avg,terminal
#
# where:
#
#     J_avg,terminal = I / (t * L_terminal)
#
# IMPORTANT:
#
# This is still a NORMAL-STATE electrical conductivity model.
#
# It does NOT yet calculate:
#
#     - superconducting critical current
#     - depairing current
#     - vortex-entry current
#     - kinetic-inductance effects
#     - electrothermal switching
#
# Those will be coupled later.
#
# The purpose of this stage is to obtain a validated spatial
# current-density field J(x,y) from arbitrary SNSPD geometry.


from __future__ import annotations


# ============================================================
# IMPORTS
# ============================================================

from dataclasses import dataclass
from pathlib import Path
import argparse
import sys

import matplotlib.pyplot as plt
import numpy as np

from shapely.geometry import Point
from shapely.ops import unary_union
from skimage.draw import polygon as raster_polygon
from skimage.morphology import skeletonize
from skimage.measure import label

from snspd.geometry.svg_importer import (
    import_svg,
)

from snspd.geometry.analyzer import (
    analyze_geometry,
    format_metrics,
)

from snspd.mesh.gmsh_mesher import (
    GmshMesher,
)

from snspd.mesh.quality import (
    analyze_mesh_quality,
    format_mesh_quality,
)

from snspd.fem.electrical.current_distribution import (
    CurrentDistributionSolver,
)

from snspd.fem.electrical.terminal import (
    build_boundary_edges,
)


# ============================================================
# PHYSICAL PARAMETERS
# ============================================================

# Film thickness.
#
# This is currently an assumed value.
#
# Change this when the actual fabricated film thickness
# is known.
FILM_THICKNESS = 5.0e-9


# Normal-state resistivity.
#
# This is a placeholder material parameter for the current
# electrical verification stage.
NORMAL_RESISTIVITY = 1.0e-6


# Conductivity:
#
#     sigma = 1 / rho
#
CONDUCTIVITY = (
    1.0
    / NORMAL_RESISTIVITY
)


# Voltage used for the linear stationary FEM solve.
#
# Since the governing equation is linear in the normal-state
# model, the resulting current density can subsequently be
# scaled linearly to any desired bias current.
SOLVE_VOLTAGE = 1.0


# FEM mesh characteristic length.
MESH_SIZE = 0.25e-6


# Desired physical bias current for reporting.
TARGET_BIAS_CURRENT = 10.0e-6


# ============================================================
# AUTOMATIC ELECTRICAL TERMINAL DETECTION
# ============================================================

# The two electrical terminals are detected automatically from the
# physical nanowire geometry. No xmin/xmax assumption is made.
#
# Method:
#   1. Union all geometry regions into the physical nanowire.
#   2. Rasterize the geometry at an adaptive resolution.
#   3. Skeletonize the nanowire.
#   4. Find the two end points of the skeleton.
#   5. Map each skeleton end point back to the nearest physical
#      FEM boundary end-cap.
#
# This is intended for arbitrary serpentine/meander geometries.

AUTO_TERMINAL_MAX_PIXELS = 2200
AUTO_TERMINAL_MIN_PIXELS = 300
AUTO_TERMINAL_PIXELS_PER_WIDTH = 6
AUTO_TERMINAL_BOUNDARY_RADIUS_FACTOR = 2.5
AUTO_TERMINAL_MIN_CAP_EDGES = 2

# ============================================================
# VISUALIZATION PARAMETERS
# ============================================================

SHOW_HEATMAP = True

SAVE_HEATMAP = True

RESULTS_DIRECTORY = (
    Path(__file__).resolve().parents[2]
    / "results"
)

HEATMAP_FILENAME = (
    "current_density_heatmap.png"
)


# ============================================================
# TERMINAL DATA STRUCTURE
# ============================================================

@dataclass
class PhysicalTerminal:
    """
    Physical electrical terminal represented by a subset of
    the exterior FEM boundary.

    Parameters
    ----------
    name:
        Human-readable terminal name.

    requested_point:
        User-specified physical coordinate (x, y) in metres.

    edge_indices:
        Indices into the global boundary-edge array.

    node_indices:
        Unique FEM node indices belonging to this terminal.

    length_m:
        Physical terminal boundary length in metres.
    """

    name: str

    requested_point: np.ndarray

    edge_indices: np.ndarray

    node_indices: np.ndarray

    length_m: float

    @property
    def edge_count(self) -> int:
        return len(self.edge_indices)

    @property
    def node_count(self) -> int:
        return len(self.node_indices)


# ============================================================
# GEOMETRY
# ============================================================

def parse_arguments():
    """
    Parse command-line arguments.

    Width and thickness are explicit simulator inputs. They are
    intentionally NOT read from SVG metadata.

    Example:
        python analyze_current_crowding.py geometry.svg \
            --wire-width-nm 100 \
            --thickness-nm 10
    """

    parser = argparse.ArgumentParser(
        description=(
            "SNSPD stationary electrical FEM current-crowding analysis."
        )
    )

    parser.add_argument(
        "input_svg",
        nargs="?",
        default=None,
        help=(
            "Input SVG geometry. If omitted, "
            "examples/simple_meander.svg is used."
        ),
    )

    parser.add_argument(
        "--wire-width-nm",
        type=float,
        required=True,
        help=(
            "Nanowire width in nm. "
            "This is an explicit physical input and does not "
            "come from SVG metadata."
        ),
    )

    parser.add_argument(
        "--thickness-nm",
        type=float,
        required=True,
        help=(
            "Superconducting film thickness in nm. "
            "This is an explicit physical input and does not "
            "come from SVG metadata."
        ),
    )

    parser.add_argument(
        "--fem-output",
        type=str,
        default=None,
        help=(
            "Output .npz file containing the validated FEM mesh, "
            "element-wise current-density field, geometry parameters, "
            "and terminal-current results. If omitted, the result is "
            "saved as results/current_crowding_fem.npz."
        ),
    )

    return parser.parse_args()


def load_geometry(
    input_svg: str | None,
):
    """Load the SVG geometry supplied by the user."""

    if input_svg is not None:

        svg_file = (
            Path(input_svg)
            .expanduser()
            .resolve()
        )

    else:

        svg_file = (
            Path(__file__).resolve().parents[1]
            / "simple_meander.svg"
        )

    if not svg_file.exists():

        raise FileNotFoundError(
            f"Geometry file not found:\n{svg_file}"
        )

    print(
        f"Geometry file : {svg_file}"
    )

    return import_svg(
        svg_file
    )


# ============================================================
# NANOWIRE WIDTH
# ============================================================

def get_nanowire_width(
    override_width_m: float,
) -> float:
    """
    Return the explicitly supplied nanowire width.

    IMPORTANT:
    The electrical analysis does NOT depend on SVG metadata for
    the nanowire width.

    The width is a physical fabrication parameter supplied by
    the user at runtime.

    Example:
        --wire-width-nm 100
    """

    width = float(
        override_width_m
    )

    if not np.isfinite(width):

        raise RuntimeError(
            "Nanowire width must be finite."
        )

    if width <= 0.0:

        raise RuntimeError(
            "Nanowire width must be positive."
        )

    return width


# ============================================================
# BOUNDARY EDGE GEOMETRY
# ============================================================

def calculate_boundary_geometry(
    nodes: np.ndarray,
    boundary_edges,
):
    """
    Calculate midpoint, length and tangent for every
    exterior FEM boundary edge.

    build_boundary_edges() returns BoundaryEdge objects,
    not an Nx2 NumPy array.
    """

    n_edges = len(boundary_edges)

    p1 = np.zeros(
        (n_edges, 2),
        dtype=float,
    )

    p2 = np.zeros(
        (n_edges, 2),
        dtype=float,
    )

    for i, edge in enumerate(
        boundary_edges
    ):

        p1[i] = nodes[
            edge.node_a
        ]

        p2[i] = nodes[
            edge.node_b
        ]

    vectors = p2 - p1

    lengths = np.linalg.norm(
        vectors,
        axis=1,
    )

    midpoints = (
        0.5
        * (
            p1 + p2
        )
    )

    tangents = np.zeros_like(
        vectors
    )

    valid = lengths > 0.0

    tangents[valid] = (
        vectors[valid]
        / lengths[valid, None]
    )

    return (
        midpoints,
        lengths,
        tangents,
    )


# ============================================================
# ANGLE BETWEEN BOUNDARY TANGENTS
# ============================================================

def acute_tangent_angle_deg(
    tangent_a: np.ndarray,
    tangent_b: np.ndarray,
) -> float:
    """
    Return the acute angle between two unoriented tangents.

    A boundary edge may be represented in either direction,
    therefore t and -t describe the same geometric tangent.

    Returns
    -------
    float
        Acute angle in degrees in [0, 90].
    """

    norm_a = np.linalg.norm(
        tangent_a
    )

    norm_b = np.linalg.norm(
        tangent_b
    )

    if norm_a <= 0.0 or norm_b <= 0.0:

        raise ValueError(
            "Cannot compare a zero-length tangent."
        )

    cosine = abs(
        float(
            np.dot(
                tangent_a,
                tangent_b,
            )
        )
        / (
            norm_a
            * norm_b
        )
    )

    cosine = np.clip(
        cosine,
        -1.0,
        1.0,
    )

    return float(
        np.degrees(
            np.arccos(
                cosine
            )
        )
    )


# ============================================================
# SELECT PHYSICAL END CAP FROM USER COORDINATE
# ============================================================

def select_terminal_from_point(
    nodes: np.ndarray,
    boundary_edges,
    requested_point: tuple[float, float] | np.ndarray,
    name: str,
):
    """
    Select a physical nanowire end-cap from a user-specified
    coordinate.

    The coordinate should be placed on or very near the centre
    of the physical end-cap.

    Selection algorithm
    -------------------
    1. Find the exterior FEM boundary edge whose midpoint is
       closest to the requested coordinate.
    2. Treat that edge as the seed edge.
    3. Follow connected exterior boundary edges.
    4. Keep only edges whose tangent is collinear with the seed.
    5. The resulting connected collinear set is the physical
       terminal end-cap.

    This deliberately avoids assumptions such as:

        x = xmin
        x = xmax
        y = ymin
        y = ymax

    because those assumptions are not valid for arbitrary
    serpentine SNSPD geometries.
    """

    point = np.asarray(
        requested_point,
        dtype=float,
    )

    if point.shape != (2,):

        raise ValueError(
            f"{name} terminal coordinate must have "
            "shape (2,)."
        )

    (
        midpoints,
        lengths,
        tangents,
    ) = calculate_boundary_geometry(
        nodes,
        boundary_edges,
    )

    if len(boundary_edges) == 0:

        raise RuntimeError(
            f"{name} terminal selection failed: "
            "FEM boundary contains no edges."
        )

    # --------------------------------------------------------
    # Find nearest boundary edge.
    # --------------------------------------------------------

    distances = np.linalg.norm(
        midpoints
        - point[None, :],
        axis=1,
    )

    seed_edge = int(
        np.argmin(
            distances
        )
    )

    seed_distance = float(
        distances[
            seed_edge
        ]
    )

    if seed_distance > TERMINAL_MAX_POINT_DISTANCE:

        raise RuntimeError(
            f"{name} terminal coordinate is too far from "
            f"the FEM boundary.\n"
            f"Requested point : "
            f"({point[0] * 1e6:.6f}, "
            f"{point[1] * 1e6:.6f}) um\n"
            f"Nearest boundary: "
            f"({midpoints[seed_edge, 0] * 1e6:.6f}, "
            f"{midpoints[seed_edge, 1] * 1e6:.6f}) um\n"
            f"Distance        : "
            f"{seed_distance * 1e9:.6f} nm\n"
            f"Allowed         : "
            f"{TERMINAL_MAX_POINT_DISTANCE * 1e9:.6f} nm"
        )

    seed_tangent = tangents[
        seed_edge
    ]

    # --------------------------------------------------------
    # Build boundary-edge adjacency.
    #
    # Two boundary edges are adjacent when they share a FEM
    # node. BoundaryEdge contains node_a and node_b.
    # --------------------------------------------------------

    node_to_edges: dict[int, list[int]] = {}

    for edge_index, edge in enumerate(
        boundary_edges
    ):

        node_to_edges.setdefault(
            int(edge.node_a),
            [],
        ).append(
            edge_index
        )

        node_to_edges.setdefault(
            int(edge.node_b),
            [],
        ).append(
            edge_index
        )

    # --------------------------------------------------------
    # Traverse connected boundary edges.
    #
    # Only geometrically collinear edges are accepted.
    # This prevents the traversal from continuing around the
    # side walls at the 90-degree corners of the nanowire.
    # --------------------------------------------------------

    selected: set[int] = {
        seed_edge
    }

    queue = [
        seed_edge
    ]

    while queue:

        current = queue.pop()

        current_edge = (
            boundary_edges[current]
        )

        neighboring_edges = set(
            node_to_edges.get(
                int(current_edge.node_a),
                [],
            )
            + node_to_edges.get(
                int(current_edge.node_b),
                [],
            )
        )

        for neighbor in neighboring_edges:

            if neighbor in selected:

                continue

            angle = (
                acute_tangent_angle_deg(
                    seed_tangent,
                    tangents[neighbor],
                )
            )

            if angle <= TERMINAL_ANGLE_TOLERANCE_DEG:

                selected.add(
                    neighbor
                )

                queue.append(
                    neighbor
                )

    selected_indices = np.array(
        sorted(selected),
        dtype=int,
    )

    # --------------------------------------------------------
    # Validate that the selected terminal is not a single
    # accidental zero-length FEM edge.
    # --------------------------------------------------------

    terminal_length = float(
        np.sum(
            lengths[
                selected_indices
            ]
        )
    )

    if terminal_length <= 0.0:

        raise RuntimeError(
            f"{name} terminal has zero boundary length."
        )

    selected_edges = [
        boundary_edges[i]
        for i in selected_indices
    ]

    selected_nodes = np.unique(
        np.array(
            [
                node
                for edge in selected_edges
                for node in (
                    edge.node_a,
                    edge.node_b,
                )
            ],
            dtype=int,
        )
    )

    # --------------------------------------------------------
    # Report the nearest FEM representation of the requested
    # coordinate. This is useful for debugging new SVGs.
    # --------------------------------------------------------

    selected_midpoints = (
        midpoints[
            selected_indices
        ]
    )

    representative_point = np.mean(
        selected_midpoints,
        axis=0,
    )

    return (
        PhysicalTerminal(
            name=name,
            requested_point=point.copy(),
            edge_indices=selected_indices,
            node_indices=selected_nodes,
            length_m=terminal_length,
        ),
        seed_edge,
        seed_distance,
        representative_point,
    )


def _geometry_mask(geometry, nx: int, ny: int):
    """Rasterize the canonical Shapely geometry into a boolean mask."""

    union = unary_union([
        region.polygon
        for region in geometry.regions
    ])

    xmin, ymin, xmax, ymax = union.bounds

    xs = np.linspace(xmin, xmax, nx)
    ys = np.linspace(ymin, ymax, ny)

    mask = np.zeros((ny, nx), dtype=bool)

    polygons = []
    if union.geom_type == "Polygon":
        polygons = [union]
    elif union.geom_type == "MultiPolygon":
        polygons = list(union.geoms)
    else:
        union = union.buffer(0)
        if union.geom_type == "Polygon":
            polygons = [union]
        elif union.geom_type == "MultiPolygon":
            polygons = list(union.geoms)

    def fill_ring(ring, value):
        coords = np.asarray(ring.coords)
        px = (coords[:, 0] - xmin) / (xmax - xmin) * (nx - 1)
        py = (coords[:, 1] - ymin) / (ymax - ymin) * (ny - 1)
        rr, cc = raster_polygon(py, px, shape=mask.shape)
        if value:
            mask[rr, cc] = True
        else:
            mask[rr, cc] = False

    for poly in polygons:
        fill_ring(poly.exterior, True)
        for hole in poly.interiors:
            fill_ring(hole, False)

    return mask, xmin, ymin, xmax, ymax


def _skeleton_endpoints(geometry, wire_width: float):
    """Return the two physical endpoints of the nanowire skeleton."""

    union = unary_union([
        region.polygon
        for region in geometry.regions
    ])

    xmin, ymin, xmax, ymax = union.bounds
    width = xmax - xmin
    height = ymax - ymin
    longest = max(width, height)

    if longest <= 0.0:
        raise RuntimeError("Cannot detect terminals in zero-size geometry.")

    # Resolve the wire sufficiently for topology while keeping arbitrary
    # large geometries computationally bounded.
    if wire_width > 0.0:
        desired = int(np.ceil(longest / wire_width * AUTO_TERMINAL_PIXELS_PER_WIDTH))
    else:
        desired = 1000

    nx = int(np.clip(desired, AUTO_TERMINAL_MIN_PIXELS, AUTO_TERMINAL_MAX_PIXELS))
    ny = max(AUTO_TERMINAL_MIN_PIXELS, int(round(nx * height / width)))
    ny = min(ny, AUTO_TERMINAL_MAX_PIXELS)

    mask, xmin, ymin, xmax, ymax = _geometry_mask(geometry, nx, ny)

    if not np.any(mask):
        raise RuntimeError("Automatic terminal detection produced an empty raster mask.")

    skeleton = skeletonize(mask)

    # Count 8-connected skeleton neighbours.
    padded = np.pad(skeleton.astype(np.uint8), 1)
    neighbours = np.zeros_like(skeleton, dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neighbours += padded[1 + dy:1 + dy + skeleton.shape[0],
                                 1 + dx:1 + dx + skeleton.shape[1]]

    # Keep the largest connected skeleton component. A valid SNSPD
    # nanowire should be one connected conductor; this also prevents
    # decorative/disconnected SVG objects from becoming terminals.
    labels = label(skeleton, connectivity=2)
    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0
    largest_component = int(np.argmax(component_sizes))

    if largest_component == 0:
        raise RuntimeError("Automatic terminal detection found no connected nanowire skeleton.")

    skeleton = labels == largest_component

    padded = np.pad(skeleton.astype(np.uint8), 1)
    neighbours = np.zeros_like(skeleton, dtype=np.uint8)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            neighbours += padded[1 + dy:1 + dy + skeleton.shape[0],
                                 1 + dx:1 + dx + skeleton.shape[1]]

    endpoint_pixels = np.argwhere(skeleton & (neighbours == 1))

    if len(endpoint_pixels) < 2:
        raise RuntimeError(
            "Automatic terminal detection could not find two skeleton endpoints. "
            "The geometry may be closed, disconnected, or too thin for the "
            "automatic raster resolution."
        )

    # For a meander the two electrical ends can be spatially close, so use
    # skeleton graph distance rather than Euclidean distance.
    skeleton_set = {tuple(p) for p in np.argwhere(skeleton)}

    def graph_distances(start):
        from collections import deque
        dist = {tuple(start): 0}
        queue = deque([tuple(start)])
        while queue:
            cy, cx = queue.popleft()
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    q = (cy + dy, cx + dx)
                    if q in skeleton_set and q not in dist:
                        dist[q] = dist[(cy, cx)] + (2 ** 0.5 if dx and dy else 1.0)
                        queue.append(q)
        return dist

    best_pair = None
    best_distance = -1.0
    for i, endpoint in enumerate(endpoint_pixels):
        distances = graph_distances(endpoint)
        for j in range(i + 1, len(endpoint_pixels)):
            d = distances.get(tuple(endpoint_pixels[j]), -1.0)
            if d > best_distance:
                best_distance = d
                best_pair = (i, j)

    if best_pair is None:
        raise RuntimeError("Could not determine the two terminal endpoints from the skeleton graph.")

    endpoint_pixels = endpoint_pixels[list(best_pair)]

    points = np.column_stack((
        xmin + endpoint_pixels[:, 1] / (nx - 1) * (xmax - xmin),
        ymin + endpoint_pixels[:, 0] / (ny - 1) * (ymax - ymin),
    ))

    return points, endpoint_pixels, skeleton, (xmin, ymin, xmax, ymax)


def _terminal_from_endpoint(
    nodes,
    boundary_edges,
    endpoint,
    skeleton_endpoint_pixel,
    skeleton,
    bounds,
    wire_width,
    name,
):
    """Map one skeleton endpoint to its physical FEM end-cap."""

    midpoints, lengths, tangents = calculate_boundary_geometry(
        nodes,
        boundary_edges,
    )

    xmin, ymin, xmax, ymax = bounds
    ny, nx = skeleton.shape
    py, px = skeleton_endpoint_pixel

    # Find the nearest skeleton pixel to the endpoint that is not itself.
    skeleton_pixels = np.argwhere(skeleton)
    distances = np.sum((skeleton_pixels - np.array([py, px])) ** 2, axis=1)
    order = np.argsort(distances)
    neighbour_pixel = None
    for idx in order:
        q = skeleton_pixels[idx]
        if not (q[0] == py and q[1] == px):
            neighbour_pixel = q
            break

    if neighbour_pixel is None:
        raise RuntimeError(f"Could not determine local skeleton direction for {name}.")

    local_direction = np.array([
        (neighbour_pixel[1] - px) / max(nx - 1, 1) * (xmax - xmin),
        (neighbour_pixel[0] - py) / max(ny - 1, 1) * (ymax - ymin),
    ], dtype=float)
    norm = np.linalg.norm(local_direction)
    if norm <= 0.0:
        raise RuntimeError(f"Could not determine local skeleton direction for {name}.")
    local_direction /= norm

    # Candidate boundary edges near the skeleton endpoint.
    distances = np.linalg.norm(midpoints - endpoint[None, :], axis=1)
    radius = AUTO_TERMINAL_BOUNDARY_RADIUS_FACTOR * wire_width
    candidates = np.where(distances <= radius)[0]

    if len(candidates) == 0:
        # Fall back to nearest boundary edge if the supplied width is only an
        # approximate value.
        candidates = np.array([int(np.argmin(distances))], dtype=int)

    # An end-cap boundary tangent is approximately perpendicular to the local
    # nanowire direction. Prefer such an edge over the longitudinal side walls.
    perpendicular_error = np.array([
        abs(90.0 - acute_tangent_angle_deg(local_direction, tangents[i]))
        for i in candidates
    ])
    score = distances[candidates] / max(wire_width, 1e-30) + perpendicular_error
    seed_edge = int(candidates[int(np.argmin(score))])
    seed_tangent = tangents[seed_edge]

    # Boundary adjacency.
    node_to_edges = {}
    for edge_index, edge in enumerate(boundary_edges):
        node_to_edges.setdefault(int(edge.node_a), []).append(edge_index)
        node_to_edges.setdefault(int(edge.node_b), []).append(edge_index)

    # Select the connected end-cap region. Unlike the old implementation,
    # curved caps are allowed: an edge may be selected if it is both close to
    # the skeleton endpoint and belongs to the local cap tangent family.
    selected = {seed_edge}
    queue = [seed_edge]
    max_radius = max(2.5 * wire_width, 3.0 * MESH_SIZE)

    while queue:
        current = queue.pop()
        edge = boundary_edges[current]
        neighbours = set(
            node_to_edges.get(int(edge.node_a), [])
            + node_to_edges.get(int(edge.node_b), [])
        )

        for nb in neighbours:
            if nb in selected:
                continue
            if distances[nb] > max_radius:
                continue
            # Accept edges that are part of the cap, including curved caps.
            angle_to_cap = abs(
                90.0 - acute_tangent_angle_deg(local_direction, tangents[nb])
            )
            if angle_to_cap <= 35.0:
                selected.add(nb)
                queue.append(nb)

    if len(selected) < AUTO_TERMINAL_MIN_CAP_EDGES:
        # The FEM mesh may represent a very small cap with only one or two
        # edges. Keep the nearest perpendicular edges rather than failing.
        ranked = sorted(
            candidates.tolist(),
            key=lambda i: score[np.where(candidates == i)[0][0]],
        )
        selected.update(ranked[:AUTO_TERMINAL_MIN_CAP_EDGES])

    selected_indices = np.array(sorted(selected), dtype=int)
    selected_edges = [boundary_edges[i] for i in selected_indices]
    selected_nodes = np.unique(np.array([
        node for edge in selected_edges
        for node in (edge.node_a, edge.node_b)
    ], dtype=int))
    terminal_length = float(np.sum(lengths[selected_indices]))

    if terminal_length <= 0.0:
        raise RuntimeError(f"{name} has zero boundary length.")

    representative = np.mean(midpoints[selected_indices], axis=0)

    return PhysicalTerminal(
        name=name,
        requested_point=np.asarray(endpoint, dtype=float),
        edge_indices=selected_indices,
        node_indices=selected_nodes,
        length_m=terminal_length,
    ), representative


def identify_physical_terminals(
    geometry,
    nodes: np.ndarray,
    triangles: np.ndarray,
    wire_width: float,
):
    """Automatically identify the two physical open ends of the SNSPD."""

    boundary_edges = build_boundary_edges(nodes, triangles)

    endpoints, endpoint_pixels, skeleton, bounds = _skeleton_endpoints(
        geometry,
        wire_width,
    )

    if len(endpoints) != 2:
        raise RuntimeError("Automatic terminal detection did not produce exactly two endpoints.")

    # Positive = first detected end; negative = second. The physical solution
    # is invariant to swapping them, but reporting is deterministic by sorting
    # first by x and then y.
    order = np.lexsort((endpoints[:, 1], endpoints[:, 0]))
    endpoints = endpoints[order]
    endpoint_pixels = endpoint_pixels[order]

    positive_terminal, positive_representative = _terminal_from_endpoint(
        nodes, boundary_edges, endpoints[0], endpoint_pixels[0], skeleton,
        bounds, wire_width, "POSITIVE / CURRENT ENTRY"
    )
    negative_terminal, negative_representative = _terminal_from_endpoint(
        nodes, boundary_edges, endpoints[1], endpoint_pixels[1], skeleton,
        bounds, wire_width, "NEGATIVE / CURRENT EXIT"
    )

    if np.intersect1d(
        positive_terminal.node_indices,
        negative_terminal.node_indices,
    ).size > 0:
        raise RuntimeError("Automatically detected terminals share FEM nodes.")

    if np.intersect1d(
        positive_terminal.edge_indices,
        negative_terminal.edge_indices,
    ).size > 0:
        raise RuntimeError("Automatically detected terminals share FEM boundary edges.")

    terminal_information = {
        "positive_endpoint": endpoints[0],
        "negative_endpoint": endpoints[1],
        "positive_representative_point": positive_representative,
        "negative_representative_point": negative_representative,
    }

    return boundary_edges, positive_terminal, negative_terminal, terminal_information



# ============================================================
# TERMINAL MIDPOINTS
# ============================================================

def terminal_midpoints(
    nodes: np.ndarray,
    boundary_edges,
    terminal: PhysicalTerminal,
) -> np.ndarray:
    """
    Return midpoint coordinates of the terminal boundary edges.
    """

    selected_edges = [
        boundary_edges[i]
        for i in terminal.edge_indices
    ]

    if len(selected_edges) == 0:

        return np.empty(
            (0, 2),
            dtype=float,
        )

    midpoints = []

    for edge in selected_edges:

        p1 = nodes[
            edge.node_a
        ]

        p2 = nodes[
            edge.node_b
        ]

        midpoints.append(
            0.5
            * (
                p1 + p2
            )
        )

    return np.asarray(
        midpoints,
        dtype=float,
    )
# ============================================================
# CURRENT-DENSITY HEATMAP
# ============================================================

def plot_current_density(
    mesh,
    result,
    boundary_edges,
    positive_terminal,
    negative_terminal,
    terminal_current,
    crowding_factor,
    hotspot,
    target_bias_current,
):
    """
    Plot FEM element current-density magnitude.

    Each triangle receives its computed FEM |J| value.

    A logarithmic color normalization is used because SNSPD
    current-density fields can span several orders of magnitude.
    """

    element_J = np.asarray(
        result.element_current_density,
        dtype=float,
    )

    J_magnitude = np.linalg.norm(
        element_J,
        axis=1,
    )

    triangles = mesh.triangles

    points = mesh.nodes[
        triangles
    ]

    x = (
        points[:, :, 0]
        * 1e6
    )

    y = (
        points[:, :, 1]
        * 1e6
    )

    # --------------------------------------------------------
    # Avoid zero/negative values in logarithmic normalization.
    # --------------------------------------------------------

    positive_values = J_magnitude[
        np.isfinite(J_magnitude)
        & (J_magnitude > 0.0)
    ]

    if len(positive_values) == 0:
        raise RuntimeError(
            "Current-density field contains no positive finite values."
        )

    # ------------------------------------------------------------
    # ROBUST HEATMAP COLOR SCALE
    # ------------------------------------------------------------
    #
    # Do NOT use the absolute Jmax as the upper color limit.
    #
    # A highly localized current-crowding hotspot can otherwise
    # compress almost the entire rest of the device into the
    # red/orange part of the colormap.
    #
    # Instead:
    #
    #   lower limit = 5th percentile
    #   upper limit = 99th percentile
    #
    # The actual Jmax is still reported separately and the
    # hotspot is marked on the plot.
    # ------------------------------------------------------------

    vmin = float(
        np.percentile(
            positive_values,
            5.0
        )
    )

    vmax = float(
        np.percentile(
            positive_values,
            99.0
        )
    )

    # Protect against degenerate distributions.
    if vmin <= 0.0:
        vmin = float(np.min(positive_values))

    if vmax <= vmin:
        vmax = float(np.max(positive_values))

    if vmax <= vmin:
        vmax = vmin * 10.0

    from matplotlib.colors import LogNorm

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    # --------------------------------------------------------
    # FEM element field
    # --------------------------------------------------------

    collection = ax.tripcolor(
        mesh.nodes[:, 0] * 1e6,
        mesh.nodes[:, 1] * 1e6,
        triangles,
        J_magnitude,
        shading="flat",
        norm=LogNorm(
            vmin=vmin,
            vmax=vmax,
        ),
        cmap="turbo",
    )

    colorbar = fig.colorbar(
        collection,
        ax=ax,
    )

    colorbar.set_label(
        r"$|\mathbf{J}|$ [A/m$^2$]"
    )

    # --------------------------------------------------------
    # FEM mesh outline
    # --------------------------------------------------------

    ax.triplot(
        mesh.nodes[:, 0] * 1e6,
        mesh.nodes[:, 1] * 1e6,
        triangles,
        linewidth=0.15,
        alpha=0.25,
        color="black",
    )

    # --------------------------------------------------------
    # Physical terminals
    # --------------------------------------------------------

    positive_points = (
        terminal_midpoints(
            mesh.nodes,
            boundary_edges,
            positive_terminal,
        )
    )

    negative_points = (
        terminal_midpoints(
            mesh.nodes,
            boundary_edges,
            negative_terminal,
        )
    )

    ax.scatter(
        positive_points[:, 0] * 1e6,
        positive_points[:, 1] * 1e6,
        s=18,
        marker="o",
        label="Positive terminal",
        zorder=20,
    )

    ax.scatter(
        negative_points[:, 0] * 1e6,
        negative_points[:, 1] * 1e6,
        s=18,
        marker="s",
        label="Negative terminal",
        zorder=20,
    )

    # --------------------------------------------------------
    # Hotspot
    # --------------------------------------------------------

    ax.scatter(
        hotspot[0] * 1e6,
        hotspot[1] * 1e6,
        s=100,
        marker="o",
        facecolors="none",
        edgecolors="white",
        linewidths=2.0,
        zorder=30,
        label="Maximum |J|",
    )

    # --------------------------------------------------------
    # Annotation
    # --------------------------------------------------------

    annotation = (
        f"Transport current = "
        f"{terminal_current:.6e} A\n"
        f"Target bias current = "
        f"{target_bias_current:.6e} A\n"
        f"Maximum |J| = "
        f"{vmax:.6e} A/m²\n"
        f"C_J = "
        f"{crowding_factor:.6f}\n"
        f"Hotspot = "
        f"({hotspot[0] * 1e6:.4f}, "
        f"{hotspot[1] * 1e6:.4f}) µm"
    )

    ax.text(
        0.02,
        0.02,
        annotation,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=dict(
            boxstyle="round",
            alpha=0.85,
        ),
    )

    ax.set_title(
        "SNSPD FEM CURRENT DENSITY |J|"
    )

    ax.set_xlabel(
        "x [µm]"
    )

    ax.set_ylabel(
        "y [µm]"
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.legend(
        loc="upper right"
    )

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    if SAVE_HEATMAP:

        RESULTS_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_file = (
            RESULTS_DIRECTORY
            / HEATMAP_FILENAME
        )

        fig.savefig(
            output_file,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            "\nCurrent-density visualization saved to:"
        )

        print(
            output_file
        )

    if SHOW_HEATMAP:

        plt.show()

    else:

        plt.close(
            fig
        )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_arguments()

    wire_width = get_nanowire_width(
        args.wire_width_nm * 1.0e-9
    )

    film_thickness = (
        args.thickness_nm * 1.0e-9
    )

    if not np.isfinite(film_thickness) or film_thickness <= 0.0:
        raise RuntimeError(
            "Film thickness must be finite and positive."
        )

    print(
        "\n"
        "====================================================\n"
        "SNSPD CURRENT CROWDING ANALYSIS\n"
        "====================================================\n"
    )

    # ========================================================
    # GEOMETRY
    # ========================================================

    geometry = load_geometry(args.input_svg)

    print(
        geometry.summary()
    )

    geometry_errors = (
        geometry.validate()
    )

    if geometry_errors:

        print(
            "\nGEOMETRY VALIDATION FAILED"
        )

        for error in geometry_errors:

            print(
                f"ERROR: {error}"
            )

        raise RuntimeError(
            "Invalid SNSPD geometry."
        )

    print(
        "Geometry validation : PASS"
    )

    # ========================================================
    # GEOMETRY METRICS
    # ========================================================

    metrics = (
        analyze_geometry(
            geometry
        )
    )

    print(
        format_metrics(
            metrics
        )
    )

    # ========================================================
    # EXPLICIT NANOWIRE PHYSICAL PARAMETERS
    # ========================================================

    print(
        "\n"
        "Nanowire electrical parameters"
    )

    print(
        "-------------------------------"
    )

    print(
        f"Nanowire width          : "
        f"{wire_width * 1e9:.6f} nm"
    )

    print(
        f"Film thickness          : "
        f"{film_thickness * 1e9:.6f} nm"
    )

    print(
        "Geometry/material metadata : NOT USED"
    )

    print(
        f"Normal resistivity      : "
        f"{NORMAL_RESISTIVITY:.6e} ohm m"
    )

    print(
        f"Normal conductivity     : "
        f"{CONDUCTIVITY:.6e} S/m"
    )

    # ========================================================
    # MESH
    # ========================================================

    print(
        "\nGenerating FEM mesh..."
    )

    mesher = GmshMesher(
        characteristic_length=MESH_SIZE
    )

    mesh = mesher.generate(
        geometry
    )

    print(
        mesh.summary()
    )

    mesh_errors = (
        mesh.validate()
    )

    if mesh_errors:

        print(
            "\nMESH VALIDATION FAILED"
        )

        for error in mesh_errors:

            print(
                f"ERROR: {error}"
            )

        raise RuntimeError(
            "Invalid FEM mesh."
        )

    print(
        "Mesh validation : PASS"
    )

    # ========================================================
    # MESH QUALITY
    # ========================================================

    quality = (
        analyze_mesh_quality(
            mesh
        )
    )

    print(
        format_mesh_quality(
            quality
        )
    )

    # ========================================================
    # PHYSICAL TERMINALS
    # ========================================================

    (
        boundary_edges,
        positive_terminal,
        negative_terminal,
        terminal_information,
    ) = identify_physical_terminals(
        geometry,
        mesh.nodes,
        mesh.triangles,
        wire_width,
    )

    print(
        "\n"
        "FEM boundary"
    )

    print(
        "------------"
    )

    print(
        f"Total boundary edges : "
        f"{len(boundary_edges)}"
    )

    print(
        "\n"
        "Electrical terminals"
    )

    print(
        "--------------------"
    )

    print(
        f"Positive terminal edges : "
        f"{positive_terminal.edge_count}"
    )

    print(
        f"Negative terminal edges : "
        f"{negative_terminal.edge_count}"
    )

    print(
        f"Positive terminal length : "
        f"{positive_terminal.length_m * 1e6:.9f} um"
    )

    print(
        f"Negative terminal length : "
        f"{negative_terminal.length_m * 1e6:.9f} um"
    )

    print("\nAutomatically detected physical open ends")
    print("----------------------------------------")
    print(
        f"Positive / entry skeleton endpoint : "
        f"({terminal_information['positive_endpoint'][0] * 1e6:.6f}, "
        f"{terminal_information['positive_endpoint'][1] * 1e6:.6f}) um"
    )
    print(
        f"Negative / exit skeleton endpoint  : "
        f"({terminal_information['negative_endpoint'][0] * 1e6:.6f}, "
        f"{terminal_information['negative_endpoint'][1] * 1e6:.6f}) um"
    )
    print(
        f"Positive FEM cap centre           : "
        f"({terminal_information['positive_representative_point'][0] * 1e6:.6f}, "
        f"{terminal_information['positive_representative_point'][1] * 1e6:.6f}) um"
    )
    print(
        f"Negative FEM cap centre           : "
        f"({terminal_information['negative_representative_point'][0] * 1e6:.6f}, "
        f"{terminal_information['negative_representative_point'][1] * 1e6:.6f}) um"
    )

    # ========================================================
    # TERMINAL NODE SETS
    # ========================================================

    positive_nodes = (
        positive_terminal.node_indices
    )

    negative_nodes = (
        negative_terminal.node_indices
    )

    print(
        f"\nPositive terminal nodes : "
        f"{len(positive_nodes)}"
    )

    print(
        f"Negative terminal nodes : "
        f"{len(negative_nodes)}"
    )

    # ========================================================
    # FEM ELECTRICAL SOLUTION
    # ========================================================

    print(
        "\nSolving stationary electrical FEM..."
    )

    solver = CurrentDistributionSolver(
        nodes=mesh.nodes,
        triangles=mesh.triangles,
        conductivity=CONDUCTIVITY,
        thickness=film_thickness,
    )

    result = solver.solve(
        positive_terminal_nodes=positive_nodes,
        negative_terminal_nodes=negative_nodes,
        voltage_difference=SOLVE_VOLTAGE,
    )

    # ========================================================
    # REQUIRED RESULT FIELDS
    # ========================================================

    required_result_fields = (
        "positive_terminal_current",
        "negative_terminal_current",
        "total_current",
    )

    missing_fields = [
        name
        for name in required_result_fields
        if not hasattr(
            result,
            name,
        )
    ]

    if missing_fields:

        raise RuntimeError(
            "CurrentDistributionResult is missing "
            "the FEM reaction-current fields: "
            + ", ".join(
                missing_fields
            )
            + ". Update "
            "src/snspd/fem/electrical/"
            "current_distribution.py "
            "to return FEM terminal reaction currents."
        )

    # ========================================================
    # TERMINAL CURRENTS
    # ========================================================

    positive_terminal_current = float(
        result.positive_terminal_current
    )

    negative_terminal_current = float(
        result.negative_terminal_current
    )

    terminal_current = float(
        result.total_current
    )

    # ========================================================
    # CURRENT CONSERVATION
    # ========================================================

    terminal_current_difference = abs(
        positive_terminal_current
        - negative_terminal_current
    )

    terminal_current_scale = max(
        abs(
            positive_terminal_current
        ),
        abs(
            negative_terminal_current
        ),
    )

    if terminal_current_scale > 0.0:

        current_difference = (
            terminal_current_difference
            / terminal_current_scale
        )

    else:

        current_difference = 0.0

    print(
        "\n"
        "POSITIVE TERMINAL\n"
        "-----------------"
    )

    print(
        f"FEM reaction current : "
        f"{positive_terminal_current:.12e} A"
    )

    print(
        "\n"
        "NEGATIVE TERMINAL\n"
        "-----------------"
    )

    print(
        f"FEM reaction current : "
        f"{negative_terminal_current:.12e} A"
    )

    print(
        "\n"
        "TERMINAL CURRENT CONSERVATION\n"
        "-----------------------------"
    )

    print(
        f"Positive |I| : "
        f"{abs(positive_terminal_current):.12e} A"
    )

    print(
        f"Negative |I| : "
        f"{abs(negative_terminal_current):.12e} A"
    )

    print(
        f"Absolute difference : "
        f"{terminal_current_difference:.12e} A"
    )

    print(
        f"Relative difference : "
        f"{current_difference:.12e}"
    )

    # ========================================================
    # CURRENT-DENSITY FIELD
    # ========================================================

    element_J = np.asarray(
        result.element_current_density,
        dtype=float,
    )

    if element_J.ndim != 2:

        raise RuntimeError(
            "Element current-density array must "
            "have shape (N, 2)."
        )

    if element_J.shape[1] != 2:

        raise RuntimeError(
            "Element current-density field must "
            "contain Jx and Jy."
        )

    if len(element_J) != mesh.element_count:

        raise RuntimeError(
            "Number of element current-density values "
            "does not match the FEM mesh."
        )

    element_J_magnitude = np.linalg.norm(
        element_J,
        axis=1,
    )

    element_areas = (
        mesh.triangle_areas()
    )

    total_area = float(
        np.sum(
            element_areas
        )
    )

    # ========================================================
    # MAXIMUM CURRENT DENSITY
    # ========================================================

    maximum_J_index = int(
        np.argmax(
            element_J_magnitude
        )
    )

    maximum_J = float(
        element_J_magnitude[
            maximum_J_index
        ]
    )

    # ========================================================
    # AREA-WEIGHTED CURRENT DENSITY
    # ========================================================

    area_average_J = float(
        np.sum(
            element_J_magnitude
            * element_areas
        )
        / total_area
    )

    # ========================================================
    # TERMINAL-AVERAGE TRANSPORT CURRENT DENSITY
    # ========================================================

    terminal_length = (
        0.5
        * (
            positive_terminal.length_m
            +
            negative_terminal.length_m
        )
    )

    if terminal_length <= 0.0:

        raise RuntimeError(
            "Terminal length is zero."
        )

    terminal_average_J = (
        terminal_current
        / (
            film_thickness
            * terminal_length
        )
    )

    # ========================================================
    # CURRENT CROWDING FACTOR
    # ========================================================

    crowding_factor = (
        maximum_J
        / terminal_average_J
    )

    # ========================================================
    # HOTSPOT
    # ========================================================

    triangle_nodes = (
        mesh.triangles[
            maximum_J_index
        ]
    )

    triangle_points = (
        mesh.nodes[
            triangle_nodes
        ]
    )

    hotspot = np.mean(
        triangle_points,
        axis=0,
    )

    # ========================================================
    # REPORT
    # ========================================================

    print(
        "\n"
        "===================================================="
    )

    print(
        "\nCURRENT DISTRIBUTION"
    )

    print(
        "--------------------"
    )

    print(
        f"Applied FEM voltage       : "
        f"{SOLVE_VOLTAGE:.6e} V"
    )

    print(
        f"Computed terminal current : "
        f"{terminal_current:.9e} A"
    )

    print(
        "Terminal current source    : "
        "FEM reaction vector"
    )

    print(
        f"Maximum |J|               : "
        f"{maximum_J:.9e} A/m²"
    )

    print(
        f"Area-weighted <|J|>       : "
        f"{area_average_J:.9e} A/m²"
    )

    print(
        f"Terminal-average |J|      : "
        f"{terminal_average_J:.9e} A/m²"
    )

    print(
        f"Current crowding factor   : "
        f"{crowding_factor:.9f}"
    )

    # ========================================================
    # HOTSPOT REPORT
    # ========================================================

    print(
        "\n"
        "CURRENT-DENSITY HOTSPOT"
    )

    print(
        "-----------------------"
    )

    print(
        f"Element index             : "
        f"{maximum_J_index}"
    )

    print(
        f"x                         : "
        f"{hotspot[0] * 1e6:.9f} um"
    )

    print(
        f"y                         : "
        f"{hotspot[1] * 1e6:.9f} um"
    )

    # ========================================================
    # BIAS SCALING
    # ========================================================

    scale = (
        TARGET_BIAS_CURRENT
        / terminal_current
    )

    bias_max_J = (
        maximum_J
        * scale
    )

    print(
        "\n"
        "SCALED BIAS-CURRENT RESULT"
    )

    print(
        "---------------------------"
    )

    print(
        f"Target bias current       : "
        f"{TARGET_BIAS_CURRENT:.9e} A"
    )

    print(
        f"Scaling factor            : "
        f"{scale:.9e}"
    )

    print(
        f"Maximum |J| at bias       : "
        f"{bias_max_J:.9e} A/m²"
    )

    # ========================================================
    # VALIDATION
    # ========================================================

    if not np.isfinite(
        terminal_current
    ):

        raise RuntimeError(
            "Terminal current is not finite."
        )

    if terminal_current <= 0.0:

        raise RuntimeError(
            "Terminal current must be positive."
        )

    if not np.isfinite(
        crowding_factor
    ):

        raise RuntimeError(
            "Current crowding factor is not finite."
        )

    if crowding_factor <= 0.0:

        raise RuntimeError(
            "Current crowding factor must be positive."
        )

    # The two FEM reaction currents should balance to numerical
    # precision in the source-free stationary problem.
    #
    # Do NOT loosen this tolerance merely to make the test pass.

    if current_difference > 1.0e-7:

        raise RuntimeError(
            "Terminal current conservation FAILED."
        )

    # ========================================================
    # EXPORT VALIDATED FEM FIELD
    # ========================================================
    #
    # This file is the interface between the electrical FEM stage
    # and later SNSPD physics stages such as critical-current
    # analysis.
    #
    # The export contains the actual element-by-element FEM
    # current-density field. No Jmax-only approximation is used.
    # ========================================================

    fem_output = (
        Path(args.fem_output).expanduser().resolve()
        if args.fem_output is not None
        else RESULTS_DIRECTORY / "current_crowding_fem.npz"
    )

    fem_output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    triangle_centers = np.mean(
        mesh.nodes[mesh.triangles],
        axis=1,
    )

    np.savez_compressed(
        fem_output,
        nodes_m=np.asarray(mesh.nodes, dtype=float),
        triangles=np.asarray(mesh.triangles, dtype=np.int64),
        triangle_areas_m2=np.asarray(element_areas, dtype=float),
        triangle_centers_m=np.asarray(triangle_centers, dtype=float),
        element_J_A_per_m2=np.asarray(element_J, dtype=float),
        element_J_magnitude_A_per_m2=np.asarray(
            element_J_magnitude,
            dtype=float,
        ),
        fem_current_A=float(terminal_current),
        positive_terminal_current_A=float(
            positive_terminal_current
        ),
        negative_terminal_current_A=float(
            negative_terminal_current
        ),
        current_conservation_relative=float(
            current_difference
        ),
        crowding_factor=float(crowding_factor),
        maximum_J_A_per_m2=float(maximum_J),
        terminal_average_J_A_per_m2=float(
            terminal_average_J
        ),
        area_average_J_A_per_m2=float(
            area_average_J
        ),
        solve_voltage_V=float(SOLVE_VOLTAGE),
        target_bias_current_A=float(
            TARGET_BIAS_CURRENT
        ),
        wire_width_m=float(wire_width),
        film_thickness_m=float(film_thickness),
        conductivity_S_per_m=float(CONDUCTIVITY),
        normal_resistivity_ohm_m=float(
            NORMAL_RESISTIVITY
        ),
        hotspot_x_m=float(hotspot[0]),
        hotspot_y_m=float(hotspot[1]),
        hotspot_element_index=int(maximum_J_index),
        positive_terminal_length_m=float(
            positive_terminal.length_m
        ),
        negative_terminal_length_m=float(
            negative_terminal.length_m
        ),
    )

    print(
        "\nValidated FEM field saved to:"
    )

    print(
        fem_output
    )

    # ========================================================
    # VISUALIZATION
    # ========================================================

    print(
        "\n"
        "Generating current-density heatmap..."
    )

    plot_current_density(
        mesh=mesh,
        result=result,
        boundary_edges=boundary_edges,
        positive_terminal=positive_terminal,
        negative_terminal=negative_terminal,
        terminal_current=terminal_current,
        crowding_factor=crowding_factor,
        hotspot=hotspot,
        target_bias_current=TARGET_BIAS_CURRENT,
    )

    # ========================================================
    # INTERPRETATION
    # ========================================================

    print(
        "\n"
        "INTERPRETATION"
    )

    print(
        "--------------"
    )

    print(
        "Terminal current:"
    )

    print(
        "    I = t * integral_Gamma(J.n ds)"
    )

    print(
        "\nTerminal-average current density:"
    )

    print(
        "    J_avg = I / (t * L_terminal)"
    )

    print(
        "\nCurrent-crowding factor:"
    )

    print(
        "    C_J = J_max / J_avg"
    )

    print(
        "\nC_J approximately 1:"
    )

    print(
        "    approximately uniform current distribution"
    )

    print(
        "\nC_J greater than 1:"
    )

    print(
        "    current crowding is present"
    )

    print(
        "\n"
        "SNSPD current-crowding analysis : PASS"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()