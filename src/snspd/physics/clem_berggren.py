"""
Clem-Berggren vortex-entry critical-current model for SNSPDs.

Reference
---------
J. R. Clem and K. K. Berggren,
"Geometry-dependent critical currents in superconducting nanocircuits",
Physical Review B 84, 174510 (2011).

This implementation combines:

    FEM volume current density
        |
        v
    sheet current K = d J
        |
        v
    local re-entrant corner geometry
        |
        v
    Clem-Berggren wedge exponent
        |
        v
    local K(r) asymptotic fit
        |
        v
    Gibbs barrier criterion
        |
        v
    local critical current
        |
        v
    device critical current = minimum local Ic

IMPORTANT
---------
The FEM solution is treated as linear in transport current.

The Clem-Berggren model is only applied where its thin-film/
London/wedge assumptions are numerically plausible.

This is a numerical FEM + local Clem-Berggren implementation.
It is NOT an analytic conformal-mapping solution for arbitrary SVG
geometries.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import e, exp, log, pi
from typing import Callable

import numpy as np

try:
    import matplotlib.tri as mtri
except ImportError as exc:
    raise ImportError(
        "The Clem-Berggren analyzer requires matplotlib for "
        "mesh-consistent local corner sampling."
    ) from exc


# ============================================================
# FUNDAMENTAL CONSTANTS
# ============================================================

MU_0 = 4.0e-7 * pi
PHI_0 = 2.067833848e-15
EULER = e


# ============================================================
# PARAMETERS
# ============================================================

@dataclass(frozen=True)
class ClemBerggrenParameters:
    """
    Parameters required by the Clem-Berggren model.

    All dimensional quantities are SI.
    """

    wire_width_m: float
    film_thickness_m: float
    penetration_depth_m: float
    coherence_length_m: float
    temperature_k: float
    material: str = "unspecified"

    # Numerical controls
    fit_r_min_factor: float = 3.0
    fit_r_max_factor: float = 0.20

    # Minimum number of samples for a local fit.
    minimum_fit_points: int = 8

    # Reject fits below this R².
    minimum_fit_r2: float = 0.80

    # Maximum allowed distance from corner as a fraction of
    # the local wire width.
    maximum_fit_width_fraction: float = 0.25

    # Number of rays used in the physical wedge sector.  Keeping this
    # explicit makes the numerical resolution controllable without
    # changing the Clem-Berggren equations.
    number_of_rays: int = 25

    # Numerical tolerance for distinguishing a genuine re-entrant
    # corner from a nearly straight FEM boundary vertex.
    # A value in degrees is used because the geometry criterion is
    # angular; this does not alter the Clem-Berggren exponent.
    minimum_reentrant_angle_excess_deg: float = 2.0

    def __post_init__(self) -> None:

        values = {
            "wire_width_m": self.wire_width_m,
            "film_thickness_m": self.film_thickness_m,
            "penetration_depth_m": self.penetration_depth_m,
            "coherence_length_m": self.coherence_length_m,
        }

        for name, value in values.items():

            if not np.isfinite(value):
                raise ValueError(
                    f"{name} must be finite."
                )

            if value <= 0.0:
                raise ValueError(
                    f"{name} must be positive."
                )

        if self.temperature_k < 0.0:

            raise ValueError(
                "Temperature cannot be negative."
            )

        if self.fit_r_min_factor <= 1.0:

            raise ValueError(
                "fit_r_min_factor should be > 1."
            )

        if self.fit_r_max_factor <= 0.0:

            raise ValueError(
                "fit_r_max_factor must be positive."
            )

        if self.minimum_fit_points < 4:

            raise ValueError(
                "minimum_fit_points must be >= 4."
            )

        if self.number_of_rays < 8:
            raise ValueError(
                "number_of_rays must be >= 8."
            )

        if self.minimum_reentrant_angle_excess_deg <= 0.0:
            raise ValueError(
                "minimum_reentrant_angle_excess_deg must be positive."
            )

        if not (
            0.0
            < self.minimum_fit_r2
            < 1.0
        ):

            raise ValueError(
                "minimum_fit_r2 must lie between 0 and 1."
            )

    # --------------------------------------------------------
    # Pearl length
    # --------------------------------------------------------

    @property
    def pearl_length_m(self) -> float:
        """
        Pearl length:

            Lambda = 2 lambda^2 / d
        """

        return (
            2.0
            * self.penetration_depth_m**2
            / self.film_thickness_m
        )

    # --------------------------------------------------------
    # Dimensionless validity ratios
    # --------------------------------------------------------

    @property
    def width_to_pearl_ratio(self) -> float:

        return (
            self.wire_width_m
            / self.pearl_length_m
        )

    @property
    def coherence_to_width_ratio(self) -> float:

        return (
            self.coherence_length_m
            / self.wire_width_m
        )

    @property
    def thickness_to_lambda_ratio(self) -> float:

        return (
            self.film_thickness_m
            / self.penetration_depth_m
        )


# ============================================================
# RESULTS
# ============================================================

@dataclass(frozen=True)
class CornerResult:

    vertex_index: int

    x_m: float
    y_m: float

    interior_angle_rad: float
    interior_angle_deg: float

    exponent_p: float

    fit_points: int

    fit_r2: float

    fit_slope: float

    fit_intercept: float

    K0_reference_A_per_m_power: float

    K0_critical_A_per_m_power: float

    delta_barrier_m: float

    delta_critical_m: float

    critical_current_A: float

    accepted: bool

    rejection_reason: str


@dataclass(frozen=True)
class ClemBerggrenResult:

    critical_current_A: float

    straight_strip_critical_current_A: float

    straight_strip_critical_sheet_current_A_per_m: float

    pearl_length_m: float

    penetration_depth_m: float

    coherence_length_m: float

    width_to_pearl_ratio: float

    coherence_to_width_ratio: float

    thickness_to_lambda_ratio: float

    limiting_x_m: float

    limiting_y_m: float

    limiting_angle_deg: float

    limiting_current_A: float

    limiting_K0_reference_A_per_m_power: float

    limiting_K0_critical_A_per_m_power: float

    corners: tuple[CornerResult, ...]

    validity_w_over_lambda: bool

    validity_xi_over_w: bool

    validity_d_over_lambda: bool


# ============================================================
# BASIC CLEM-BERGREN QUANTITIES
# ============================================================

def straight_strip_critical_sheet_current(
    params: ClemBerggrenParameters,
) -> float:
    """
    Clem-Berggren straight-strip critical sheet current:

        K_c =
            Phi_0 /
            (e*pi*mu_0*xi*Lambda)

    Returns
    -------
    float
        Critical sheet current [A/m].
    """

    Lambda = params.pearl_length_m

    return (
        PHI_0
        /
        (
            EULER
            * pi
            * MU_0
            * params.coherence_length_m
            * Lambda
        )
    )


def straight_strip_critical_current(
    params: ClemBerggrenParameters,
) -> float:
    """
    Ideal straight-strip critical current:

        I_c = K_c W
    """

    return (
        straight_strip_critical_sheet_current(params)
        * params.wire_width_m
    )


# ============================================================
# WEDGE GEOMETRY
# ============================================================

def wedge_exponent(
    interior_angle_rad: float,
) -> float:
    """
    Clem-Berggren local wedge exponent:

        p = pi/alpha - 1

    For alpha = pi:

        p = 0

    For a re-entrant corner:

        alpha > pi

    therefore:

        p < 0

    indicating current crowding.
    """

    alpha = float(interior_angle_rad)

    if not (
        0.0
        < alpha
        < 2.0 * pi
    ):

        raise ValueError(
            "Interior angle must satisfy "
            "0 < alpha < 2*pi."
        )

    return (
        pi / alpha
        - 1.0
    )


def wedge_self_energy_factor(
    interior_angle_rad: float,
) -> float:
    """
    Local geometric factor:

        C = 2 alpha / pi
    """

    alpha = float(interior_angle_rad)

    return (
        2.0
        * alpha
        / pi
    )


# ============================================================
# GIBBS BARRIER
# ============================================================

def critical_barrier_distance(
    params: ClemBerggrenParameters,
    interior_angle_rad: float,
) -> float:
    """
    Critical vortex distance from the corner.

        delta_c =
            (xi/C) exp[1/(p+1)]

    """

    p = wedge_exponent(
        interior_angle_rad
    )

    C = wedge_self_energy_factor(
        interior_angle_rad
    )

    return (
        params.coherence_length_m
        / C
        * exp(
            1.0
            / (p + 1.0)
        )
    )


def wedge_critical_K0(
    params: ClemBerggrenParameters,
    interior_angle_rad: float,
) -> float:
    """
    Critical coefficient K0 obtained from:

        G(delta_c) = 0

    giving

        K0,c =
            Phi0 /
            [
                2*pi*mu0*Lambda*delta_c^(p+1)
            ]
    """

    p = wedge_exponent(
        interior_angle_rad
    )

    delta_c = critical_barrier_distance(
        params,
        interior_angle_rad,
    )

    return (
        PHI_0
        /
        (
            2.0
            * pi
            * MU_0
            * params.pearl_length_m
            * delta_c**(p + 1.0)
        )
    )


def gibbs_energy_local(
    delta_m: float,
    K0_A_per_m: float,
    interior_angle_rad: float,
    params: ClemBerggrenParameters,
) -> float:
    """
    Local Gibbs free energy:

        G = E_self - W_I

    """

    if delta_m <= 0.0:

        raise ValueError(
            "delta must be positive."
        )

    p = wedge_exponent(
        interior_angle_rad
    )

    C = wedge_self_energy_factor(
        interior_angle_rad
    )

    Lambda = params.pearl_length_m

    E_self = (
        PHI_0**2
        /
        (
            2.0
            * pi
            * MU_0
            * Lambda
        )
        * log(
            C
            * delta_m
            / params.coherence_length_m
        )
    )

    work = (
        PHI_0
        * K0_A_per_m
        * delta_m**(p + 1.0)
        /
        (p + 1.0)
    )

    return E_self - work


# ============================================================
# MESH BOUNDARY
# ============================================================

def boundary_edges_from_triangles(
    triangles: np.ndarray,
) -> np.ndarray:
    """
    Extract edges appearing exactly once.
    """

    triangles = np.asarray(
        triangles,
        dtype=np.int64,
    )

    if triangles.ndim != 2 or triangles.shape[1] != 3:

        raise ValueError(
            "triangles must have shape (N, 3)."
        )

    edges = np.vstack(
        [
            triangles[:, [0, 1]],
            triangles[:, [1, 2]],
            triangles[:, [2, 0]],
        ]
    )

    edges = np.sort(
        edges,
        axis=1,
    )

    unique_edges, counts = np.unique(
        edges,
        axis=0,
        return_counts=True,
    )

    return unique_edges[
        counts == 1
    ]


def boundary_vertex_neighbors(
    boundary_edges: np.ndarray,
) -> dict[int, list[int]]:
    """
    Build the boundary graph.
    """

    graph: dict[int, list[int]] = {}

    for a, b in boundary_edges:

        a = int(a)
        b = int(b)

        graph.setdefault(a, []).append(b)
        graph.setdefault(b, []).append(a)

    return graph


# ============================================================
# LOCAL INTERIOR ANGLE
# ============================================================

def vertex_interior_angle(
    vertex: int,
    nodes_m: np.ndarray,
    graph: dict[int, list[int]],
    triangles: np.ndarray | None = None,
    incident_centroid_mean: np.ndarray | None = None,
) -> float | None:
    """
    Determine the physical interior angle.

    The smaller angle between the two boundary rays is first
    calculated. The FEM material sector is then used to decide
    whether the physical interior is that sector or its reflex
    complement.
    """

    neighbors = graph.get(
        int(vertex),
        [],
    )

    if len(neighbors) != 2:

        return None

    p = nodes_m[vertex]

    v1 = (
        nodes_m[neighbors[0]]
        - p
    )

    v2 = (
        nodes_m[neighbors[1]]
        - p
    )

    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)

    if n1 <= 0.0 or n2 <= 0.0:

        return None

    v1 = v1 / n1
    v2 = v2 / n2

    beta = float(
        np.arccos(
            np.clip(
                np.dot(v1, v2),
                -1.0,
                1.0,
            )
        )
    )

    if incident_centroid_mean is not None:
        interior_point = incident_centroid_mean[int(vertex)]
        if not np.all(np.isfinite(interior_point)):
            return beta
    else:
        if triangles is None:
            return beta

        incident = np.where(
            np.any(
                triangles == vertex,
                axis=1,
            )
        )[0]

        if len(incident) == 0:
            return beta

        centroids = np.mean(
            nodes_m[triangles[incident]],
            axis=1,
        )
        interior_point = np.mean(
            centroids,
            axis=0,
        )

    vp = interior_point - p

    norm_vp = np.linalg.norm(vp)

    if norm_vp <= 0.0:

        return beta

    vp /= norm_vp

    bisector = v1 + v2

    norm_b = np.linalg.norm(
        bisector
    )

    if norm_b <= 0.0:

        return beta

    bisector /= norm_b

    if np.dot(
        vp,
        bisector,
    ) > 0.0:

        return beta

    return (
        2.0 * pi
        - beta
    )


# ============================================================
# LOCAL CORNER BASIS
# ============================================================

def corner_local_basis(
    vertex: int,
    nodes_m: np.ndarray,
    graph: dict[int, list[int]],
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Construct a local orthonormal basis at a boundary vertex.

    Returns
    -------
    (tangent, bisector)

    The tangent/bisector basis is used to identify the physical
    corner sector and reject FEM points that are not local to the
    wedge.
    """

    neighbors = graph.get(
        int(vertex),
        [],
    )

    if len(neighbors) != 2:

        return None

    p = nodes_m[vertex]

    rays = []

    for n in neighbors:

        r = (
            nodes_m[n]
            - p
        )

        norm = np.linalg.norm(r)

        if norm <= 0.0:

            return None

        rays.append(
            r / norm
        )

    r1, r2 = rays

    # The sum r1+r2 points into the SMALL sector between the two
    # boundary rays.  For a re-entrant corner the superconducting
    # material occupies the REFLEX sector, so its bisector is the
    # opposite direction.
    small_bisector = r1 + r2

    norm_b = np.linalg.norm(
        small_bisector
    )

    if norm_b <= 0.0:

        return None

    small_bisector /= norm_b

    tangent = r1 - r2

    norm_t = np.linalg.norm(
        tangent
    )

    if norm_t <= 0.0:

        return None

    tangent /= norm_t

    return (
        tangent,
        small_bisector,
    )


# ============================================================
# LOCAL FEM FIT
# ============================================================

def _local_mesh_length(
    corner_xy: np.ndarray,
    nodes_m: np.ndarray,
    triangles: np.ndarray,
    graph: dict[int, list[int]] | None = None,
    vertex: int | None = None,
) -> float:
    """
    Estimate the local FEM length scale near a corner.

    The estimate is based on the edges of triangles incident on the
    corner.  It is used only to determine whether the FEM mesh can
    actually resolve the Clem-Berggren asymptotic region.

    No interpolation or artificial sub-element resolution is introduced.
    """

    if (
        graph is None
        or vertex is None
    ):
        return np.nan

    neighbors = graph.get(int(vertex), [])

    if len(neighbors) != 2:
        return np.nan

    lengths = []

    p = nodes_m[int(vertex)]

    for n in neighbors:
        q = nodes_m[int(n)]
        length = float(np.linalg.norm(q - p))
        if length > 0.0:
            lengths.append(length)

    if not lengths:
        return np.nan

    return float(np.median(lengths))


def _ray_triangle_samples(
    corner_xy: np.ndarray,
    theta: float,
    radii_m: np.ndarray,
    triangulation,
    K_A_per_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Sample the *actual piecewise-constant FEM element field* along one ray.

    Each query point is assigned to the FEM triangle containing that point.
    No nodal reconstruction of J/K is performed.

    Returns
    -------
    r_valid, K_valid, element_ids
    """

    corner_xy = np.asarray(corner_xy, dtype=float)
    radii_m = np.asarray(radii_m, dtype=float)

    points = (
        corner_xy[None, :]
        + radii_m[:, None]
        * np.array(
            [np.cos(theta), np.sin(theta)],
            dtype=float,
        )[None, :]
    )

    finder = triangulation

    element_ids = np.asarray(
        finder(points[:, 0], points[:, 1]),
        dtype=int,
    )

    valid = (
        (element_ids >= 0)
        & (element_ids < len(K_A_per_m))
        & np.isfinite(K_A_per_m[element_ids])
        & (K_A_per_m[element_ids] > 0.0)
    )

    return (
        radii_m[valid],
        np.asarray(
            K_A_per_m[element_ids[valid]],
            dtype=float,
        ),
        element_ids[valid],
    )


def _fixed_exponent_fit(
    r_m: np.ndarray,
    K_A_per_m: np.ndarray,
    p: float,
) -> tuple[float, float, float, float]:
    """
    Fit

        K(r) = A r^p

    with the Clem-Berggren wedge exponent p FIXED by geometry.

    Only A is fitted:

        ln A = mean[ln K - p ln r]

    This is important: the FEM data are not permitted to change the
    theoretical wedge exponent.  The fit therefore tests whether the
    numerical field is actually compatible with the required asymptotic
    law rather than fitting away the physics.

    Returns
    -------
    K0, R2_fixed, fitted_slope, intercept
    """

    r = np.asarray(r_m, dtype=float)
    K = np.asarray(K_A_per_m, dtype=float)

    if len(r) < 2:
        return np.nan, np.nan, np.nan, np.nan

    x = np.log(r)
    y = np.log(K)

    log_K0_samples = y - p * x

    intercept = float(np.mean(log_K0_samples))
    K0 = float(np.exp(intercept))

    y_pred_fixed = intercept + p * x

    ss_res = float(
        np.sum((y - y_pred_fixed) ** 2)
    )

    ss_tot = float(
        np.sum((y - np.mean(y)) ** 2)
    )

    if ss_tot <= 0.0:
        r2 = 0.0
    else:
        r2 = float(
            1.0 - ss_res / ss_tot
        )

    # Diagnostic unconstrained slope only.
    A = np.column_stack(
        [
            np.ones_like(x),
            x,
        ]
    )

    coefficients, *_ = np.linalg.lstsq(
        A,
        y,
        rcond=None,
    )

    fitted_slope = float(coefficients[1])

    return (
        K0,
        r2,
        fitted_slope,
        intercept,
    )


def _physical_sector_angles(
    vertex: int,
    nodes_m: np.ndarray,
    graph: dict[int, list[int]],
    interior_angle_rad: float,
    number_of_rays: int,
) -> np.ndarray:
    """
    Generate rays only inside the *physical re-entrant sector*.

    The two boundary rays define two complementary angular sectors.
    For a re-entrant corner, the superconducting material occupies the
    sector whose angular span equals the supplied interior angle alpha.

    This avoids the previous full-2*pi ray sweep.  The FEM triangulation
    remains the authority for point-in-material tests, so this is only a
    reduction of candidate directions, not an approximation to J/K.
    """
    neighbors = graph.get(int(vertex), [])
    if len(neighbors) != 2:
        return np.array([], dtype=float)

    p0 = nodes_m[int(vertex)]
    rays = []
    for n in neighbors:
        r = nodes_m[int(n)] - p0
        nr = np.linalg.norm(r)
        if nr <= 0.0:
            return np.array([], dtype=float)
        rays.append(r / nr)

    a1 = float(np.arctan2(rays[0][1], rays[0][0]))
    a2 = float(np.arctan2(rays[1][1], rays[1][0]))
    two_pi = 2.0 * pi

    ccw = (a2 - a1) % two_pi
    cw = two_pi - ccw

    # Choose the boundary-ray ordering whose sector span is alpha.
    if abs(ccw - interior_angle_rad) <= abs(cw - interior_angle_rad):
        start = a1
        span = ccw
    else:
        start = a2
        span = cw

    # Keep rays away from the mathematical boundary itself.  The endpoint
    # exclusion prevents sampling exactly on a boundary edge where the
    # triangulation finder can return an ambiguous neighboring element.
    n = max(8, int(number_of_rays))
    margin = min(0.02, 0.1 * span / max(n, 1))
    lo = start + margin
    hi = start + span - margin

    if hi <= lo:
        return np.array([], dtype=float)

    return np.linspace(lo, hi, n, endpoint=True)


def fit_corner_current(
    corner_xy: np.ndarray,
    triangle_centers_m: np.ndarray,
    K_A_per_m: np.ndarray,
    p: float,
    xi_m: float,
    wire_width_m: float,
    interior_angle_rad: float,
    graph: dict[int, list[int]] | None = None,
    vertex: int | None = None,
    nodes_m: np.ndarray | None = None,
    triangles: np.ndarray | None = None,
    minimum_fit_points: int = 8,
    minimum_fit_r2: float = 0.80,
    fit_r_min_factor: float = 3.0,
    maximum_fit_width_fraction: float = 0.25,
    number_of_rays: int = 25,
    trifinder=None,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[
    float,
    int,
    float,
    float,
    float,
    str,
]:
    """
    Determine the local Clem-Berggren coefficient from the FEM field.

    The previous implementation fitted arbitrary triangle-centre points
    in a 2-D annulus.  That mixes the radial wedge law with its angular
    dependence and, more importantly, can use a mesh whose element size
    is larger than the entire asymptotic fitting region.

    This implementation instead:

      1. checks the local FEM length scale;
      2. constructs the physical wedge sector from the actual boundary;
      3. samples rays through the actual FEM triangulation;
      4. keeps only points belonging to superconducting FEM elements;
      5. fits K(r) = K0 r^p with p fixed by the Clem-Berggren geometry;
      6. selects the best-resolved ray;
      7. rejects the corner if the FEM does not resolve the asymptotic
         region.

    This is deliberately conservative.  It never manufactures
    sub-element current-density information.
    """

    if (
        graph is None
        or vertex is None
        or nodes_m is None
        or triangles is None
    ):
        return (
            np.nan,
            0,
            np.nan,
            np.nan,
            np.nan,
            "mesh topology is required for ray-resolved fit",
        )

    nodes_m = np.asarray(nodes_m, dtype=float)
    triangles = np.asarray(triangles, dtype=np.int64)

    if triangles.ndim != 2 or triangles.shape[1] != 3:
        return (
            np.nan,
            0,
            np.nan,
            np.nan,
            np.nan,
            "invalid FEM triangle connectivity",
        )

    local_h = _local_mesh_length(
        corner_xy=corner_xy,
        nodes_m=nodes_m,
        triangles=triangles,
        graph=graph,
        vertex=vertex,
    )

    r_min_physics = max(
        fit_r_min_factor * xi_m,
        1.0e-12,
    )

    r_max = (
        maximum_fit_width_fraction
        * wire_width_m
    )

    # A pointwise asymptotic fit is meaningful only when multiple FEM
    # elements exist between the core-exclusion radius and the outer
    # fitting radius.  The factor 0.5 is deliberately conservative:
    # at least roughly two local element lengths should fit in the
    # radial fitting interval.
    if np.isfinite(local_h):
        if r_max <= 2.0 * local_h:
            return (
                np.nan,
                0,
                np.nan,
                np.nan,
                np.nan,
                (
                    "FEM mesh is too coarse for the selected "
                    "Clem-Berggren asymptotic window: "
                    f"r_max={r_max*1e9:.3f} nm, "
                    f"local_h={local_h*1e9:.3f} nm"
                ),
            )

        r_min = max(
            r_min_physics,
            1.25 * local_h,
        )
    else:
        r_min = r_min_physics

    if r_min >= r_max:
        return (
            np.nan,
            0,
            np.nan,
            np.nan,
            np.nan,
            (
                "Clem-Berggren fitting interval is empty: "
                f"r_min={r_min*1e9:.3f} nm, "
                f"r_max={r_max*1e9:.3f} nm"
            ),
        )

    # The exact FEM triangulation is normally constructed once by the
    # top-level solver and its trifinder is reused for every corner.
    # This avoids rebuilding the spatial search structure thousands of
    # times while preserving the exact piecewise-constant FEM field.
    if trifinder is None:
        triangulation = mtri.Triangulation(
            nodes_m[:, 0],
            nodes_m[:, 1],
            triangles,
        )
        trifinder = triangulation.get_trifinder()

    # Log-spaced radial samples resolve the near-corner region better
    # than linear spacing.
    radii = np.geomspace(
        r_min,
        r_max,
        24,
    )

    candidate_angles = _physical_sector_angles(
        vertex=vertex,
        nodes_m=nodes_m,
        graph=graph,
        interior_angle_rad=interior_angle_rad,
        number_of_rays=number_of_rays,
    )

    best = None

    for theta in candidate_angles:

        r, k, element_ids = _ray_triangle_samples(
            corner_xy=corner_xy,
            theta=float(theta),
            radii_m=radii,
            triangulation=trifinder,
            K_A_per_m=K_A_per_m,
        )

        if len(r) < minimum_fit_points:
            continue

        # Remove repeated samples that hit the same FEM element.
        # Keeping one representative radius per element prevents a
        # piecewise-constant FEM element from being counted repeatedly.
        unique_elements, unique_indices = np.unique(
            element_ids,
            return_index=True,
        )

        if len(unique_elements) < minimum_fit_points:
            continue

        order = np.sort(unique_indices)

        r_unique = r[order]
        k_unique = k[order]

        (
            K0,
            r2,
            fitted_slope,
            intercept,
        ) = _fixed_exponent_fit(
            r_unique,
            k_unique,
            p,
        )

        if not np.isfinite(r2):
            continue

        candidate = (
            r2,
            K0,
            len(r_unique),
            fitted_slope,
            intercept,
            r_unique,
            k_unique,
        )

        if (
            best is None
            or candidate[0] > best[0]
        ):
            best = candidate

    if best is None:
        return (
            np.nan,
            0,
            np.nan,
            np.nan,
            np.nan,
            (
                "FEM mesh does not provide enough distinct "
                "resolved elements along any physical corner ray"
            ),
        )

    (
        best_r2,
        K0,
        n_points,
        fitted_slope,
        intercept,
        _,
        _,
    ) = best

    if best_r2 < minimum_fit_r2:
        return (
            K0,
            n_points,
            best_r2,
            fitted_slope,
            intercept,
            (
                "poor fixed-exponent local fit "
                f"(R²={best_r2:.4f}); "
                "refine the corner mesh rather than lowering "
                "the physical acceptance criterion"
            ),
        )

    return (
        K0,
        n_points,
        best_r2,
        fitted_slope,
        intercept,
        "accepted",
    )


# ============================================================
# MAIN SOLVER
# ============================================================

def analyze_clem_berggren(
    nodes_m: np.ndarray,
    triangles: np.ndarray,
    triangle_centers_m: np.ndarray,
    element_J_magnitude_A_per_m2: np.ndarray,
    fem_current_A: float,
    params: ClemBerggrenParameters,
    progress_callback: Callable[[int, int, str], None] | None = None,
) -> ClemBerggrenResult:
    """
    Perform FEM + Clem-Berggren critical-current analysis.
    """

    if fem_current_A <= 0.0:

        raise ValueError(
            "FEM current must be positive."
        )

    nodes_m = np.asarray(
        nodes_m,
        dtype=float,
    )

    triangles = np.asarray(
        triangles,
        dtype=np.int64,
    )

    triangle_centers_m = np.asarray(
        triangle_centers_m,
        dtype=float,
    )

    J = np.asarray(
        element_J_magnitude_A_per_m2,
        dtype=float,
    )

    if len(J) != len(
        triangle_centers_m
    ):

        raise ValueError(
            "Current-density array and triangle-center "
            "array have different lengths."
        )

    # ========================================================
    # SHEET CURRENT
    # ========================================================

    K = (
        J
        * params.film_thickness_m
    )

    # ========================================================
    # STRAIGHT STRIP
    # ========================================================

    Kc_straight = (
        straight_strip_critical_sheet_current(
            params
        )
    )

    Ic_straight = (
        Kc_straight
        * params.wire_width_m
    )

    # ========================================================
    # VALIDITY CONDITIONS
    # ========================================================

    validity_d_over_lambda = (
        params.thickness_to_lambda_ratio
        < 0.1
    )

    validity_w_over_lambda = (
        params.width_to_pearl_ratio
        < 0.1
    )

    validity_xi_over_w = (
        params.coherence_to_width_ratio
        < 0.1
    )

    # ========================================================
    # BOUNDARY
    # ========================================================

    boundary_edges = (
        boundary_edges_from_triangles(
            triangles
        )
    )

    graph = (
        boundary_vertex_neighbors(
            boundary_edges
        )
    )

    # --------------------------------------------------------
    # Precompute incident-triangle centroid means.
    # vertex_interior_angle previously searched the entire triangle
    # array for every boundary vertex, which is O(N_boundary*N_tri).
    # This one vectorized pass reduces that part to O(N_tri).
    # --------------------------------------------------------
    triangle_centroids = np.asarray(
        triangle_centers_m,
        dtype=float,
    )

    incident_sum = np.zeros_like(nodes_m, dtype=float)
    incident_count = np.zeros(len(nodes_m), dtype=np.int64)

    for local_vertex in range(3):
        ids = triangles[:, local_vertex]
        np.add.at(incident_sum, ids, triangle_centroids)
        np.add.at(incident_count, ids, 1)

    incident_centroid_mean = np.full_like(nodes_m, np.nan, dtype=float)
    valid_incident = incident_count > 0
    incident_centroid_mean[valid_incident] = (
        incident_sum[valid_incident]
        / incident_count[valid_incident, None]
    )

    # Construct the FEM spatial locator once and reuse it for every ray
    # of every corner.  Rebuilding Triangulation/get_trifinder inside each
    # corner was one of the dominant runtime costs.
    triangulation = mtri.Triangulation(
        nodes_m[:, 0],
        nodes_m[:, 1],
        triangles,
    )
    trifinder = triangulation.get_trifinder()

    # ========================================================
    # CORNER ANALYSIS
    # ========================================================

    # Identify the actual re-entrant vertices first.  This lets the
    # progress display report meaningful progress over the expensive
    # operations only.
    boundary_vertices = sorted(graph)
    corner_vertices: list[tuple[int, float]] = []

    for vertex in boundary_vertices:
        alpha = vertex_interior_angle(
            vertex,
            nodes_m,
            graph,
            triangles,
            incident_centroid_mean,
        )
        if (
            alpha is not None
            and np.degrees(alpha)
            > 180.0 + params.minimum_reentrant_angle_excess_deg
        ):
            corner_vertices.append((int(vertex), float(alpha)))

    total_corners = len(corner_vertices)
    corners: list[CornerResult] = []

    def _emit_progress(done: int, message: str) -> None:
        if progress_callback is not None:
            progress_callback(done, total_corners, message)
            return

        width = 32
        frac = done / max(total_corners, 1)
        filled = int(width * frac)
        bar = "=" * filled + ">" + " " * max(0, width - filled - 1)
        print(
            f"\rClem-Berggren corners [{bar}] "
            f"{done}/{total_corners}  {message:<24}",
            end="",
            flush=True,
        )

    if total_corners == 0:
        raise RuntimeError(
            "No re-entrant FEM boundary corners were detected."
        )

    for corner_number, (vertex, alpha) in enumerate(
        corner_vertices,
        start=1,
    ):
        p = wedge_exponent(alpha)

        (
            K0_ref,
            nfit,
            r2,
            fitted_slope,
            intercept,
            fit_status,
        ) = fit_corner_current(
            corner_xy=nodes_m[vertex],
            triangle_centers_m=triangle_centers_m,
            K_A_per_m=K,
            p=p,
            xi_m=params.coherence_length_m,
            wire_width_m=params.wire_width_m,
            interior_angle_rad=alpha,
            graph=graph,
            vertex=vertex,
            nodes_m=nodes_m,
            triangles=triangles,
            minimum_fit_points=params.minimum_fit_points,
            minimum_fit_r2=params.minimum_fit_r2,
            fit_r_min_factor=params.fit_r_min_factor,
            maximum_fit_width_fraction=(
                params.maximum_fit_width_fraction
            ),
            number_of_rays=params.number_of_rays,
            trifinder=trifinder,
        )

        K0_critical = wedge_critical_K0(
            params,
            alpha,
        )

        delta_c = critical_barrier_distance(
            params,
            alpha,
        )

        accepted = (
            np.isfinite(K0_ref)
            and K0_ref > 0.0
            and np.isfinite(r2)
            and r2 >= params.minimum_fit_r2
        )

        if accepted:
            Ic = (
                fem_current_A
                * K0_critical
                / K0_ref
            )
            rejection_reason = ""
        else:
            Ic = np.inf
            rejection_reason = fit_status

        if np.isfinite(K0_ref) and K0_ref > 0.0:
            delta_barrier = (
                PHI_0
                /
                (
                    2.0
                    * pi
                    * MU_0
                    * params.pearl_length_m
                    * K0_ref
                )
            )
        else:
            delta_barrier = np.nan

        corners.append(
            CornerResult(
                vertex_index=int(vertex),
                x_m=float(nodes_m[vertex, 0]),
                y_m=float(nodes_m[vertex, 1]),
                interior_angle_rad=float(alpha),
                interior_angle_deg=float(np.degrees(alpha)),
                exponent_p=float(p),
                fit_points=int(nfit),
                fit_r2=float(r2),
                fit_slope=float(fitted_slope),
                fit_intercept=float(intercept),
                K0_reference_A_per_m_power=float(K0_ref),
                K0_critical_A_per_m_power=float(K0_critical),
                delta_barrier_m=float(delta_barrier),
                delta_critical_m=float(delta_c),
                critical_current_A=float(Ic),
                accepted=bool(accepted),
                rejection_reason=str(rejection_reason),
            )
        )

        _emit_progress(
            corner_number,
            f"vertex {vertex}",
        )

    if progress_callback is None:
        print()

    # ========================================================
    # ACCEPTED CORNERS
    # ========================================================

    accepted_corners = [
        c
        for c in corners
        if c.accepted
        and np.isfinite(
            c.critical_current_A
        )
    ]

    if len(accepted_corners) == 0:

        raise RuntimeError(
            "No re-entrant corner produced a valid "
            "Clem-Berggren local fit. "
            "Do NOT interpret the result until mesh "
            "resolution and corner geometry are checked."
        )

    limiting = min(
        accepted_corners,
        key=lambda c:
        c.critical_current_A,
    )

    return ClemBerggrenResult(
        critical_current_A=float(
            limiting.critical_current_A
        ),
        straight_strip_critical_current_A=float(
            Ic_straight
        ),
        straight_strip_critical_sheet_current_A_per_m=float(
            Kc_straight
        ),
        pearl_length_m=float(
            params.pearl_length_m
        ),
        penetration_depth_m=float(
            params.penetration_depth_m
        ),
        coherence_length_m=float(
            params.coherence_length_m
        ),
        width_to_pearl_ratio=float(
            params.width_to_pearl_ratio
        ),
        coherence_to_width_ratio=float(
            params.coherence_to_width_ratio
        ),
        thickness_to_lambda_ratio=float(
            params.thickness_to_lambda_ratio
        ),
        limiting_x_m=float(
            limiting.x_m
        ),
        limiting_y_m=float(
            limiting.y_m
        ),
        limiting_angle_deg=float(
            limiting.interior_angle_deg
        ),
        limiting_current_A=float(
            limiting.critical_current_A
        ),
        limiting_K0_reference_A_per_m_power=float(
            limiting.K0_reference_A_per_m_power
        ),
        limiting_K0_critical_A_per_m_power=float(
            limiting.K0_critical_A_per_m_power
        ),
        corners=tuple(
            corners
        ),
        validity_w_over_lambda=bool(
            validity_w_over_lambda
        ),
        validity_xi_over_w=bool(
            validity_xi_over_w
        ),
        validity_d_over_lambda=bool(
            validity_d_over_lambda
        ),
    )


# ============================================================
# REPORT
# ============================================================

def format_result(
    result: ClemBerggrenResult,
) -> str:

    reduction = (
        1.0
        -
        result.critical_current_A
        /
        result.straight_strip_critical_current_A
    ) * 100.0

    lines = []

    lines.append(
        "\n"
        "========================================================\n"
        "SNSPD CLEM-BERGREN CRITICAL CURRENT ANALYSIS\n"
        "========================================================"
    )

    lines.append(
        "\n"
        "SUPERCONDUCTING LENGTH SCALES\n"
        "-----------------------------"
    )

    lines.append(
        f"Penetration depth lambda : "
        f"{result.penetration_depth_m * 1e9:.6f} nm"
    )

    lines.append(
        f"Coherence length xi      : "
        f"{result.coherence_length_m * 1e9:.6f} nm"
    )

    lines.append(
        f"Pearl length Lambda      : "
        f"{result.pearl_length_m * 1e6:.6f} um"
    )

    lines.append(
        f"W / Lambda               : "
        f"{result.width_to_pearl_ratio:.6e}"
    )

    lines.append(
        f"xi / W                   : "
        f"{result.coherence_to_width_ratio:.6e}"
    )

    lines.append(
        f"d / lambda               : "
        f"{result.thickness_to_lambda_ratio:.6e}"
    )

    lines.append(
        "\n"
        "MODEL VALIDITY\n"
        "--------------"
    )

    lines.append(
        f"d << lambda              : "
        f"{'PASS' if result.validity_d_over_lambda else 'CHECK'}"
    )

    lines.append(
        f"W << Lambda              : "
        f"{'PASS' if result.validity_w_over_lambda else 'CHECK'}"
    )

    lines.append(
        f"xi << W                  : "
        f"{'PASS' if result.validity_xi_over_w else 'CHECK'}"
    )

    lines.append(
        "\n"
        "STRAIGHT STRIP REFERENCE\n"
        "------------------------"
    )

    lines.append(
        f"Critical sheet current   : "
        f"{result.straight_strip_critical_sheet_current_A_per_m:.6e} A/m"
    )

    lines.append(
        f"Ic,straight              : "
        f"{result.straight_strip_critical_current_A:.6e} A"
    )

    lines.append(
        f"Ic,straight              : "
        f"{result.straight_strip_critical_current_A * 1e6:.6f} uA"
    )

    lines.append(
        "\n"
        "CLEM-BERGREN DEVICE RESULT\n"
        "--------------------------"
    )

    lines.append(
        f"Device critical current  : "
        f"{result.critical_current_A:.6e} A"
    )

    lines.append(
        f"Device critical current  : "
        f"{result.critical_current_A * 1e6:.6f} uA"
    )

    lines.append(
        f"Critical-current reduction: "
        f"{reduction:.6f} %"
    )

    lines.append(
        "\n"
        "LIMITING LOCATION\n"
        "-----------------"
    )

    lines.append(
        f"x                        : "
        f"{result.limiting_x_m * 1e6:.6f} um"
    )

    lines.append(
        f"y                        : "
        f"{result.limiting_y_m * 1e6:.6f} um"
    )

    lines.append(
        f"Interior angle           : "
        f"{result.limiting_angle_deg:.6f} deg"
    )

    lines.append(
        f"K0 reference             : "
        f"{result.limiting_K0_reference_A_per_m_power:.6e} A/m"
    )

    lines.append(
        f"K0 critical              : "
        f"{result.limiting_K0_critical_A_per_m_power:.6e} A/m"
    )

    lines.append(
        "\n"
        "CORNER ANALYSIS\n"
        "---------------"
    )

    for c in sorted(
        result.corners,
        key=lambda item:
        item.critical_current_A,
    ):

        lines.append(
            f"\nVertex {c.vertex_index}"
        )

        lines.append(
            f"    position       : "
            f"({c.x_m * 1e6:.6f}, "
            f"{c.y_m * 1e6:.6f}) um"
        )

        lines.append(
            f"    angle          : "
            f"{c.interior_angle_deg:.6f} deg"
        )

        lines.append(
            f"    exponent p     : "
            f"{c.exponent_p:.6f}"
        )

        lines.append(
            f"    fit points     : "
            f"{c.fit_points}"
        )

        lines.append(
            f"    theoretical p  : "
            f"{c.exponent_p:.6f}"
        )

        lines.append(
            f"    fitted slope   : "
            f"{c.fit_slope:.6f}"
        )

        lines.append(
            f"    fit R²         : "
            f"{c.fit_r2:.6f}"
        )

        lines.append(
            f"    K0 reference   : "
            f"{c.K0_reference_A_per_m_power:.6e} A/m"
        )

        lines.append(
            f"    K0 critical    : "
            f"{c.K0_critical_A_per_m_power:.6e} A/m"
        )

        if c.accepted:

            lines.append(
                f"    Ic             : "
                f"{c.critical_current_A * 1e6:.6f} uA"
            )

            lines.append(
                "    status         : ACCEPTED"
            )

        else:

            lines.append(
                "    Ic             : INVALID"
            )

            lines.append(
                f"    status         : "
                f"REJECTED ({c.rejection_reason})"
            )

    lines.append(
        "\n"
        "PHYSICS\n"
        "-------"
    )

    lines.append(
        "K = d J"
    )

    lines.append(
        "Lambda = 2 lambda^2 / d"
    )

    lines.append(
        "K(r,theta) = A(theta) r^p"
    )

    lines.append(
        "p = pi/alpha - 1"
    )

    lines.append(
        "G(delta) = E_self - W_I"
    )

    lines.append(
        "Critical current defined by vanishing Gibbs barrier"
    )

    lines.append(
        "\n"
        "Clem-Berggren analysis : PASS"
    )

    return "\n".join(lines)