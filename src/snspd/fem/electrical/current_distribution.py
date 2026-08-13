# FILE:
# src/snspd/fem/electrical/current_distribution.py
#
# PURPOSE:
# General stationary thin-film electrical FEM solver for SNSPDs.
#
# Governing equation:
#
#     -div(sigma * grad(V)) = q
#
# with:
#
#     J = -sigma * grad(V)
#
# and therefore:
#
#     div(J) = q
#
# For the source-free transport problem:
#
#     q = 0
#
# Thin-film current:
#
#     I = integral_boundary (J . n) t ds
#
# The FEM formulation uses:
#
#     K_ij = t * integral_Omega sigma grad(N_i).grad(N_j) dA
#
# and:
#
#     F_i = t * integral_Omega q N_i dA
#
# Terminal currents are calculated from the FEM reaction vector.
# This is the discrete boundary flux corresponding directly to
# the assembled weak form and is therefore the preferred
# conservation-consistent terminal-current calculation.


from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from scipy.sparse import (
    lil_matrix,
    csr_matrix,
)

from scipy.sparse.linalg import spsolve


# ============================================================
# RESULT
# ============================================================


@dataclass(frozen=True)
class CurrentDistributionResult:

    potential: np.ndarray

    nodal_current_density: np.ndarray

    nodal_current_density_magnitude: np.ndarray

    element_current_density: np.ndarray

    element_current_density_magnitude: np.ndarray

    total_current: float

    positive_terminal_current: float

    negative_terminal_current: float

    maximum_current_density: float

    maximum_element_index: int


# ============================================================
# GEOMETRY UTILITIES
# ============================================================


def triangle_area(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
) -> float:
    """
    Return the absolute area of a triangle.
    """

    determinant = (
        (p2[0] - p1[0])
        * (p3[1] - p1[1])
        -
        (p3[0] - p1[0])
        * (p2[1] - p1[1])
    )

    return 0.5 * abs(determinant)


def triangle_gradient_matrix(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
) -> np.ndarray:
    """
    Return the P1 triangular shape-function gradient matrix.

    B[:, i] = grad(N_i)

    Shape:
        (2, 3)
    """

    twice_area = (
        (p2[0] - p1[0])
        * (p3[1] - p1[1])
        -
        (p3[0] - p1[0])
        * (p2[1] - p1[1])
    )

    if abs(twice_area) < 1.0e-30:

        raise ValueError(
            "Degenerate triangular FEM element encountered."
        )

    return np.array(
        [
            [
                (p2[1] - p3[1]) / twice_area,
                (p3[1] - p1[1]) / twice_area,
                (p1[1] - p2[1]) / twice_area,
            ],
            [
                (p3[0] - p2[0]) / twice_area,
                (p1[0] - p3[0]) / twice_area,
                (p2[0] - p1[0]) / twice_area,
            ],
        ],
        dtype=float,
    )


# ============================================================
# TRIANGLE QUADRATURE
# ============================================================


def triangle_quadrature(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
) -> list[tuple[np.ndarray, float]]:
    """
    Three-point symmetric Gaussian quadrature rule on a
    triangle.

    Exact for polynomials up to degree 2.

    Each physical quadrature weight is:

        A / 3
    """

    area = triangle_area(
        p1,
        p2,
        p3,
    )

    points = [

        p1
        + (1.0 / 6.0)
        * (p2 - p1)
        + (1.0 / 6.0)
        * (p3 - p1),

        p1
        + (2.0 / 3.0)
        * (p2 - p1)
        + (1.0 / 6.0)
        * (p3 - p1),

        p1
        + (1.0 / 6.0)
        * (p2 - p1)
        + (2.0 / 3.0)
        * (p3 - p1),
    ]

    weight = area / 3.0

    return [
        (
            np.asarray(
                point,
                dtype=float,
            ),
            weight,
        )
        for point in points
    ]


# ============================================================
# SOLVER
# ============================================================


class CurrentDistributionSolver:
    """
    General stationary electrical FEM solver.

    Solves:

        -div(sigma * grad(V)) = q

    for a thin conducting film.

    Parameters
    ----------
    nodes:
        Nx2 array of coordinates [m].

    triangles:
        Mx3 triangular connectivity.

    conductivity:
        Electrical conductivity [S/m].

    thickness:
        Film thickness [m].

    source:
        Optional volumetric source q(x,y) [A/m^3].
    """

    def __init__(
        self,
        nodes: np.ndarray,
        triangles: np.ndarray,
        conductivity: float,
        thickness: float,
        source: Callable[
            [float, float],
            float,
        ] | None = None,
    ) -> None:

        self.nodes = np.asarray(
            nodes,
            dtype=float,
        )

        self.triangles = np.asarray(
            triangles,
            dtype=int,
        )

        self.conductivity = float(
            conductivity
        )

        self.thickness = float(
            thickness
        )

        self.source = source

        if (
            self.nodes.ndim != 2
            or self.nodes.shape[1] != 2
        ):

            raise ValueError(
                "nodes must have shape (N, 2)."
            )

        if (
            self.triangles.ndim != 2
            or self.triangles.shape[1] != 3
        ):

            raise ValueError(
                "triangles must have shape (M, 3)."
            )

        if self.conductivity <= 0.0:

            raise ValueError(
                "conductivity must be positive."
            )

        if self.thickness <= 0.0:

            raise ValueError(
                "thickness must be positive."
            )


    # ========================================================
    # FEM ASSEMBLY
    # ========================================================


    def assemble_system(
        self,
    ) -> tuple[csr_matrix, np.ndarray]:
        """
        Assemble the global FEM matrix and load vector.

        Weak form:

            integral sigma grad(V).grad(w) dA
            =
            integral q w dA

        with film thickness included.
        """

        node_count = len(
            self.nodes
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

        for triangle in self.triangles:

            indices = np.asarray(
                triangle,
                dtype=int,
            )

            p1 = self.nodes[
                indices[0]
            ]

            p2 = self.nodes[
                indices[1]
            ]

            p3 = self.nodes[
                indices[2]
            ]

            area = triangle_area(
                p1,
                p2,
                p3,
            )

            if area <= 0.0:

                raise ValueError(
                    "Mesh contains a zero-area triangle."
                )

            B = triangle_gradient_matrix(
                p1,
                p2,
                p3,
            )

            # ------------------------------------------------
            # Conductivity matrix
            # ------------------------------------------------

            local_K = (
                self.conductivity
                * self.thickness
                * area
                * (
                    B.T @ B
                )
            )

            for local_i in range(3):

                global_i = int(
                    indices[local_i]
                )

                for local_j in range(3):

                    global_j = int(
                        indices[local_j]
                    )

                    K[
                        global_i,
                        global_j,
                    ] += local_K[
                        local_i,
                        local_j
                    ]

            # ------------------------------------------------
            # Source term
            # ------------------------------------------------

            if self.source is not None:

                local_F = np.zeros(
                    3,
                    dtype=float,
                )

                for point, weight in (
                    triangle_quadrature(
                        p1,
                        p2,
                        p3,
                    )
                ):

                    x = point[0]
                    y = point[1]

                    q_value = float(
                        self.source(
                            x,
                            y,
                        )
                    )

                    denominator = (
                        (
                            p2[1]
                            - p3[1]
                        )
                        * (
                            p1[0]
                            - p3[0]
                        )
                        +
                        (
                            p3[0]
                            - p2[0]
                        )
                        * (
                            p1[1]
                            - p3[1]
                        )
                    )

                    N1 = (
                        (
                            p2[1]
                            - p3[1]
                        )
                        * (
                            x
                            - p3[0]
                        )
                        +
                        (
                            p3[0]
                            - p2[0]
                        )
                        * (
                            y
                            - p3[1]
                        )
                    ) / denominator

                    N2 = (
                        (
                            p3[1]
                            - p1[1]
                        )
                        * (
                            x
                            - p3[0]
                        )
                        +
                        (
                            p1[0]
                            - p3[0]
                        )
                        * (
                            y
                            - p3[1]
                        )
                    ) / denominator

                    N3 = (
                        1.0
                        - N1
                        - N2
                    )

                    shape_functions = np.array(
                        [
                            N1,
                            N2,
                            N3,
                        ],
                        dtype=float,
                    )

                    local_F += (
                        self.thickness
                        * weight
                        * q_value
                        * shape_functions
                    )

                for local_i in range(3):

                    global_i = int(
                        indices[local_i]
                    )

                    F[
                        global_i
                    ] += local_F[
                        local_i
                    ]

        return (
            K.tocsr(),
            F,
        )


    # ========================================================
    # DIRICHLET CONDITIONS
    # ========================================================


    @staticmethod
    def apply_dirichlet(
        K: csr_matrix,
        rhs: np.ndarray,
        prescribed_values: dict[int, float],
    ) -> tuple[
        csr_matrix,
        np.ndarray,
    ]:

        K_mod = K.tolil()

        rhs_mod = np.asarray(
            rhs,
            dtype=float,
        ).copy()

        for node, value in (
            prescribed_values.items()
        ):

            node = int(node)
            value = float(value)

            # Eliminate the prescribed degree of freedom
            # from all other equations.
            for row in range(
                K_mod.shape[0]
            ):

                if row == node:
                    continue

                coefficient = K_mod[
                    row,
                    node
                ]

                if coefficient != 0.0:

                    rhs_mod[row] -= (
                        coefficient
                        * value
                    )

                    K_mod[
                        row,
                        node
                    ] = 0.0

            K_mod[
                node,
                :
            ] = 0.0

            K_mod[
                :,
                node
            ] = 0.0

            K_mod[
                node,
                node
            ] = 1.0

            rhs_mod[
                node
            ] = value

        return (
            K_mod.tocsr(),
            rhs_mod,
        )


    # ========================================================
    # POTENTIAL SOLUTION
    # ========================================================


    def solve_potential(
        self,
        positive_terminal_nodes: np.ndarray,
        negative_terminal_nodes: np.ndarray,
        voltage_difference: float = 1.0,
    ) -> np.ndarray:

        K, F = (
            self.assemble_system()
        )

        prescribed_values = {}

        for node in positive_terminal_nodes:

            prescribed_values[
                int(node)
            ] = float(
                voltage_difference
            )

        for node in negative_terminal_nodes:

            prescribed_values[
                int(node)
            ] = 0.0

        K_bc, F_bc = (
            self.apply_dirichlet(
                K,
                F,
                prescribed_values,
            )
        )

        return spsolve(
            K_bc,
            F_bc,
        )


    # ========================================================
    # ELEMENT CURRENT DENSITY
    # ========================================================


    def calculate_element_current_density(
        self,
        potential: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate element-wise:

            J = -sigma grad(V)

        Units:
            A/m²
        """

        element_count = len(
            self.triangles
        )

        current_density = np.zeros(
            (
                element_count,
                2,
            ),
            dtype=float,
        )

        for element_index, triangle in enumerate(
            self.triangles
        ):

            indices = np.asarray(
                triangle,
                dtype=int,
            )

            p1 = self.nodes[
                indices[0]
            ]

            p2 = self.nodes[
                indices[1]
            ]

            p3 = self.nodes[
                indices[2]
            ]

            B = triangle_gradient_matrix(
                p1,
                p2,
                p3,
            )

            local_potential = potential[
                indices
            ]

            grad_V = (
                B
                @ local_potential
            )

            current_density[
                element_index
            ] = (
                -self.conductivity
                * grad_V
            )

        return current_density


    # ========================================================
    # CURRENT MAGNITUDE
    # ========================================================


    @staticmethod
    def current_density_magnitude(
        current_density: np.ndarray,
    ) -> np.ndarray:

        return np.linalg.norm(
            current_density,
            axis=1,
        )


    # ========================================================
    # NODAL RECOVERY
    # ========================================================


    def calculate_nodal_current_density(
        self,
        element_current_density: np.ndarray,
    ) -> np.ndarray:

        node_count = len(
            self.nodes
        )

        nodal_current = np.zeros(
            (
                node_count,
                2,
            ),
            dtype=float,
        )

        nodal_weight = np.zeros(
            node_count,
            dtype=float,
        )

        for element_index, triangle in enumerate(
            self.triangles
        ):

            indices = np.asarray(
                triangle,
                dtype=int,
            )

            p1 = self.nodes[
                indices[0]
            ]

            p2 = self.nodes[
                indices[1]
            ]

            p3 = self.nodes[
                indices[2]
            ]

            area = triangle_area(
                p1,
                p2,
                p3,
            )

            for node in indices:

                node = int(node)

                nodal_current[
                    node
                ] += (
                    area
                    * element_current_density[
                        element_index
                    ]
                )

                nodal_weight[
                    node
                ] += area

        valid = (
            nodal_weight > 0.0
        )

        nodal_current[
            valid
        ] /= nodal_weight[
            valid,
            None
        ]

        return nodal_current


    # ========================================================
    # FEM REACTION VECTOR
    # ========================================================


    def calculate_reaction_vector(
        self,
        potential: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate the FEM reaction vector.

        The unconstrained discrete system is:

            K V = F + boundary reaction contribution

        Therefore:

            R = K V - F

        at prescribed Dirichlet degrees of freedom.

        For the present sign convention,

            R_i = - integral_boundary_i (J . n) t ds

        so the magnitude of the summed reaction on a terminal
        gives the physical terminal current.
        """

        K, F = (
            self.assemble_system()
        )

        return (
            K @ potential
            - F
        )


    # ========================================================
    # TERMINAL CURRENT FROM FEM REACTIONS
    # ========================================================


    def calculate_terminal_current(
        self,
        potential: np.ndarray,
        terminal_nodes: np.ndarray,
    ) -> float:
        """
        Calculate terminal current from the FEM reaction vector.

        This is the discrete boundary flux corresponding
        directly to the assembled weak form.

        Returns
        -------
        float
            Terminal current magnitude [A].
        """

        reaction = (
            self.calculate_reaction_vector(
                potential
            )
        )

        terminal_indices = np.asarray(
            terminal_nodes,
            dtype=int,
        )

        terminal_reaction = float(
            np.sum(
                reaction[
                    terminal_indices
                ]
            )
        )

        return abs(
            terminal_reaction
        )


    # ========================================================
    # SIGNED TERMINAL CURRENT
    # ========================================================


    def calculate_signed_terminal_current(
        self,
        potential: np.ndarray,
        terminal_nodes: np.ndarray,
    ) -> float:
        """
        Return the signed FEM reaction associated with
        a terminal.

        The physical outward current has the opposite sign
        of the reaction under the present weak-form convention.
        """

        reaction = (
            self.calculate_reaction_vector(
                potential
            )
        )

        terminal_indices = np.asarray(
            terminal_nodes,
            dtype=int,
        )

        return float(
            np.sum(
                reaction[
                    terminal_indices
                ]
            )
        )


    # ========================================================
    # COMPLETE SOLUTION
    # ========================================================


    def solve(
        self,
        positive_terminal_nodes: np.ndarray,
        negative_terminal_nodes: np.ndarray,
        voltage_difference: float = 1.0,
    ) -> CurrentDistributionResult:
        """
        Execute the complete stationary electrical FEM solve.

        Terminal currents are obtained from the FEM reaction
        vector associated with the Dirichlet boundary conditions.
        """

        potential = self.solve_potential(
            positive_terminal_nodes,
            negative_terminal_nodes,
            voltage_difference,
        )

        # --------------------------------------------------------
        # Current density
        # --------------------------------------------------------

        element_current_density = (
            self.calculate_element_current_density(
                potential
            )
        )

        element_current_density_magnitude = (
            self.current_density_magnitude(
                element_current_density
            )
        )

        nodal_current_density = (
            self.calculate_nodal_current_density(
                element_current_density
            )
        )

        nodal_current_density_magnitude = (
            self.current_density_magnitude(
                nodal_current_density
            )
        )

        # --------------------------------------------------------
        # Maximum current density
        # --------------------------------------------------------

        maximum_element_index = int(
            np.argmax(
                element_current_density_magnitude
            )
        )

        maximum_current_density = float(
            element_current_density_magnitude[
                maximum_element_index
            ]
        )

        # --------------------------------------------------------
        # FEM reaction currents
        # --------------------------------------------------------

        positive_signed_current = (
            self.calculate_signed_terminal_current(
                potential,
                positive_terminal_nodes,
            )
        )

        negative_signed_current = (
            self.calculate_signed_terminal_current(
                potential,
                negative_terminal_nodes,
            )
        )

        positive_terminal_current = abs(
            positive_signed_current
        )

        negative_terminal_current = abs(
            negative_signed_current
        )

        # --------------------------------------------------------
        # Total transport current
        # --------------------------------------------------------

        total_current = (
            0.5
            * (
                positive_terminal_current
                + negative_terminal_current
            )
        )

        return CurrentDistributionResult(

            potential=potential,

            nodal_current_density=(
                nodal_current_density
            ),

            nodal_current_density_magnitude=(
                nodal_current_density_magnitude
            ),

            element_current_density=(
                element_current_density
            ),

            element_current_density_magnitude=(
                element_current_density_magnitude
            ),

            total_current=total_current,

            positive_terminal_current=(
                positive_terminal_current
            ),

            negative_terminal_current=(
                negative_terminal_current
            ),

            maximum_current_density=(
                maximum_current_density
            ),

            maximum_element_index=(
                maximum_element_index
            ),
        )