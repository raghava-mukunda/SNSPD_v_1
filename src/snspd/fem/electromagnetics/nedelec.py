# FILE: src/snspd/fem/electromagnetics/nedelec.py
# PURPOSE:
# Implements the lowest-order Nedelec first-family edge element
# on a triangular element.
#
# Mathematical definition:
#
#     N_ij =
#         lambda_i grad(lambda_j)
#         -
#         lambda_j grad(lambda_i)
#
# where lambda_i are barycentric coordinates.
#
# The degrees of freedom are edge circulations:
#
#     DOF_e = integral_e E · dl
#
# with the corresponding orientation.
#
# The local triangle has three edges:
#
#     edge 0 : (0,1)
#     edge 1 : (1,2)
#     edge 2 : (2,0)
#
# This implementation supports:
#
#     - barycentric coordinates
#     - barycentric gradients
#     - Nedelec basis functions
#     - basis curls
#     - curl-curl element matrix
#     - electric-field mass matrix
#
# The electric-field mass matrix is constructed from the
# vector inner product:
#
#     M_ij =
#         integral_K
#         epsilon
#         N_i* · N_j
#         dA
#
# where * denotes complex conjugation.
#
# The implementation therefore supports complex-valued
# frequency-domain material coefficients.


from __future__ import annotations


from dataclasses import dataclass


import numpy as np


@dataclass
class NedelecTriangle:
    """
    Lowest-order Nedelec first-family element
    on a triangular domain.
    """

    coordinates: np.ndarray

    def __post_init__(self):

        self.coordinates = np.asarray(
            self.coordinates,
            dtype=float,
        )

        if self.coordinates.shape != (
            3,
            2,
        ):

            raise ValueError(
                "Triangle coordinates must have "
                "shape (3,2)."
            )

        if self.area <= np.finfo(
            float
        ).eps:

            raise ValueError(
                "Degenerate triangular element."
            )

    # ========================================================
    # SIGNED AREA
    # ========================================================

    @property
    def signed_area(
        self,
    ) -> float:
        """
        Signed triangle area.
        """

        x1, y1 = self.coordinates[0]
        x2, y2 = self.coordinates[1]
        x3, y3 = self.coordinates[2]

        return 0.5 * (
            x1 * (y2 - y3)
            + x2 * (y3 - y1)
            + x3 * (y1 - y2)
        )

    # ========================================================
    # POSITIVE AREA
    # ========================================================

    @property
    def area(
        self,
    ) -> float:
        """
        Physical triangle area.
        """

        return abs(
            self.signed_area
        )

    # ========================================================
    # BARYCENTRIC GRADIENTS
    # ========================================================

    @property
    def barycentric_gradients(
        self,
    ) -> np.ndarray:
        """
        Return:

            grad(lambda_1)
            grad(lambda_2)
            grad(lambda_3)

        as a (3,2) array.

        Each barycentric-coordinate gradient is constant
        throughout a linear triangular element.
        """

        x1, y1 = self.coordinates[0]
        x2, y2 = self.coordinates[1]
        x3, y3 = self.coordinates[2]

        twice_signed_area = (
            2.0 * self.signed_area
        )

        if abs(
            twice_signed_area
        ) <= np.finfo(float).eps:

            raise ValueError(
                "Degenerate triangular element."
            )

        b = np.array(
            [
                y2 - y3,
                y3 - y1,
                y1 - y2,
            ],
            dtype=float,
        )

        c = np.array(
            [
                x3 - x2,
                x1 - x3,
                x2 - x1,
            ],
            dtype=float,
        )

        return (
            np.column_stack(
                (
                    b,
                    c,
                )
            )
            / twice_signed_area
        )

    # ========================================================
    # BARYCENTRIC COORDINATES
    # ========================================================

    def barycentric_coordinates(
        self,
        point,
    ) -> np.ndarray:
        """
        Evaluate:

            lambda_1,
            lambda_2,
            lambda_3

        at a physical point.
        """

        x = float(
            point[0]
        )

        y = float(
            point[1]
        )

        x1, y1 = self.coordinates[0]
        x2, y2 = self.coordinates[1]
        x3, y3 = self.coordinates[2]

        denominator = (
            (y2 - y3)
            * (x1 - x3)
            +
            (x3 - x2)
            * (y1 - y3)
        )

        if abs(
            denominator
        ) <= np.finfo(float).eps:

            raise ValueError(
                "Degenerate triangular element."
            )

        lambda_1 = (
            (
                (y2 - y3)
                * (x - x3)
                +
                (x3 - x2)
                * (y - y3)
            )
            / denominator
        )

        lambda_2 = (
            (
                (y3 - y1)
                * (x - x3)
                +
                (x1 - x3)
                * (y - y3)
            )
            / denominator
        )

        lambda_3 = (
            1.0
            - lambda_1
            - lambda_2
        )

        return np.array(
            [
                lambda_1,
                lambda_2,
                lambda_3,
            ],
            dtype=float,
        )

    # ========================================================
    # NÉDÉLEC BASIS FUNCTION
    # ========================================================

    def basis_function(
        self,
        edge_index: int,
        point,
    ) -> np.ndarray:
        """
        Evaluate one local Nedelec basis function.

        Local edge ordering:

            edge 0 : vertex 0 -> vertex 1
            edge 1 : vertex 1 -> vertex 2
            edge 2 : vertex 2 -> vertex 0

        Mathematical definition:

            N_ij =
                lambda_i grad(lambda_j)
                -
                lambda_j grad(lambda_i)
        """

        edge_vertices = (
            (0, 1),
            (1, 2),
            (2, 0),
        )

        if edge_index not in (
            0,
            1,
            2,
        ):

            raise ValueError(
                "Nedelec triangle edge index "
                "must be 0, 1, or 2."
            )

        i, j = edge_vertices[
            edge_index
        ]

        lambdas = (
            self.barycentric_coordinates(
                point
            )
        )

        gradients = (
            self.barycentric_gradients
        )

        return (
            lambdas[i]
            * gradients[j]
            -
            lambdas[j]
            * gradients[i]
        )

    # ========================================================
    # ALL BASIS FUNCTIONS
    # ========================================================

    def basis_functions(
        self,
        point,
    ) -> np.ndarray:
        """
        Evaluate all three local Nedelec basis functions.

        Returns
        -------
        ndarray
            Shape (3,2).

        Row i contains:

            N_i = [N_ix, N_iy]
        """

        return np.asarray(
            [
                self.basis_function(
                    i,
                    point,
                )
                for i in range(3)
            ],
            dtype=float,
        )

    # ========================================================
    # BASIS CURL
    # ========================================================

    def basis_curl(
        self,
        edge_index: int,
    ) -> float:
        """
        Calculate the scalar 2D curl of a Nedelec basis:

            curl(N)
              =
              dN_y/dx
              -
              dN_x/dy

        For the lowest-order Nedelec triangle this is
        constant throughout the element.
        """

        edge_vertices = (
            (0, 1),
            (1, 2),
            (2, 0),
        )

        if edge_index not in (
            0,
            1,
            2,
        ):

            raise ValueError(
                "Nedelec triangle edge index "
                "must be 0, 1, or 2."
            )

        i, j = edge_vertices[
            edge_index
        ]

        gradients = (
            self.barycentric_gradients
        )

        grad_i = gradients[i]

        grad_j = gradients[j]

        return float(
            2.0
            * (
                grad_i[0]
                * grad_j[1]
                -
                grad_i[1]
                * grad_j[0]
            )
        )

    # ========================================================
    # ALL BASIS CURLS
    # ========================================================

    def basis_curls(
        self,
    ) -> np.ndarray:
        """
        Return all three constant basis curls.
        """

        return np.asarray(
            [
                self.basis_curl(i)
                for i in range(3)
            ],
            dtype=float,
        )

    # ========================================================
    # CURL-CURL MATRIX
    # ========================================================

    def curl_curl_matrix(
        self,
        inverse_mu=1.0,
    ) -> np.ndarray:
        """
        Calculate the local curl-curl matrix:

            K_ij =
                integral_K
                mu^{-1}
                curl(N_i)^*
                curl(N_j)
                dA

        For first-order Nedelec elements, the curls are
        constant, so this integral is exact analytically.

        For scalar real/complex mu:

            K_ij =
                mu^{-1}
                A
                curl(N_i)
                curl(N_j)
        """

        curls = (
            self.basis_curls()
        )

        # For scalar material coefficients the curl basis
        # functions are real-valued. Material dispersion or
        # loss may enter through complex inverse_mu.
        return (
            inverse_mu
            * self.area
            * np.outer(
                curls,
                curls,
            )
        )

    # ========================================================
    # ELECTRIC-FIELD MASS MATRIX
    # ========================================================

    def mass_matrix(
        self,
        coefficient=1.0,
    ) -> np.ndarray:
        """
        Calculate the local electric-field mass matrix:

            M_ij =
                integral_K
                coefficient
                N_i^* · N_j
                dA

        For a constant coefficient and lowest-order Nedelec
        basis functions:

            N_i(x,y)

        are linear functions, therefore:

            N_i^* · N_j

        is quadratic.

        The three-point symmetric triangular quadrature rule
        used here is exact for polynomials of total degree <= 2.

        The resulting matrix is 3x3.
        """

        coefficient = np.asarray(
            coefficient
        )

        coefficient_is_complex = (
            np.iscomplexobj(
                coefficient
            )
        )

        dtype = (
            complex
            if coefficient_is_complex
            else float
        )

        matrix = np.zeros(
            (
                3,
                3,
            ),
            dtype=dtype,
        )

        # ----------------------------------------------------
        # Three-point symmetric triangle quadrature.
        #
        # Each point has barycentric coordinates:
        #
        #     (1/6, 1/6, 2/3)
        #     (1/6, 2/3, 1/6)
        #     (2/3, 1/6, 1/6)
        #
        # Each physical weight is:
        #
        #     A/3
        #
        # ----------------------------------------------------

        quadrature_points = np.array(
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
            ],
            dtype=float,
        )

        physical_weight = (
            self.area / 3.0
        )

        # ====================================================
        # QUADRATURE
        # ====================================================

        for barycentric in (
            quadrature_points
        ):

            point = (
                barycentric
                @ self.coordinates
            )

            # Shape:
            #
            #     (3,2)
            #
            # Row i:
            #
            #     [N_ix, N_iy]
            #
            basis = (
                self.basis_functions(
                    point
                )
            )

            if basis.shape != (
                3,
                2,
            ):

                raise RuntimeError(
                    "Internal Nedelec basis "
                    "shape error."
                )

            # ------------------------------------------------
            # Vector-valued Gram matrix:
            #
            #     G_ij = N_i^* · N_j
            #
            # basis.conj() has shape (3,2)
            # basis.T     has shape (2,3)
            #
            # therefore:
            #
            #     (3,2) @ (2,3)
            #
            # ->  (3,3)
            #
            # This is the mathematically correct vector
            # inner product.
            # ------------------------------------------------

            gram_matrix = (
                basis.conj()
                @ basis.T
            )

            matrix += (
                coefficient
                * physical_weight
                * gram_matrix
            )

        return matrix

    # ========================================================
    # EDGE CIRCULATION VERIFICATION
    # ========================================================

    def edge_circulation_matrix(
        self,
    ) -> np.ndarray:
        """
        Numerically verify the fundamental Nedelec DOF property:

            integral_edge_j N_i · dl

        should equal:

            1, if i == j
            0, otherwise

        for consistently oriented local edges.

        This routine uses exact line integration for the
        lowest-order basis because the basis is linear.
        """

        vertices = (
            self.coordinates
        )

        edges = (
            (0, 1),
            (1, 2),
            (2, 0),
        )

        result = np.zeros(
            (
                3,
                3,
            ),
            dtype=float,
        )

        for edge_j, (
            start,
            end,
        ) in enumerate(edges):

            r0 = vertices[
                start
            ]

            r1 = vertices[
                end
            ]

            tangent = (
                r1 - r0
            )

            # Parameterization:
            #
            #     r(t)
            #       =
            #       r0 + t(r1-r0)
            #
            #     0 <= t <= 1
            #
            #     dr/dt = tangent
            #
            # Therefore:
            #
            #     integral N·dr
            #
            #     =
            #
            #     integral_0^1
            #     N(r(t))·tangent dt
            #
            # N is linear, so midpoint quadrature is exact.

            midpoint = (
                0.5
                * (
                    r0 + r1
                )
            )

            for edge_i in range(3):

                basis = (
                    self.basis_function(
                        edge_i,
                        midpoint,
                    )
                )

                result[
                    edge_i,
                    edge_j,
                ] = np.dot(
                    basis,
                    tangent,
                )

        return result