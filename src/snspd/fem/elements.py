# FILE: src/snspd/fem/elements.py
# PURPOSE:
# Implements element-level mathematical operators for linear
# three-node triangular P1 finite elements.
#
# Supported operators:
#
#     - element area
#     - shape-function gradients
#     - stiffness matrix
#     - consistent mass matrix
#
# These operators are generic numerical FEM mathematics and are
# independent of SNSPD-specific physics.


from __future__ import annotations


from dataclasses import dataclass


import numpy as np


@dataclass
class TriangleElement:
    """
    Linear three-node triangular P1 finite element.

    Coordinates are stored in SI units [m], although the generic
    FEM verification problems may use dimensionless coordinates.
    """

    coordinates: np.ndarray

    def __post_init__(self):

        self.coordinates = np.asarray(
            self.coordinates,
            dtype=float,
        )

        if self.coordinates.shape != (3, 2):

            raise ValueError(
                "Triangle coordinates must have "
                "shape (3, 2)."
            )

    # =========================================================
    # AREA
    # =========================================================

    @property
    def signed_area(self) -> float:
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

    @property
    def area(self) -> float:
        """
        Positive physical triangle area.
        """

        return abs(
            self.signed_area
        )

    # =========================================================
    # SHAPE-FUNCTION GRADIENTS
    # =========================================================

    @property
    def shape_function_gradients(
        self,
    ) -> np.ndarray:
        """
        Return:

            [dNi/dx, dNi/dy]

        for i = 1,2,3.

        Returns
        -------
        ndarray
            Shape (3,2).
        """

        x1, y1 = self.coordinates[0]
        x2, y2 = self.coordinates[1]
        x3, y3 = self.coordinates[2]

        twice_area = (
            2.0 * self.signed_area
        )

        if abs(twice_area) <= np.finfo(
            float
        ).eps:

            raise ValueError(
                "Degenerate triangular element "
                "has zero area."
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

        return np.column_stack(
            (
                b,
                c,
            )
        ) / twice_area

    # =========================================================
    # STIFFNESS MATRIX
    # =========================================================

    def stiffness_matrix(
        self,
        coefficient=1.0,
    ) -> np.ndarray:
        """
        Calculate:

            Ke_ij =
                ∫Ωe k ∇Ni · ∇Nj dΩ

        For a linear triangle:

            Ke = k A BᵀB
        """

        gradients = (
            self.shape_function_gradients
        )

        return (
            coefficient
            * self.area
            * (
                gradients
                @ gradients.T
            )
        )

    # =========================================================
    # MASS MATRIX
    # =========================================================

    def mass_matrix(
        self,
        coefficient=1.0,
    ) -> np.ndarray:
        """
        Calculate the consistent P1 triangular mass matrix:

            Me_ij =
                ∫Ωe k Ni Nj dΩ

        For a linear triangle:

            M =
                k A/12
                [[2,1,1],
                 [1,2,1],
                 [1,1,2]]
        """

        return (
            coefficient
            * self.area
            / 12.0
            * np.array(
                [
                    [2.0, 1.0, 1.0],
                    [1.0, 2.0, 1.0],
                    [1.0, 1.0, 2.0],
                ],
                dtype=float,
            )
        )

    # =========================================================
    # LOAD VECTOR
    # =========================================================

    def load_vector(
        self,
        source_function,
    ) -> np.ndarray:
        """
        Calculate the element load vector:

            Fe_i =
                ∫Ωe Ni f(x,y) dΩ

        using three-point triangular quadrature.

        The returned dtype is automatically selected so complex
        source functions are preserved.
        """

        barycentric = np.array(
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
            ]
        )

        points = (
            barycentric
            @ self.coordinates
        )

        weights = np.full(
            3,
            self.area / 3.0,
        )

        # Evaluate source once to determine whether the system
        # is real or complex.
        test_value = source_function(
            points[0, 0],
            points[0, 1],
        )

        dtype = (
            complex
            if np.iscomplexobj(test_value)
            else float
        )

        result = np.zeros(
            3,
            dtype=dtype,
        )

        for point, weight, bary in zip(
            points,
            weights,
            barycentric,
        ):

            source = source_function(
                point[0],
                point[1],
            )

            result += (
                weight
                * source
                * bary
            )

        return result