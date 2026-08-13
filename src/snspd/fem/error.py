# FILE: src/snspd/fem/error.py
# PURPOSE:
# Implements FEM error norms for real and complex-valued fields.
#
# Supported norms:
#
#     L2:
#
#         ||e||_L2
#         =
#         sqrt(
#             ∫ |e|² dΩ
#         )
#
#     H1 seminorm:
#
#         |e|_H1
#         =
#         sqrt(
#             ∫ |∇e|² dΩ
#         )
#
# Complex fields use the Hermitian magnitude:
#
#     |z|² = z* z


from __future__ import annotations


import numpy as np


from snspd.fem.elements import (
    TriangleElement,
)


from snspd.mesh.mesh import Mesh


def _triangle_quadrature_points(
    coordinates: np.ndarray,
):
    """
    Three-point Gaussian quadrature rule for a triangle.
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
        @ coordinates
    )

    x1, y1 = coordinates[0]
    x2, y2 = coordinates[1]
    x3, y3 = coordinates[2]

    area = abs(
        0.5
        * (
            (x2 - x1)
            * (y3 - y1)
            -
            (x3 - x1)
            * (y2 - y1)
        )
    )

    weights = np.full(
        3,
        area / 3.0,
    )

    return (
        barycentric,
        points,
        weights,
    )


def calculate_l2_error(
    mesh: Mesh,
    numerical_solution: np.ndarray,
    exact_solution,
) -> float:
    """
    Calculate:

        ||u - uh||_L2

        =
        sqrt(
            ∫Ω |u - uh|² dΩ
        )

    Supports real and complex fields.
    """

    numerical_solution = np.asarray(
        numerical_solution
    )

    if len(numerical_solution) != (
        mesh.node_count
    ):

        raise ValueError(
            "Numerical solution length does "
            "not match mesh node count."
        )

    integral = 0.0

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

        element_solution = (
            numerical_solution[
                node_indices
            ]
        )

        (
            barycentric,
            points,
            weights,
        ) = _triangle_quadrature_points(
            coordinates
        )

        for bary, point, weight in zip(
            barycentric,
            points,
            weights,
        ):

            uh = np.dot(
                bary,
                element_solution,
            )

            exact = exact_solution(
                point[0],
                point[1],
            )

            difference = (
                exact - uh
            )

            integral += (
                np.real(
                    np.conjugate(
                        difference
                    )
                    * difference
                )
                * weight
            )

    return float(
        np.sqrt(
            max(
                integral,
                0.0,
            )
        )
    )


def calculate_h1_seminorm_error(
    mesh: Mesh,
    numerical_solution: np.ndarray,
    exact_gradient,
) -> float:
    """
    Calculate:

        |u - uh|_H1

        =
        sqrt(
            ∫Ω |∇u - ∇uh|² dΩ
        )

    Supports real and complex fields.
    """

    numerical_solution = np.asarray(
        numerical_solution
    )

    integral = 0.0

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

        element_solution = (
            numerical_solution[
                node_indices
            ]
        )

        element = TriangleElement(
            coordinates
        )

        gradients = (
            element.shape_function_gradients
        )

        # grad(uh) =
        #
        #     Σ ui grad(Ni)
        #
        numerical_gradient = (
            element_solution
            @ gradients
        )

        (
            _,
            points,
            weights,
        ) = _triangle_quadrature_points(
            coordinates
        )

        for point, weight in zip(
            points,
            weights,
        ):

            exact_gradient_value = np.asarray(
                exact_gradient(
                    point[0],
                    point[1],
                )
            )

            difference = (
                exact_gradient_value
                - numerical_gradient
            )

            integral += (
                np.real(
                    np.vdot(
                        difference,
                        difference,
                    )
                )
                * weight
            )

    return float(
        np.sqrt(
            max(
                integral,
                0.0,
            )
        )
    )