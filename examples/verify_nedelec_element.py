# FILE: examples/verify_nedelec_element.py
# PURPOSE:
# Verifies the fundamental degrees of freedom of the
# lowest-order Nedelec first-family triangular element.
#
# For consistently oriented local edges:
#
#     integral_ej Ni · dl = delta_ij
#
# Therefore the edge-circulation matrix must be the
# 3x3 identity matrix up to numerical roundoff.


from __future__ import annotations


import numpy as np


from snspd.fem.electromagnetics.nedelec import (
    NedelecTriangle,
)


def main():

    print(
        "\n"
        "============================================\n"
        "NÉDÉLEC ELEMENT DOF VERIFICATION\n"
        "============================================\n"
    )

    # --------------------------------------------------------
    # Non-degenerate arbitrary triangle.
    #
    # We intentionally do NOT use an equilateral triangle.
    # This checks that the formulation is geometrically
    # invariant rather than accidentally correct for a
    # special geometry.
    # --------------------------------------------------------

    coordinates = np.array(
        [
            [0.10, 0.20],
            [1.30, 0.15],
            [0.35, 1.10],
        ],
        dtype=float,
    )

    element = NedelecTriangle(
        coordinates
    )

    circulation = (
        element.edge_circulation_matrix()
    )

    expected = np.eye(
        3
    )

    error = (
        circulation
        -
        expected
    )

    max_error = np.max(
        np.abs(error)
    )

    print(
        "Triangle coordinates:"
    )

    print(
        coordinates
    )

    print(
        "\nEdge circulation matrix:"
    )

    print(
        circulation
    )

    print(
        "\nExpected:"
    )

    print(
        expected
    )

    print(
        "\nMaximum absolute error:"
    )

    print(
        f"{max_error:.8e}"
    )

    # ========================================================
    # MASS MATRIX
    # ========================================================

    mass = (
        element.mass_matrix()
    )

    print(
        "\nNedelec mass matrix:"
    )

    print(
        mass
    )

    print(
        "\nMass matrix shape:"
    )

    print(
        mass.shape
    )

    if mass.shape != (
        3,
        3,
    ):

        raise RuntimeError(
            "Nedelec mass matrix has "
            "incorrect dimensions."
        )

    # --------------------------------------------------------
    # Hermitian check.
    #
    # For real positive epsilon:
    #
    #     M = M^H
    # --------------------------------------------------------

    hermitian_error = np.max(
        np.abs(
            mass
            -
            mass.conj().T
        )
    )

    print(
        "\nMass matrix Hermitian error:"
    )

    print(
        f"{hermitian_error:.8e}"
    )

    # --------------------------------------------------------
    # Positive definiteness check.
    # --------------------------------------------------------

    eigenvalues = np.linalg.eigvalsh(
        mass
    )

    print(
        "\nMass matrix eigenvalues:"
    )

    print(
        eigenvalues
    )

    if max_error > 1e-12:

        raise RuntimeError(
            "Nedelec DOF verification FAILED."
        )

    if hermitian_error > 1e-12:

        raise RuntimeError(
            "Nedelec mass matrix Hermitian "
            "verification FAILED."
        )

    if np.min(
        eigenvalues
    ) <= 0.0:

        raise RuntimeError(
            "Nedelec mass matrix is not "
            "positive definite."
        )

    print(
        "\n"
        "NÉDÉLEC ELEMENT VERIFICATION : PASS"
    )


if __name__ == "__main__":

    main()