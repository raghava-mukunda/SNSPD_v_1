# FILE: src/snspd/fem/solver.py
# PURPOSE:
# Solves sparse FEM linear systems:
#
#     A u = F
#
# Supports:
#
#     - real-valued systems
#     - complex-valued systems
#
# Complex support is required for frequency-domain electromagnetic
# calculations.


from __future__ import annotations


import numpy as np


from scipy.sparse.linalg import (
    spsolve,
)


def solve_linear_system(
    K,
    F,
) -> np.ndarray:
    """
    Solve:

        K u = F

    using sparse direct linear algebra.

    The dtype is preserved, including complex-valued solutions.
    """

    K = K.tocsr()

    F = np.asarray(
        F
    )

    if K.shape[0] != K.shape[1]:

        raise ValueError(
            "FEM system matrix must be square."
        )

    if K.shape[0] != len(F):

        raise ValueError(
            "System matrix and RHS dimensions "
            "do not match."
        )

    solution = spsolve(
        K,
        F,
    )

    if not np.all(
        np.isfinite(
            solution
        )
    ):

        raise RuntimeError(
            "FEM solver produced "
            "non-finite values."
        )

    return solution