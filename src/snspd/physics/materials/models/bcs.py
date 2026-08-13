# FILE: src/snspd/physics/materials/models/bcs.py
# PURPOSE:
# Generic BCS superconducting relations.
#
# This module contains mathematical models only.
# It contains NO material-specific parameters.
#
# Material-specific files such as nbn.py, wsi.py and mosi.py
# will call these functions when the assumptions of the model
# are appropriate.
#
# Implemented:
#
#   Δ(T) / Δ(0)
#   Δ(T)
#   superfluid-density temperature dependence
#   normalized London penetration depth
#
# The weak-coupling BCS gap ratio is NOT hard-coded into a
# material model. It is supplied explicitly by the caller.


from __future__ import annotations


from math import (
    sqrt,
    tanh,
)


def validate_reduced_temperature(
    reduced_temperature: float,
) -> float:
    """
    Validate and return reduced temperature:

        t = T / Tc

    Valid superconducting range:

        0 <= t < 1
    """

    t = float(reduced_temperature)

    if t < 0.0:

        raise ValueError(
            "Reduced temperature cannot be negative."
        )

    if t >= 1.0:

        raise ValueError(
            "Reduced temperature must satisfy "
            "0 <= T/Tc < 1."
        )

    return t


def normalized_gap(
    reduced_temperature: float,
) -> float:
    """
    Weak-coupling BCS approximation:

        Δ(T)
        ──────
        Δ(0)

        =
        tanh[
            1.74 sqrt(Tc/T - 1)
        ]

    expressed using:

        t = T/Tc

    therefore:

        Δ(T)/Δ(0)
        =
        tanh[
            1.74 sqrt(1/t - 1)
        ]

    At T = 0 the limiting value is exactly 1.
    """

    t = validate_reduced_temperature(
        reduced_temperature
    )

    if t == 0.0:

        return 1.0

    return tanh(
        1.74
        * sqrt(
            1.0 / t - 1.0
        )
    )


def superconducting_gap(
    delta_zero: float,
    reduced_temperature: float,
) -> float:
    """
    Calculate:

        Δ(T) = Δ(0) f(T/Tc)

    Parameters
    ----------
    delta_zero:
        Experimentally supplied or otherwise traceable
        zero-temperature gap [J].

    reduced_temperature:
        T/Tc.
    """

    if delta_zero <= 0.0:

        raise ValueError(
            "delta_zero must be positive."
        )

    return (
        delta_zero
        * normalized_gap(
            reduced_temperature
        )
    )


def normalized_superfluid_density(
    reduced_temperature: float,
) -> float:
    """
    Two-fluid/BCS-compatible phenomenological approximation:

        n_s(T) / n_s(0)
        =
        1 - (T/Tc)^4

    This function is explicitly identified as an approximation
    and must NOT be confused with the full microscopic
    BCS/Mattis-Bardeen superfluid-density calculation.

    It is provided as a model component so that material models
    can explicitly choose whether this approximation is valid.

    Valid range:

        0 <= T/Tc < 1
    """

    t = validate_reduced_temperature(
        reduced_temperature
    )

    return 1.0 - t**4


def normalized_penetration_depth(
    reduced_temperature: float,
) -> float:
    """
    Penetration-depth relation corresponding to the selected
    normalized superfluid-density model:

        λ(T) / λ(0)
        =
        1 / sqrt[n_s(T)/n_s(0)]
    """

    ns = normalized_superfluid_density(
        reduced_temperature
    )

    if ns <= 0.0:

        raise RuntimeError(
            "Superfluid density became non-positive."
        )

    return 1.0 / sqrt(ns)