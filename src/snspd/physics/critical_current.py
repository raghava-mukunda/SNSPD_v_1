"""
FILE: src/snspd/physics/critical_current.py

PURPOSE
-------
Critical-current model for the SNSPD digital twin.

This is Phase A + Phase B of the critical-current pipeline:

    FEM current density J(x,y)
        |
        +--> transport current
        |
        +--> current-crowding factor C_J
        |
        +--> straight-wire critical current
        |
        +--> geometry-limited critical current

PHYSICAL MODEL
--------------
For a uniform straight nanowire,

    I_c,straight = J_c * w * t

where

    J_c : material critical current density [A/m^2]
    w   : nanowire width [m]
    t   : superconducting film thickness [m]

For a geometry whose FEM current-density field is linear in
transport current,

    J(x,y; I) = (I / I_FEM) * J_FEM(x,y)

The device reaches the local critical-current condition when

    max |J(x,y; I_c)| = J_c

and therefore

    I_c,geometry
        = I_FEM * J_c / J_max,FEM

Equivalently, defining

    J_transport = I_FEM / (w*t)

    C_J = J_max,FEM / J_transport

gives

    I_c,geometry
        = I_c,straight / C_J

This module intentionally does NOT yet model:
    - temperature dependence of J_c
    - magnetic-field dependence
    - vortex entry
    - Ginzburg-Landau order parameter
    - kinetic inductance dynamics
    - electrothermal switching

Those are later physics layers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NanowireParameters:
    """
    Geometric parameters of the superconducting nanowire.

    Parameters
    ----------
    width_m:
        Nanowire width [m].
    thickness_m:
        Superconducting film thickness [m].
    """

    width_m: float
    thickness_m: float

    def __post_init__(self) -> None:
        if self.width_m <= 0.0:
            raise ValueError("Nanowire width must be positive.")

        if self.thickness_m <= 0.0:
            raise ValueError(
                "Nanowire thickness must be positive."
            )

    @property
    def cross_sectional_area_m2(self) -> float:
        """
        Superconducting cross-sectional area [m^2].
        """
        return self.width_m * self.thickness_m


@dataclass(frozen=True)
class CriticalCurrentParameters:
    """
    Material critical-current parameter set.

    Phase A uses a prescribed critical current density J_c.

    J_c should come from the selected material/device characterization
    rather than being silently assumed by the simulator.
    """

    critical_current_density_A_m2: float

    material: str = "unspecified"

    temperature_K: float | None = None

    def __post_init__(self) -> None:
        if self.critical_current_density_A_m2 <= 0.0:
            raise ValueError(
                "Critical current density J_c must be positive."
            )

        if (
            self.temperature_K is not None
            and self.temperature_K <= 0.0
        ):
            raise ValueError(
                "Temperature must be positive when supplied."
            )


@dataclass(frozen=True)
class CriticalCurrentResult:
    """
    Critical-current calculation results.
    """

    material: str

    temperature_K: float | None

    critical_current_density_A_m2: float

    wire_width_m: float

    film_thickness_m: float

    cross_sectional_area_m2: float

    straight_wire_critical_current_A: float

    fem_transport_current_A: float

    fem_max_current_density_A_m2: float

    transport_current_density_A_m2: float

    current_crowding_factor: float

    geometry_limited_critical_current_A: float

    reduction_fraction: float

    reduction_percent: float

    @property
    def geometry_to_straight_ratio(self) -> float:
        """
        Ratio Ic,geometry / Ic,straight.
        """
        return (
            self.geometry_limited_critical_current_A
            / self.straight_wire_critical_current_A
        )


def calculate_straight_wire_critical_current(
    wire: NanowireParameters,
    material: CriticalCurrentParameters,
) -> float:
    """
    Calculate the straight-wire critical current.

        I_c = J_c * w * t

    Returns
    -------
    float
        Critical current [A].
    """

    return (
        material.critical_current_density_A_m2
        * wire.cross_sectional_area_m2
    )


def calculate_transport_current_density(
    transport_current_A: float,
    wire: NanowireParameters,
) -> float:
    """
    Calculate nominal transport current density.

        J_transport = I / (w*t)

    Returns
    -------
    float
        Transport current density [A/m^2].
    """

    if transport_current_A <= 0.0:
        raise ValueError(
            "Transport current must be positive."
        )

    return (
        transport_current_A
        / wire.cross_sectional_area_m2
    )


def calculate_current_crowding_factor(
    fem_max_current_density_A_m2: float,
    transport_current_A: float,
    wire: NanowireParameters,
) -> float:
    """
    Calculate

        C_J = J_max / J_transport

    Returns
    -------
    float
        Dimensionless current-crowding factor.
    """

    if fem_max_current_density_A_m2 <= 0.0:
        raise ValueError(
            "FEM maximum current density must be positive."
        )

    transport_J = calculate_transport_current_density(
        transport_current_A,
        wire,
    )

    return (
        fem_max_current_density_A_m2
        / transport_J
    )


def calculate_geometry_limited_critical_current(
    fem_transport_current_A: float,
    fem_max_current_density_A_m2: float,
    material: CriticalCurrentParameters,
) -> float:
    """
    Calculate the geometry-limited critical current directly
    from the FEM current-density solution.

        I_c,geometry
            = I_FEM * J_c / J_max,FEM

    Returns
    -------
    float
        Geometry-limited critical current [A].
    """

    if fem_transport_current_A <= 0.0:
        raise ValueError(
            "FEM transport current must be positive."
        )

    if fem_max_current_density_A_m2 <= 0.0:
        raise ValueError(
            "FEM maximum current density must be positive."
        )

    return (
        fem_transport_current_A
        * material.critical_current_density_A_m2
        / fem_max_current_density_A_m2
    )


def analyze_critical_current(
    wire: NanowireParameters,
    material: CriticalCurrentParameters,
    fem_transport_current_A: float,
    fem_max_current_density_A_m2: float,
) -> CriticalCurrentResult:
    """
    Perform Phase A + Phase B critical-current analysis.

    The calculation is performed in two independent ways:

    1. Straight-wire:
           Ic = Jc*w*t

    2. Geometry-limited:
           Ic = I_FEM*Jc/Jmax

    The geometry-limited result should also satisfy

           Ic,geometry = Ic,straight / C_J

    for a consistent FEM normalization.
    """

    straight_ic = calculate_straight_wire_critical_current(
        wire,
        material,
    )

    transport_J = calculate_transport_current_density(
        fem_transport_current_A,
        wire,
    )

    crowding_factor = calculate_current_crowding_factor(
        fem_max_current_density_A_m2,
        fem_transport_current_A,
        wire,
    )

    geometry_ic = calculate_geometry_limited_critical_current(
        fem_transport_current_A,
        fem_max_current_density_A_m2,
        material,
    )

    # Geometry-limited critical current should be smaller than or
    # equal to the straight-wire value for C_J >= 1.
    reduction_fraction = (
        1.0
        - geometry_ic / straight_ic
    )

    reduction_percent = (
        100.0 * reduction_fraction
    )

    return CriticalCurrentResult(
        material=material.material,
        temperature_K=material.temperature_K,
        critical_current_density_A_m2=(
            material.critical_current_density_A_m2
        ),
        wire_width_m=wire.width_m,
        film_thickness_m=wire.thickness_m,
        cross_sectional_area_m2=(
            wire.cross_sectional_area_m2
        ),
        straight_wire_critical_current_A=straight_ic,
        fem_transport_current_A=fem_transport_current_A,
        fem_max_current_density_A_m2=(
            fem_max_current_density_A_m2
        ),
        transport_current_density_A_m2=transport_J,
        current_crowding_factor=crowding_factor,
        geometry_limited_critical_current_A=geometry_ic,
        reduction_fraction=reduction_fraction,
        reduction_percent=reduction_percent,
    )


def calculate_local_critical_current_map(
    fem_element_current_density_A_m2: np.ndarray,
    fem_transport_current_A: float,
    critical_current_density_A_m2: float,
) -> np.ndarray:
    """
    Calculate the local transport current at which each FEM
    element would individually reach J_c.

        I_c,local(e)
            = I_FEM * J_c / |J_FEM(e)|

    Elements with zero current density receive +infinity.

    Parameters
    ----------
    fem_element_current_density_A_m2:
        Array of element current-density magnitudes [A/m^2].

    fem_transport_current_A:
        Transport current represented by the FEM solution [A].

    critical_current_density_A_m2:
        Material critical current density [A/m^2].

    Returns
    -------
    ndarray
        Local critical-current estimate for each element [A].
    """

    J = np.asarray(
        fem_element_current_density_A_m2,
        dtype=float,
    )

    if fem_transport_current_A <= 0.0:
        raise ValueError(
            "FEM transport current must be positive."
        )

    if critical_current_density_A_m2 <= 0.0:
        raise ValueError(
            "Critical current density must be positive."
        )

    local_ic = np.full_like(
        J,
        np.inf,
        dtype=float,
    )

    mask = (
        np.isfinite(J)
        & (J > 0.0)
    )

    local_ic[mask] = (
        fem_transport_current_A
        * critical_current_density_A_m2
        / J[mask]
    )

    return local_ic


def format_critical_current_result(
    result: CriticalCurrentResult,
) -> str:
    """
    Generate a human-readable critical-current report.
    """

    temperature = (
        f"{result.temperature_K:.6f} K"
        if result.temperature_K is not None
        else "not specified"
    )

    return (
        "\n"
        "SNSPD CRITICAL CURRENT ANALYSIS\n"
        "===============================\n"
        f"Material                     : "
        f"{result.material}\n"
        f"Temperature                  : "
        f"{temperature}\n"
        f"Wire width                   : "
        f"{result.wire_width_m * 1e9:.6f} nm\n"
        f"Film thickness               : "
        f"{result.film_thickness_m * 1e9:.6f} nm\n"
        f"Cross-sectional area         : "
        f"{result.cross_sectional_area_m2:.6e} m²\n"
        "\n"
        f"Critical current density Jc  : "
        f"{result.critical_current_density_A_m2:.6e} A/m²\n"
        "\n"
        "STRAIGHT WIRE\n"
        "-------------\n"
        f"Ic,straight                  : "
        f"{result.straight_wire_critical_current_A:.6e} A\n"
        "\n"
        "FEM CURRENT DISTRIBUTION\n"
        "------------------------\n"
        f"FEM transport current        : "
        f"{result.fem_transport_current_A:.6e} A\n"
        f"FEM maximum |J|              : "
        f"{result.fem_max_current_density_A_m2:.6e} A/m²\n"
        f"Transport current density    : "
        f"{result.transport_current_density_A_m2:.6e} A/m²\n"
        f"Current crowding factor C_J  : "
        f"{result.current_crowding_factor:.6f}\n"
        "\n"
        "GEOMETRY-LIMITED CRITICAL CURRENT\n"
        "---------------------------------\n"
        f"Ic,geometry                  : "
        f"{result.geometry_limited_critical_current_A:.6e} A\n"
        f"Ic,geometry / Ic,straight    : "
        f"{result.geometry_to_straight_ratio:.6f}\n"
        f"Critical-current reduction   : "
        f"{result.reduction_percent:.6f} %\n"
    )


def validate_critical_current_consistency(
    result: CriticalCurrentResult,
    relative_tolerance: float = 1.0e-10,
) -> None:
    """
    Verify the identity

        Ic,geometry = Ic,straight / C_J

    for the Phase A/B model.
    """

    expected = (
        result.straight_wire_critical_current_A
        / result.current_crowding_factor
    )

    actual = (
        result.geometry_limited_critical_current_A
    )

    relative_error = abs(
        actual - expected
    ) / max(
        abs(expected),
        1.0e-30,
    )

    if relative_error > relative_tolerance:
        raise RuntimeError(
            "Critical-current consistency verification FAILED. "
            f"Relative error = {relative_error:.6e}"
        )
