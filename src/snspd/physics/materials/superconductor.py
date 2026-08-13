"""
Superconducting constitutive physics for SNSPD materials.

Scope
-----
This module implements temperature-dependent superconducting quantities
required by the SNSPD device model.

The model is intentionally parameter-driven:
    - No NbN parameters are hard-coded.
    - All dimensional material parameters must be supplied explicitly.
    - Derived quantities are calculated from the supplied parameters.

Primary model:
    Dirty-limit superconducting film.

Implemented quantities:
    Δ(T)                  superconducting energy gap
    R_square              normal-state sheet resistance
    Lk_square(0)          zero-temperature kinetic inductance / square
    Lk_square(T)          temperature-dependent kinetic inductance / square
    Lk                    device kinetic inductance
    j_dep(T)              dirty-limit depairing current density
    I_dep(T)              depairing current

References
----------
Kupriyanov & Lukichev:
    Dirty-limit depairing current.

Frasca et al.:
    Determining the depairing current in superconducting nanowire
    single-photon detectors.

Sidorova et al.:
    Timing Jitter and Electron-Phonon Interaction in SNSPDs.

You:
    Superconducting Nanowire Single-Photon Detectors for Quantum Information.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import (
    exp,
    pi,
    sqrt,
    tanh,
)

# ---------------------------------------------------------------------------
# Fundamental constants
# ---------------------------------------------------------------------------

K_B = 1.380649e-23          # J/K
H_BAR = 1.054571817e-34     # J s
E_CHARGE = 1.602176634e-19  # C
MU_0 = 4.0e-7 * pi


# ---------------------------------------------------------------------------
# Material parameter container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SuperconductorParameters:
    """
    Experimental/material parameters for a superconducting SNSPD film.

    All dimensional parameters are SI.

    Parameters
    ----------
    Tc:
        Superconducting transition temperature [K].

    delta_0:
        Zero-temperature superconducting energy gap [J].

    rho_n:
        Normal-state resistivity [Ω m].

    diffusion_coefficient:
        Electronic diffusion coefficient D [m²/s].

    thickness:
        Film thickness d [m].

    gap_ratio:
        Δ(0)/(k_B Tc).

        This MUST be explicitly supplied.

        For weak-coupling BCS:
            gap_ratio ≈ 1.764

        For an experimentally characterized NbN film, use the
        experimentally measured value instead.

    Notes
    -----
    No material-specific defaults are provided intentionally.
    """

    Tc: float
    delta_0: float
    rho_n: float
    diffusion_coefficient: float
    thickness: float
    gap_ratio: float

    def __post_init__(self) -> None:
        values = {
            "Tc": self.Tc,
            "delta_0": self.delta_0,
            "rho_n": self.rho_n,
            "diffusion_coefficient": self.diffusion_coefficient,
            "thickness": self.thickness,
            "gap_ratio": self.gap_ratio,
        }

        for name, value in values.items():
            if not isinstance(value, (int, float)):
                raise TypeError(
                    f"{name} must be a real scalar, got {type(value)}"
                )

            if not float(value) > 0.0:
                raise ValueError(
                    f"{name} must be strictly positive, got {value}"
                )

        expected_delta = self.gap_ratio * K_B * self.Tc

        # Do not silently accept an inconsistent material definition.
        relative_error = abs(self.delta_0 - expected_delta) / expected_delta

        if relative_error > 1e-8:
            raise ValueError(
                "Inconsistent superconducting parameters: "
                "delta_0 must satisfy "
                "delta_0 = gap_ratio * k_B * Tc. "
                f"Expected {expected_delta:.12e} J, "
                f"received {self.delta_0:.12e} J."
            )

    @property
    def sheet_resistance(self) -> float:
        """
        Normal-state sheet resistance:

            R□ = ρ_n / d
        """
        return self.rho_n / self.thickness

    @property
    def delta_0_ev(self) -> float:
        """Zero-temperature gap in electron-volts."""
        return self.delta_0 / E_CHARGE


# ---------------------------------------------------------------------------
# BCS temperature dependence
# ---------------------------------------------------------------------------

def normalized_gap_bcs(reduced_temperature: float) -> float:
    """
    Weak-coupling BCS approximation for Δ(T) / Δ(0).

    Valid for:
        0 <= T/Tc < 1

    Formula:

        Δ(T)/Δ(0)
            ≈ tanh[
                1.74 * sqrt(Tc/T - 1)
            ]

    The approximation is not used as a material-specific NbN assumption.
    It describes the temperature dependence once the experimentally supplied
    Δ(0) is known.
    """

    t = float(reduced_temperature)

    if t < 0.0:
        raise ValueError("Reduced temperature cannot be negative.")

    if t == 0.0:
        return 1.0

    if t >= 1.0:
        return 0.0

    return tanh(1.74 * sqrt(1.0 / t - 1.0))


def superconducting_gap(
    params: SuperconductorParameters,
    temperature: float,
) -> float:
    """
    Temperature-dependent superconducting gap Δ(T) [J].
    """

    T = float(temperature)

    if T < 0.0:
        raise ValueError("Temperature cannot be negative.")

    if T >= params.Tc:
        return 0.0

    t = T / params.Tc

    return params.delta_0 * normalized_gap_bcs(t)


# ---------------------------------------------------------------------------
# Kinetic inductance
# ---------------------------------------------------------------------------

def kinetic_inductance_per_square_zero_temperature(
    params: SuperconductorParameters,
) -> float:
    """
    Dirty-limit zero-temperature kinetic inductance per square.

        Lk□(0) = ħ R□ / (π Δ0)

    Units:
        H / square
    """

    return (
        H_BAR
        * params.sheet_resistance
        / (pi * params.delta_0)
    )


def kinetic_inductance_per_square(
    params: SuperconductorParameters,
    temperature: float,
) -> float:
    """
    Temperature-dependent dirty-limit kinetic inductance per square.

    Low-frequency superconducting response:

        Lk□(T)
        =
        Lk□(0)
        /
        [
            (Δ(T)/Δ0)
            tanh(Δ(T)/(2 kB T))
        ]

    This expression is evaluated only in the superconducting state.
    """

    T = float(temperature)

    if T < 0.0:
        raise ValueError("Temperature cannot be negative.")

    if T >= params.Tc:
        raise ValueError(
            "Kinetic inductance in this superconducting model is undefined "
            "at or above Tc."
        )

    if T == 0.0:
        return kinetic_inductance_per_square_zero_temperature(params)

    delta_T = superconducting_gap(params, T)

    if delta_T <= 0.0:
        raise RuntimeError(
            "Computed superconducting gap is non-positive below Tc."
        )

    delta_ratio = delta_T / params.delta_0

    denominator = (
        delta_ratio
        * tanh(delta_T / (2.0 * K_B * T))
    )

    if denominator <= 0.0:
        raise RuntimeError(
            "Invalid kinetic-inductance temperature factor."
        )

    return (
        kinetic_inductance_per_square_zero_temperature(params)
        / denominator
    )


def kinetic_inductance(
    params: SuperconductorParameters,
    length: float,
    width: float,
    temperature: float,
) -> float:
    """
    Total kinetic inductance of a uniform nanowire.

        Lk = Lk□ * (length / width)

    Parameters
    ----------
    length:
        Electrical nanowire length [m].

    width:
        Nanowire width [m].
    """

    if length <= 0.0:
        raise ValueError("length must be positive.")

    if width <= 0.0:
        raise ValueError("width must be positive.")

    return (
        kinetic_inductance_per_square(params, temperature)
        * length
        / width
    )


# ---------------------------------------------------------------------------
# Dirty-limit depairing current
# ---------------------------------------------------------------------------

def kl_correction_factor(
    temperature: float,
    Tc: float,
) -> float:
    """
    Kupriyanov-Lukichev temperature correction.

        C(T) =
            0.65 *
            sqrt(3 - (T/Tc)^5)

    ------------------------------------------------
    Note:
        This factor is used in the dirty-limit
        depairing-current expression implemented below.
    """

    if Tc <= 0.0:
        raise ValueError("Tc must be positive.")

    T = float(temperature)

    if T < 0.0:
        raise ValueError("Temperature cannot be negative.")

    if T >= Tc:
        return 0.0

    t = T / Tc

    return 0.65 * sqrt(3.0 - t**5)


def depairing_current_density(
    params: SuperconductorParameters,
    temperature: float,
) -> float:
    """
    Dirty-limit depairing current density.

    Uses the Kupriyanov-Lukichev corrected expression:

        j_dep(T)
        =
        j_dep,GL(0)
        *
        [1 - (T/Tc)^2]^(3/2)
        *
        C(T)

    where

        j_dep,GL(0)
        =
        [4 sqrt(pi) exp(2γ)]
        /
        [21 sqrt(3) ζ(3)]
        *
        [Δ0^2]
        /
        [e ρ_n sqrt(D ħ k_B Tc)]

    The material diffusion coefficient D and normal-state resistivity
    are explicit inputs.
    """

    T = float(temperature)

    if T < 0.0:
        raise ValueError("Temperature cannot be negative.")

    if T >= params.Tc:
        return 0.0

    gamma_e = 0.5772156649015329
    zeta_3 = 1.202056903159594

    prefactor = (
        4.0
        * sqrt(pi)
        * exp(2.0 * gamma_e)
        /
        (21.0 * sqrt(3.0) * zeta_3)
    )

    denominator = (
        E_CHARGE
        * params.rho_n
        * sqrt(
            params.diffusion_coefficient
            * H_BAR
            * K_B
            * params.Tc
        )
    )

    if denominator <= 0.0:
        raise RuntimeError(
            "Invalid denominator in depairing-current calculation."
        )

    j_dep_0 = (
        prefactor
        * params.delta_0**2
        / denominator
    )

    t = T / params.Tc

    return (
        j_dep_0
        * (1.0 - t**2)**1.5
        * kl_correction_factor(T, params.Tc)
    )


def depairing_current(
    params: SuperconductorParameters,
    width: float,
    temperature: float,
) -> float:
    """
    Depairing current:

        I_dep = j_dep * A

    with

        A = width * thickness
    """

    if width <= 0.0:
        raise ValueError("width must be positive.")

    return (
        depairing_current_density(params, temperature)
        * width
        * params.thickness
    )