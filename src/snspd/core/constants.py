# FILE: src/snspd/core/constants.py
# PURPOSE:
# Provides the single source of truth for universal physical constants
# used throughout the SNSPD digital twin.
#
# All constants are represented in SI units.
# No physics module should hard-code fundamental constants independently.

from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicalConstants:
    """
    Immutable collection of fundamental physical constants.

    All quantities are expressed in SI units.
    """

    # Speed of light in vacuum [m/s]
    c: float = 299_792_458.0

    # Planck constant [J s]
    h: float = 6.626_070_15e-34

    # Reduced Planck constant [J s]
    hbar: float = 1.054_571_817e-34

    # Elementary charge [C]
    e: float = 1.602_176_634e-19

    # Boltzmann constant [J/K]
    k_B: float = 1.380_649e-23

    # Vacuum permeability [H/m]
    mu_0: float = 1.256_637_062_12e-6

    # Vacuum permittivity [F/m]
    epsilon_0: float = 8.854_187_8128e-12


# Global immutable constants object.
CONSTANTS = PhysicalConstants()