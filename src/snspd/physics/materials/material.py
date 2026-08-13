# FILE: src/snspd/physics/materials/material.py
# PURPOSE:
# Defines the base material interface for the SNSPD
# multiphysics engine.
#
# This module defines the constitutive interface used by:
#
#   - electromagnetic solvers
#   - superconducting electrodynamics
#   - thermal solvers
#   - quasiparticle solvers
#   - electrothermal coupling
#
# No numerical material parameters are assumed here.
#
# Every physical material must explicitly provide the
# constitutive quantities required by the solver.


from __future__ import annotations


from abc import ABC, abstractmethod

from dataclasses import dataclass


@dataclass(frozen=True)
class MaterialMetadata:
    """
    Metadata identifying a physical material model.

    Parameters
    ----------
    name:
        Material name.

    source:
        Literature reference, DOI, database identifier,
        experimental dataset, etc.

    model:
        Physical model used by the implementation.

    notes:
        Additional information about assumptions,
        validity range, fabrication process, etc.
    """

    name: str

    source: str

    model: str

    notes: str = ""


class MaterialModel(ABC):
    """
    Abstract base class for all SNSPD material models.

    A concrete material implementation must provide the
    constitutive relations required by the multiphysics solver.

    Electromagnetic quantities:

        epsilon_r(omega, T)
        mu_r(omega, T)
        sigma(omega, T)

    Thermal quantities:

        C_e(T)
        C_ph(T)
        kappa(T)

    Superconducting quantities:

        Tc
        Delta(T)
        lambda(T)
        xi(T)
        D(T)
    """

    def __init__(
        self,
        metadata: MaterialMetadata,
    ) -> None:

        self.metadata = metadata

    # ========================================================
    # ELECTROMAGNETIC CONSTITUTIVE RELATIONS
    # ========================================================

    @abstractmethod
    def permittivity(
        self,
        frequency_hz: float,
        temperature_k: float,
    ) -> complex:
        """
        Relative complex permittivity:

            epsilon_r(omega, T)

        Constitutive relation:

            D = epsilon_0 epsilon_r E
        """

        raise NotImplementedError

    @abstractmethod
    def permeability(
        self,
        frequency_hz: float,
        temperature_k: float,
    ) -> complex:
        """
        Relative complex permeability:

            mu_r(omega, T)

        Constitutive relation:

            B = mu_0 mu_r H
        """

        raise NotImplementedError

    @abstractmethod
    def conductivity(
        self,
        frequency_hz: float,
        temperature_k: float,
    ) -> complex:
        """
        Complex electrical conductivity:

            sigma(omega, T)

        Constitutive relation:

            J = sigma E
        """

        raise NotImplementedError

    # ========================================================
    # THERMAL PROPERTIES
    # ========================================================

    @abstractmethod
    def electronic_heat_capacity(
        self,
        temperature_k: float,
    ) -> float:
        """
        Electronic heat capacity per unit volume:

            C_e(T)

        Units:

            J m^-3 K^-1
        """

        raise NotImplementedError

    @abstractmethod
    def phonon_heat_capacity(
        self,
        temperature_k: float,
    ) -> float:
        """
        Phonon heat capacity per unit volume:

            C_ph(T)

        Units:

            J m^-3 K^-1
        """

        raise NotImplementedError

    @abstractmethod
    def thermal_conductivity(
        self,
        temperature_k: float,
    ) -> float:
        """
        Thermal conductivity:

            kappa(T)

        Units:

            W m^-1 K^-1
        """

        raise NotImplementedError

    # ========================================================
    # SUPERCONDUCTING PROPERTIES
    # ========================================================

    @abstractmethod
    def critical_temperature(
        self,
    ) -> float:
        """
        Superconducting critical temperature:

            Tc
        """

        raise NotImplementedError

    @abstractmethod
    def superconducting_gap(
        self,
        temperature_k: float,
    ) -> float:
        """
        Superconducting energy gap:

            Delta(T)

        Units:

            joules
        """

        raise NotImplementedError

    @abstractmethod
    def london_penetration_depth(
        self,
        temperature_k: float,
    ) -> float:
        """
        London penetration depth:

            lambda(T)

        Units:

            metres
        """

        raise NotImplementedError

    @abstractmethod
    def coherence_length(
        self,
        temperature_k: float,
    ) -> float:
        """
        Superconducting coherence length:

            xi(T)

        Units:

            metres
        """

        raise NotImplementedError

    @abstractmethod
    def diffusion_coefficient(
        self,
        temperature_k: float,
    ) -> float:
        """
        Quasiparticle diffusion coefficient:

            D(T)

        Units:

            m^2 s^-1
        """

        raise NotImplementedError

    # ========================================================
    # STATE VALIDATION
    # ========================================================

    def validate_state(
        self,
        frequency_hz: float,
        temperature_k: float,
    ) -> None:
        """
        Validate the physical state before evaluating
        constitutive equations.
        """

        if frequency_hz < 0.0:

            raise ValueError(
                "Frequency cannot be negative."
            )

        if temperature_k <= 0.0:

            raise ValueError(
                "Temperature must be greater than zero."
            )

        tc = self.critical_temperature()

        if tc <= 0.0:

            raise ValueError(
                "Critical temperature must be positive."
            )

    # ========================================================
    # MATERIAL STATE SUMMARY
    # ========================================================

    def state_summary(
        self,
        frequency_hz: float,
        temperature_k: float,
    ) -> dict:
        """
        Evaluate all currently available constitutive
        quantities for a specified thermodynamic/electromagnetic
        state.
        """

        self.validate_state(
            frequency_hz,
            temperature_k,
        )

        return {
            "material":
                self.metadata.name,

            "model":
                self.metadata.model,

            "frequency_hz":
                frequency_hz,

            "temperature_k":
                temperature_k,

            "critical_temperature_k":
                self.critical_temperature(),

            "permittivity":
                self.permittivity(
                    frequency_hz,
                    temperature_k,
                ),

            "permeability":
                self.permeability(
                    frequency_hz,
                    temperature_k,
                ),

            "conductivity_s_per_m":
                self.conductivity(
                    frequency_hz,
                    temperature_k,
                ),

            "gap_j":
                self.superconducting_gap(
                    temperature_k,
                ),

            "penetration_depth_m":
                self.london_penetration_depth(
                    temperature_k,
                ),

            "coherence_length_m":
                self.coherence_length(
                    temperature_k,
                ),

            "diffusion_coefficient_m2_per_s":
                self.diffusion_coefficient(
                    temperature_k,
                ),

            "electronic_heat_capacity":
                self.electronic_heat_capacity(
                    temperature_k,
                ),

            "phonon_heat_capacity":
                self.phonon_heat_capacity(
                    temperature_k,
                ),

            "thermal_conductivity":
                self.thermal_conductivity(
                    temperature_k,
                ),
        }
    