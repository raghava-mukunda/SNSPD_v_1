# FILE: src/snspd/physics/materials/nbn.py
# PURPOSE:
# Material-specific NbN superconducting model.
#
# This module connects a traceable NbN MaterialRecord to the
# generic superconducting mathematical models.
#
# IMPORTANT:
# No universal NbN numerical parameters are defined here.
#
# A particular NbN film must be supplied through MaterialRecord.
#
# The purpose of this class is to specify:
#
#   - which physical models are used
#   - which material-record parameters are required
#   - how those parameters enter the models
#
# Different NbN films can therefore be represented independently.


from __future__ import annotations


from math import sqrt


from snspd.physics.materials.material_record import (
    MaterialRecord,
)


from snspd.physics.materials.models.bcs import (
    normalized_gap,
    superconducting_gap,
    normalized_superfluid_density,
    normalized_penetration_depth,
)


class NbNMaterial:
    """
    Traceable superconducting NbN film.

    This class currently provides the equilibrium superconducting
    relations that can be derived from the supplied material record.

    No default material parameters are allowed.
    """

    MATERIAL_NAME = "NbN"

    REQUIRED_PARAMETERS = (
        "critical_temperature",
        "energy_gap_zero",
        "normal_resistivity",
        "diffusion_coefficient",
        "film_thickness",
        "penetration_depth_zero",
        "coherence_length_zero",
    )

    def __init__(
        self,
        record: MaterialRecord,
    ) -> None:

        if record.material_name != self.MATERIAL_NAME:

            raise ValueError(
                "NbNMaterial requires "
                "material_name='NbN'. "
                f"Received '{record.material_name}'."
            )

        self.record = record

        self._validate_required_parameters()

    # ========================================================
    # PARAMETER ACCESS
    # ========================================================

    def parameter(
        self,
        name: str,
    ) -> float:

        return self.record.value(name)

    def _validate_required_parameters(
        self,
    ) -> None:

        for name in self.REQUIRED_PARAMETERS:

            try:

                self.record.get(name)

            except KeyError as exc:

                raise ValueError(
                    "NbN material record is incomplete. "
                    f"Missing required parameter: '{name}'."
                ) from exc

    # ========================================================
    # MATERIAL PARAMETERS
    # ========================================================

    @property
    def Tc(self) -> float:

        return self.parameter(
            "critical_temperature"
        )

    @property
    def delta_zero(self) -> float:

        return self.parameter(
            "energy_gap_zero"
        )

    @property
    def rho_n(self) -> float:

        return self.parameter(
            "normal_resistivity"
        )

    @property
    def diffusion_coefficient(self) -> float:

        return self.parameter(
            "diffusion_coefficient"
        )

    @property
    def thickness(self) -> float:

        return self.parameter(
            "film_thickness"
        )

    @property
    def lambda_zero(self) -> float:

        return self.parameter(
            "penetration_depth_zero"
        )

    @property
    def xi_zero(self) -> float:

        return self.parameter(
            "coherence_length_zero"
        )

    # ========================================================
    # REDUCED TEMPERATURE
    # ========================================================

    def reduced_temperature(
        self,
        temperature_k: float,
    ) -> float:

        if temperature_k < 0.0:

            raise ValueError(
                "Temperature cannot be negative."
            )

        if self.Tc <= 0.0:

            raise RuntimeError(
                "Critical temperature must be positive."
            )

        return (
            temperature_k
            / self.Tc
        )

    # ========================================================
    # SUPERCONDUCTING GAP
    # ========================================================

    def gap(
        self,
        temperature_k: float,
    ) -> float:

        if temperature_k >= self.Tc:

            return 0.0

        return superconducting_gap(
            self.delta_zero,
            self.reduced_temperature(
                temperature_k
            ),
        )

    def normalized_gap(
        self,
        temperature_k: float,
    ) -> float:

        if temperature_k >= self.Tc:

            return 0.0

        return normalized_gap(
            self.reduced_temperature(
                temperature_k
            )
        )

    # ========================================================
    # SUPERFLUID DENSITY
    # ========================================================

    def normalized_superfluid_density(
        self,
        temperature_k: float,
    ) -> float:

        if temperature_k >= self.Tc:

            return 0.0

        return normalized_superfluid_density(
            self.reduced_temperature(
                temperature_k
            )
        )

    # ========================================================
    # PENETRATION DEPTH
    # ========================================================

    def penetration_depth(
        self,
        temperature_k: float,
    ) -> float:

        if temperature_k >= self.Tc:

            raise ValueError(
                "Superconducting penetration depth "
                "model is undefined at or above Tc."
            )

        return (
            self.lambda_zero
            * normalized_penetration_depth(
                self.reduced_temperature(
                    temperature_k
                )
            )
        )

    # ========================================================
    # SHEET RESISTANCE
    # ========================================================

    @property
    def normal_sheet_resistance(self) -> float:

        return (
            self.rho_n
            / self.thickness
        )

    # ========================================================
    # GEOMETRY-DEPENDENT KINETIC INDUCTANCE
    # ========================================================

    def kinetic_inductance(
        self,
        length_m: float,
        width_m: float,
        temperature_k: float,
    ) -> float:
        """
        Thin-film kinetic inductance:

            Lk = μ0 λ(T)^2 L / (w t)

        where:

            λ(T) = London penetration depth
            L    = wire length
            w    = wire width
            t    = film thickness

        This formulation explicitly connects the material model
        to the device geometry.
        """

        if length_m <= 0.0:

            raise ValueError(
                "length_m must be positive."
            )

        if width_m <= 0.0:

            raise ValueError(
                "width_m must be positive."
            )

        lambda_t = self.penetration_depth(
            temperature_k
        )

        return (
            4.0e-7
            * 3.141592653589793
            * lambda_t**2
            * length_m
            /
            (
                width_m
                * self.thickness
            )
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    def summary(self) -> dict:

        return {
            "material":
                self.record.material_name,

            "film_id":
                self.record.film_id,

            "substrate":
                self.record.substrate,

            "deposition_method":
                self.record.deposition_method,

            "measurement_temperature_k":
                self.record.measurement_temperature_k,

            "parameters": {
                parameter.name: {
                    "value": parameter.value,
                    "unit": parameter.unit,
                    "origin": parameter.origin.value,
                    "source": parameter.source,
                    "uncertainty": parameter.uncertainty,
                    "notes": parameter.notes,
                }
                for parameter
                in self.record.parameters
            },
        }