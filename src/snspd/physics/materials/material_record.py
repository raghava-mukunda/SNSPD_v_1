# FILE: src/snspd/physics/materials/material_record.py
# PURPOSE:
# Defines traceable experimental/material records for SNSPD
# superconducting films.
#
# Every experimentally supplied parameter must carry:
#
#   1. value
#   2. SI unit
#   3. measurement/model origin
#   4. source/reference
#   5. uncertainty, when available
#
# The simulator must distinguish:
#
#   measured
#   derived
#   assumed
#   calculated
#
# No silent material parameters are permitted.


from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ParameterOrigin(Enum):
    """
    Origin of a material parameter.
    """

    MEASURED = "measured"

    DERIVED = "derived"

    LITERATURE = "literature"

    ASSUMED = "assumed"

    CALCULATED = "calculated"


@dataclass(frozen=True)
class PhysicalParameter:
    """
    Traceable physical parameter.

    Parameters
    ----------
    name:
        Parameter name.

    value:
        Numerical value in SI units.

    unit:
        Human-readable SI unit.

    origin:
        Experimental/literature/derived origin.

    source:
        Reference identifier.

    uncertainty:
        One-sigma or otherwise explicitly defined uncertainty.
        None if unavailable.

    notes:
        Additional information about measurement conditions,
        film thickness, temperature, frequency, etc.
    """

    name: str

    value: float

    unit: str

    origin: ParameterOrigin

    source: str

    uncertainty: float | None = None

    notes: str = ""

    def __post_init__(self) -> None:

        if not self.name.strip():
            raise ValueError(
                "Parameter name cannot be empty."
            )

        if not self.unit.strip():
            raise ValueError(
                "Parameter unit cannot be empty."
            )

        if not self.source.strip():
            raise ValueError(
                "Every physical parameter must have "
                "a source/reference."
            )

        if self.uncertainty is not None:

            if self.uncertainty < 0.0:

                raise ValueError(
                    "Uncertainty cannot be negative."
                )


@dataclass(frozen=True)
class MaterialRecord:
    """
    Complete traceable record for one superconducting film.

    This represents an actual material/process state, not merely
    the chemical composition.

    For SNSPDs this distinction is critical because properties
    depend on film thickness, disorder, deposition conditions,
    stoichiometry, annealing, substrate, and geometry.
    """

    material_name: str

    film_id: str

    parameters: tuple[PhysicalParameter, ...]

    fabrication_notes: str = ""

    substrate: str = ""

    deposition_method: str = ""

    measurement_temperature_k: float | None = None

    def __post_init__(self) -> None:

        if not self.material_name.strip():

            raise ValueError(
                "Material name cannot be empty."
            )

        if not self.film_id.strip():

            raise ValueError(
                "Film ID cannot be empty."
            )

        if len(self.parameters) == 0:

            raise ValueError(
                "Material record must contain "
                "at least one physical parameter."
            )

    def get(
        self,
        name: str,
    ) -> PhysicalParameter:
        """
        Retrieve a parameter by exact name.
        """

        matches = [
            parameter
            for parameter in self.parameters
            if parameter.name == name
        ]

        if len(matches) == 0:

            raise KeyError(
                f"Parameter '{name}' is not present "
                f"in material record '{self.film_id}'."
            )

        if len(matches) > 1:

            raise ValueError(
                f"Duplicate parameter '{name}' found "
                f"in material record '{self.film_id}'."
            )

        return matches[0]

    def value(
        self,
        name: str,
    ) -> float:
        """
        Return numerical parameter value.
        """

        return self.get(name).value