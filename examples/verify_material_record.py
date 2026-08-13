# FILE: examples/verify_material_record.py
# PURPOSE:
# Verifies that SNSPD material parameters are traceable,
# source-labelled, and cannot silently disappear into the model.


from snspd.physics.materials.material_record import (
    MaterialRecord,
    PhysicalParameter,
    ParameterOrigin,
)

from snspd.physics.materials.nbn import (
    NbNMaterial,
)


def main() -> None:

    print("=" * 60)
    print("SNSPD MATERIAL RECORD VERIFICATION")
    print("=" * 60)

    # --------------------------------------------------------
    # This is an ARCHITECTURAL test record.
    #
    # It is NOT claimed to represent a real NbN film.
    # --------------------------------------------------------

    record = MaterialRecord(

        material_name="NbN",

        film_id="TEST-NBN-001",

        substrate="verification",

        deposition_method="verification",

        measurement_temperature_k=4.2,

        parameters=(

            PhysicalParameter(
                name="critical_temperature",
                value=10.0,
                unit="K",
                origin=ParameterOrigin.MEASURED,
                source="TEST_REFERENCE",
            ),

            PhysicalParameter(
                name="energy_gap_zero",
                value=2.435465e-22,
                unit="J",
                origin=ParameterOrigin.MEASURED,
                source="TEST_REFERENCE",
            ),

            PhysicalParameter(
                name="normal_resistivity",
                value=1.0e-6,
                unit="ohm m",
                origin=ParameterOrigin.MEASURED,
                source="TEST_REFERENCE",
            ),

            PhysicalParameter(
                name="diffusion_coefficient",
                value=5.0e-5,
                unit="m^2/s",
                origin=ParameterOrigin.MEASURED,
                source="TEST_REFERENCE",
            ),

            PhysicalParameter(
                name="film_thickness",
                value=5.0e-9,
                unit="m",
                origin=ParameterOrigin.MEASURED,
                source="TEST_REFERENCE",
            ),

            PhysicalParameter(
                name="penetration_depth_zero",
                value=350.0e-9,
                unit="m",
                origin=ParameterOrigin.MEASURED,
                source="TEST_REFERENCE",
            ),

            PhysicalParameter(
                name="coherence_length_zero",
                value=5.0e-9,
                unit="m",
                origin=ParameterOrigin.MEASURED,
                source="TEST_REFERENCE",
            ),
        ),
    )

    material = NbNMaterial(record)

    print("\nMaterial")
    print("--------")
    print(f"Name       : {record.material_name}")
    print(f"Film ID    : {record.film_id}")

    print("\nRequired parameters")
    print("-------------------")

    for name in NbNMaterial.REQUIRED_PARAMETERS:

        parameter = record.get(name)

        print(
            f"{name:25s}"
            f"{parameter.value:.6e} "
            f"{parameter.unit}"
        )

        if not parameter.source:

            raise RuntimeError(
                f"Parameter {name} has no source."
            )

    print(
        "\nParameter provenance          : PASS"
    )

    # --------------------------------------------------------
    # Temperature sweep
    # --------------------------------------------------------

    print("\nTemperature sweep")
    print("-----------------")

    for temperature in [
        0.0,
        2.0,
        4.0,
        6.0,
        8.0,
        9.0,
    ]:

        gap = material.gap(
            temperature
        )

        density = (
            material.normalized_superfluid_density(
                temperature
            )
        )

        penetration = (
            material.penetration_depth(
                temperature
            )
        )

        print(
            f"T = {temperature:5.2f} K   "
            f"Delta = {gap:.6e} J   "
            f"ns/n0 = {density:.6f}   "
            f"lambda = {penetration:.6e} m"
        )

    # --------------------------------------------------------
    # Geometry coupling
    # --------------------------------------------------------

    Lk = material.kinetic_inductance(
        length_m=100.0e-6,
        width_m=100.0e-9,
        temperature_k=4.0,
    )

    print(
        "\nKinetic inductance"
    )

    print(
        f"Lk = {Lk:.6e} H"
    )

    if Lk <= 0.0:

        raise RuntimeError(
            "Kinetic inductance must be positive."
        )

    print(
        "Material → geometry coupling : PASS"
    )

    print("\n" + "=" * 60)
    print(
        "MATERIAL RECORD VERIFICATION : PASS"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()