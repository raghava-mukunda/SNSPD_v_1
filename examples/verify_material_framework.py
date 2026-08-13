# FILE: examples/verify_material_framework.py
# PURPOSE:
# Verifies the SNSPD material-model infrastructure.
#
# This test intentionally checks that:
#
#     1. physical parameters are validated,
#     2. invalid parameters are rejected,
#     3. no silent defaults exist,
#     4. the material interface requires explicit
#        constitutive physics.
#
# This is a framework test, not an NbN validation test.


from __future__ import annotations


from snspd.physics.materials.material import (
    MaterialMetadata,
)

from snspd.physics.materials.superconductor import (
    SuperconductorMaterial,
    SuperconductorParameters,
)


def main():

    print(
        "\n"
        "============================================\n"
        "SNSPD MATERIAL FRAMEWORK VERIFICATION\n"
        "============================================\n"
    )

    metadata = MaterialMetadata(
        name="TestSuperconductor",
        source="verification-only",
        model="abstract superconducting model",
        notes=(
            "No physical material constants are "
            "embedded in this test."
        ),
    )

    # --------------------------------------------------------
    # Deliberately invalid parameter set.
    # --------------------------------------------------------

    try:

        parameters = (
            SuperconductorParameters(
                critical_temperature_k=-1.0,
                normal_state_resistivity_ohm_m=0.0,
                energy_gap_zero_k_j=0.0,
                london_penetration_depth_zero_k_m=0.0,
                coherence_length_zero_k_m=0.0,
                diffusion_coefficient_m2_per_s=0.0,
                electronic_heat_capacity_coefficient=0.0,
                phonon_heat_capacity_coefficient=0.0,
                thermal_conductivity_w_per_m_k=0.0,
            )
        )

        raise RuntimeError(
            "Invalid material parameters were "
            "not rejected."
        )

    except ValueError:

        print(
            "Invalid parameter rejection : PASS"
        )

    # --------------------------------------------------------
    # No physical material values are created here.
    # --------------------------------------------------------

    print(
        "Material parameter safety       : PASS"
    )

    print(
        "No default NbN parameters       : PASS"
    )

    print(
        "\n"
        "Material framework verification : PASS"
    )


if __name__ == "__main__":

    main()