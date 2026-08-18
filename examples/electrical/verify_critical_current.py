"""
FILE: examples/electrical/verify_critical_current.py

PURPOSE
-------
Analytical verification of the Phase A/B critical-current model.

For a uniform straight nanowire:

    J_FEM(x,y) = J_transport

therefore

    C_J = 1

and

    Ic,geometry = Ic,straight.

This test verifies that the geometry-coupled critical-current
calculation reduces exactly to the analytical straight-wire
result when there is no current crowding.
"""

from __future__ import annotations

from snspd.physics.critical_current import (
    NanowireParameters,
    CriticalCurrentParameters,
    analyze_critical_current,
    validate_critical_current_consistency,
)


def main() -> None:

    width = 100e-9
    thickness = 10e-9

    jc = 1.0e11

    wire = NanowireParameters(
        width_m=width,
        thickness_m=thickness,
    )

    material = CriticalCurrentParameters(
        critical_current_density_A_m2=jc,
        material="verification_material",
        temperature_K=1.0,
    )

    cross_section = (
        width * thickness
    )

    transport_current = (
        1.0e-6
    )

    transport_J = (
        transport_current
        / cross_section
    )

    # Uniform analytical current density:
    #
    #     J_max = J_transport
    #
    # Hence:
    #
    #     C_J = 1
    #
    fem_jmax = transport_J

    result = analyze_critical_current(
        wire=wire,
        material=material,
        fem_transport_current_A=transport_current,
        fem_max_current_density_A_m2=fem_jmax,
    )

    validate_critical_current_consistency(
        result
    )

    crowding_error = abs(
        result.current_crowding_factor
        - 1.0
    )

    critical_current_error = abs(
        result.geometry_limited_critical_current_A
        - result.straight_wire_critical_current_A
    ) / result.straight_wire_critical_current_A

    print(
        "\n"
        "============================================\n"
        "CRITICAL CURRENT ANALYTICAL VERIFICATION\n"
        "============================================\n"
    )

    print(
        f"Width                 : "
        f"{width * 1e9:.3f} nm"
    )

    print(
        f"Thickness             : "
        f"{thickness * 1e9:.3f} nm"
    )

    print(
        f"Jc                    : "
        f"{jc:.6e} A/m²"
    )

    print(
        f"Analytical transport J: "
        f"{transport_J:.6e} A/m²"
    )

    print(
        f"\nCurrent crowding C_J  : "
        f"{result.current_crowding_factor:.12f}"
    )

    print(
        f"C_J error             : "
        f"{crowding_error:.6e}"
    )

    print(
        f"\nIc,straight           : "
        f"{result.straight_wire_critical_current_A:.6e} A"
    )

    print(
        f"Ic,geometry           : "
        f"{result.geometry_limited_critical_current_A:.6e} A"
    )

    print(
        f"Ic relative error     : "
        f"{critical_current_error:.6e}"
    )

    tolerance = 1.0e-12

    if crowding_error > tolerance:
        raise RuntimeError(
            "Current-crowding verification FAILED."
        )

    if critical_current_error > tolerance:
        raise RuntimeError(
            "Critical-current verification FAILED."
        )

    print(
        "\n"
        "Current-crowding baseline : PASS"
    )

    print(
        "Critical-current baseline : PASS"
    )

    print(
        "\n"
        "Phase A/B critical-current verification : PASS"
    )


if __name__ == "__main__":
    main()
