"""
Verification of the superconducting constitutive model.

IMPORTANT:
No NbN material values are embedded here.

The verification first checks mathematical identities using
synthetic but explicitly declared parameters, then verifies
physical monotonicity of the constitutive relations.

For actual NbN validation, replace the test material parameters
with experimentally measured/literature-traceable values.
"""

from math import isclose

from snspd.physics.materials.superconductor import (
    K_B,
    E_CHARGE,
    SuperconductorParameters,
    superconducting_gap,
    normalized_gap_bcs,
    kinetic_inductance_per_square_zero_temperature,
    kinetic_inductance_per_square,
    kinetic_inductance,
    depairing_current_density,
    depairing_current,
)


def check(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(f"FAIL: {message}")


def main() -> None:

    print("=" * 60)
    print("SNSPD SUPERCONDUCTING CONSTITUTIVE MODEL VERIFICATION")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Explicitly declared verification material.
    #
    # This is NOT a default NbN material.
    # ------------------------------------------------------------------

    Tc = 10.0
    gap_ratio = 1.764

    delta_0 = gap_ratio * K_B * Tc

    rho_n = 1.0e-6
    diffusion_coefficient = 5.0e-5
    thickness = 5.0e-9

    material = SuperconductorParameters(
        Tc=Tc,
        delta_0=delta_0,
        rho_n=rho_n,
        diffusion_coefficient=diffusion_coefficient,
        thickness=thickness,
        gap_ratio=gap_ratio,
    )

    print("\nMaterial definition")
    print("-------------------")
    print(f"Tc              = {material.Tc:.6e} K")
    print(f"Delta(0)        = {material.delta_0:.6e} J")
    print(f"Delta(0)        = {material.delta_0_ev:.6e} eV")
    print(f"rho_n           = {material.rho_n:.6e} ohm m")
    print(f"D               = {material.diffusion_coefficient:.6e} m^2/s")
    print(f"thickness       = {material.thickness:.6e} m")
    print(f"R_square        = {material.sheet_resistance:.6e} ohm")

    # ------------------------------------------------------------------
    # 1. Gap at zero temperature
    # ------------------------------------------------------------------

    print("\n1. ZERO-TEMPERATURE GAP")

    check(
        isclose(
            normalized_gap_bcs(0.0),
            1.0,
            rel_tol=0.0,
            abs_tol=1e-14,
        ),
        "Delta(0)/Delta0 != 1",
    )

    check(
        isclose(
            superconducting_gap(material, 0.0),
            delta_0,
            rel_tol=1e-12,
        ),
        "Delta(0) incorrect",
    )

    print("Gap normalization              : PASS")

    # ------------------------------------------------------------------
    # 2. Gap must decrease monotonically with temperature
    # ------------------------------------------------------------------

    print("\n2. GAP TEMPERATURE DEPENDENCE")

    temperatures = [
        0.0,
        0.2 * Tc,
        0.4 * Tc,
        0.6 * Tc,
        0.8 * Tc,
        0.9 * Tc,
        0.99 * Tc,
    ]

    gaps = [
        superconducting_gap(material, T)
        for T in temperatures
    ]

    for a, b in zip(gaps[:-1], gaps[1:]):
        check(
            b < a,
            "Superconducting gap is not monotonically decreasing.",
        )

    check(
        superconducting_gap(material, Tc) == 0.0,
        "Gap does not vanish at Tc.",
    )

    print("Delta(T) monotonic decrease     : PASS")
    print("Delta(Tc) = 0                   : PASS")

    # ------------------------------------------------------------------
    # 3. Kinetic inductance
    # ------------------------------------------------------------------

    print("\n3. KINETIC INDUCTANCE")

    Lk0 = kinetic_inductance_per_square_zero_temperature(material)

    check(
        Lk0 > 0.0,
        "Zero-temperature kinetic inductance is non-positive.",
    )

    print(
        f"Lk_square(0)                   : {Lk0:.6e} H/square"
    )

    Lk_values = [
        kinetic_inductance_per_square(material, T)
        for T in [
            0.0,
            0.2 * Tc,
            0.4 * Tc,
            0.6 * Tc,
            0.8 * Tc,
            0.9 * Tc,
        ]
    ]

    for a, b in zip(Lk_values[:-1], Lk_values[1:]):
        check(
            b > a,
            "Kinetic inductance does not increase toward Tc.",
        )

    print("Lk(T) monotonic increase        : PASS")

    # ------------------------------------------------------------------
    # 4. Device kinetic inductance
    # ------------------------------------------------------------------

    length = 100.0e-6
    width = 100.0e-9
    temperature = 0.4 * Tc

    Lk = kinetic_inductance(
        material,
        length=length,
        width=width,
        temperature=temperature,
    )

    check(
        Lk > 0.0,
        "Device kinetic inductance is non-positive.",
    )

    print(
        f"Device Lk                      : {Lk:.6e} H"
    )
    print("Device kinetic inductance       : PASS")

    # ------------------------------------------------------------------
    # 5. Depairing current density
    # ------------------------------------------------------------------

    print("\n4. DEPAIRING CURRENT")

    j_values = [
        depairing_current_density(material, T)
        for T in [
            0.0,
            0.2 * Tc,
            0.4 * Tc,
            0.6 * Tc,
            0.8 * Tc,
            0.9 * Tc,
        ]
    ]

    for j in j_values:
        check(
            j >= 0.0,
            "Negative depairing current density.",
        )

    for a, b in zip(j_values[:-1], j_values[1:]):
        check(
            b < a,
            "Depairing current density does not decrease with T.",
        )

    check(
        depairing_current_density(material, Tc) == 0.0,
        "Depairing current density does not vanish at Tc.",
    )

    print(
        f"j_dep(0)                      : {j_values[0]:.6e} A/m^2"
    )
    print("j_dep(T) monotonic decrease     : PASS")
    print("j_dep(Tc) = 0                   : PASS")

    # ------------------------------------------------------------------
    # 6. Depairing current
    # ------------------------------------------------------------------

    I_dep = depairing_current(
        material,
        width=width,
        temperature=0.4 * Tc,
    )

    check(
        I_dep > 0.0,
        "Depairing current is non-positive.",
    )

    print(
        f"I_dep(0.4Tc)                  : {I_dep:.6e} A"
    )

    print("Depairing current calculation   : PASS")

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------

    print("\n" + "=" * 60)
    print("SUPERCONDUCTING MODEL VERIFICATION : PASS")
    print("=" * 60)


if __name__ == "__main__":
    main()