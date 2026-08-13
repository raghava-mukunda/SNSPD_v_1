from __future__ import annotations

from pathlib import Path

from snspd.geometry.nanowire_svg import (
    SVGExtractionConfig,
    extract_nanowire_geometry,
)


def main() -> None:

    print(
        "\n"
        "====================================================\n"
        "SNSPD SVG NANOWIRE EXTRACTION VERIFICATION\n"
        "====================================================\n"
    )

    svg_file = (
        Path(__file__).resolve().parents[1]
        / "nanowire.svg"
    )

    if not svg_file.exists():

        raise FileNotFoundError(
            f"SVG not found:\n{svg_file}"
        )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # CHANGE THIS AFTER WE KNOW THE PHYSICAL SCALE.
    #
    # Example:
    #
    # 1 SVG unit = 1 um
    #
    #     1e-6
    #
    # --------------------------------------------------------

    meters_per_svg_unit = 1.0e-6

    config = SVGExtractionConfig(
        meters_per_svg_unit=(
            meters_per_svg_unit
        ),

        minimum_area_svg_units2=1.0,

        samples_per_segment=8,

        minimum_red=220,

        maximum_green=240,

        maximum_blue=240,

        background_distance_threshold=8,

        minimum_component_area_m2=(
            1.0e-18
        ),
    )

    geometry = extract_nanowire_geometry(
        svg_file,
        config,
    )

    print(
        geometry.summary()
    )

    errors = geometry.validate()

    if errors:

        print(
            "\nGEOMETRY VALIDATION FAILED"
        )

        for error in errors:

            print(
                f"ERROR: {error}"
            )

        raise RuntimeError(
            "Extracted SVG geometry is invalid."
        )

    print(
        "\nGeometry validation : PASS"
    )

    print(
        "\nExtraction metadata"
    )

    print(
        "-------------------"
    )

    for key, value in (
        geometry.metadata.items()
    ):

        print(
            f"{key:30s}: {value}"
        )

    print(
        "\nSVG nanowire extraction : PASS"
    )


if __name__ == "__main__":

    main()