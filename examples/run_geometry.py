# FILE: examples/run_geometry.py
# PURPOSE:
# Runs the first complete SNSPD geometry pipeline:
#
#     SVG
#       ↓
#     SVG importer
#       ↓
#     DeviceGeometry
#       ↓
#     Geometry validation
#       ↓
#     Geometry analysis
#       ↓
#     Visualization
#
# This is the first integration test of the SNSPD software framework.

from pathlib import Path

from snspd.geometry.svg_importer import (
    import_svg,
)

from snspd.geometry.analyzer import (
    analyze_geometry,
    format_metrics,
)

from snspd.visualization.geometry_plot import (
    plot_geometry,
)


def main():

    svg_file = (
        Path(__file__).parent
        / "meander_processed.svg"
    )

    print(
        "Loading SVG..."
    )

    print(
        f"File: {svg_file}"
    )

    geometry = import_svg(
        svg_file
    )

    print(
        geometry.summary()
    )

    print(
        "Validating geometry..."
    )

    errors = geometry.validate()

    if errors:

        print(
            "\nSNSPD GEOMETRY "
            "VALIDATION FAILED"
        )

        print(
            "=========================="
        )

        for error in errors:

            print(
                f"ERROR: {error}"
            )

        raise RuntimeError(
            "Invalid SNSPD geometry."
        )

    print(
        "Geometry validation: PASS"
    )

    metrics = analyze_geometry(
        geometry
    )

    print(
        format_metrics(
            metrics
        )
    )

    plot_geometry(
        geometry,
        show_vertices=False,
    )


if __name__ == "__main__":

    main()