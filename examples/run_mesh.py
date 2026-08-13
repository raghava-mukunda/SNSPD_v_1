# FILE: examples/run_mesh.py
# PURPOSE:
# Demonstrates the first complete SNSPD geometry-to-FEM-mesh pipeline:
#
#     SVG
#       ↓
#     DeviceGeometry
#       ↓
#     NanowireGeometry
#       ↓
#     Gmsh
#       ↓
#     Canonical Mesh
#       ↓
#     Mesh validation
#       ↓
#     Mesh quality analysis
#       ↓
#     Visualization

from pathlib import Path

from snspd.geometry.svg_importer import (
    import_svg,
)

from snspd.mesh.gmsh_mesher import (
    GmshMesher,
)

from snspd.mesh.quality import (
    analyze_mesh_quality,
    format_mesh_quality,
)

from snspd.visualization.mesh_plot import (
    plot_mesh,
)


def main():

    svg_file = (
        Path(__file__).parent
        / "simple_meander.svg"
    )

    print(
        "Loading SNSPD geometry..."
    )

    geometry = import_svg(
        svg_file
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
            "Cannot mesh invalid geometry."
        )

    print(
        "Geometry validation: PASS"
    )

    # --------------------------------------------------------
    # Mesh generation
    # --------------------------------------------------------

    # 0.25 um target element size.
    characteristic_length = (
        0.25e-6
    )

    print(
        "\nGenerating FEM mesh..."
    )

    mesher = GmshMesher(
        characteristic_length=(
            characteristic_length
        )
    )

    mesh = mesher.generate(
        geometry
    )

    print(
        mesh.summary()
    )

    # --------------------------------------------------------
    # Mesh validation
    # --------------------------------------------------------

    mesh_errors = (
        mesh.validate()
    )

    if mesh_errors:

        print(
            "\nMESH VALIDATION FAILED"
        )

        for error in mesh_errors:

            print(
                f"ERROR: {error}"
            )

        raise RuntimeError(
            "Invalid FEM mesh."
        )

    print(
        "Mesh validation: PASS"
    )

    # --------------------------------------------------------
    # Mesh quality
    # --------------------------------------------------------

    quality = (
        analyze_mesh_quality(
            mesh
        )
    )

    print(
        format_mesh_quality(
            quality
        )
    )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    plot_mesh(
        mesh,
        show_nodes=False,
    )


if __name__ == "__main__":

    main()