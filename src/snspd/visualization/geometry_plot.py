# FILE: src/snspd
#/visualization/geometry_plot.py
# PURPOSE:
# Provides diagnostic visualization of the canonical SNSPD geometry.
#
# This module only visualizes geometry.
# It does not perform any physical simulation.

from __future__ import annotations

import matplotlib.pyplot as plt

from snspd.geometry.geometry import DeviceGeometry


def plot_geometry(
    geometry: DeviceGeometry,
    show_vertices: bool = False,
) -> None:
    """
    Plot the SNSPD geometry.

    Parameters
    ----------
    geometry:
        Canonical DeviceGeometry.

    show_vertices:
        Display polygon vertices if True.
    """

    fig, ax = plt.subplots()

    for index, region in enumerate(
        geometry.regions
    ):

        polygon = region.polygon

        x, y = polygon.exterior.xy

        # Convert meters to micrometers for visualization.
        x_um = [
            value * 1e6
            for value in x
        ]

        y_um = [
            value * 1e6
            for value in y
        ]

        ax.fill(
            x_um,
            y_um,
            alpha=0.5,
        )

        ax.plot(
            x_um,
            y_um,
        )

        if show_vertices:

            ax.scatter(
                x_um,
                y_um,
                s=10,
            )

            centroid = (
                polygon.centroid
            )

            ax.text(
                centroid.x * 1e6,
                centroid.y * 1e6,
                str(index),
            )

    ax.set_xlabel(
        "x [µm]"
    )

    ax.set_ylabel(
        "y [µm]"
    )

    ax.set_title(
        "SNSPD Device Geometry"
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.grid(True)

    plt.tight_layout()

    plt.show()