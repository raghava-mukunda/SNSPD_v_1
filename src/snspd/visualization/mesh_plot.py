# FILE: src/snspd/visualization/mesh_plot.py
# PURPOSE:
# Visualizes the triangular FEM mesh generated for the SNSPD geometry.
#
# This is a diagnostic tool used to inspect mesh density, topology,
# and element quality before numerical physics is solved.

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from snspd.mesh.mesh import Mesh


def plot_mesh(
    mesh: Mesh,
    show_nodes: bool = False,
) -> None:
    """
    Plot the triangular FEM mesh.

    Parameters
    ----------
    mesh:
        Canonical FEM mesh.

    show_nodes:
        Display mesh nodes if True.
    """

    x = mesh.nodes[:, 0] * 1e6
    y = mesh.nodes[:, 1] * 1e6

    triangulation = mtri.Triangulation(
        x,
        y,
        mesh.triangles,
    )

    fig, ax = plt.subplots()

    ax.triplot(
        triangulation,
        linewidth=0.5,
    )

    if show_nodes:

        ax.scatter(
            x,
            y,
            s=4,
        )

    ax.set_xlabel(
        "x [µm]"
    )

    ax.set_ylabel(
        "y [µm]"
    )

    ax.set_title(
        "SNSPD FEM Mesh"
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.grid(True)

    plt.tight_layout()

    plt.show()