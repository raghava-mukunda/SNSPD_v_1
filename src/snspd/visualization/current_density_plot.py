# FILE: src/snspd/visualization/current_density_plot.py
#
# PURPOSE:
# Visualize FEM-computed SNSPD current density.
#
# The visualization operates directly on the canonical FEM mesh
# and element-wise current-density solution.
#
# Quantities visualized:
#
#     |J|                 : current-density magnitude [A/m^2]
#
#     C_J,local = |J| / J_transport
#
# where
#
#     J_transport = I / (w*t)
#
# is the nominal transport current density.
#
# The second quantity is particularly useful for SNSPD design because
# values > 1 indicate local current concentration relative to the
# nominal cross-sectional transport density.
#
# IMPORTANT:
# This module performs NO physics calculations.
# It only visualizes an already verified FEM solution.


from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.colors import LogNorm
from matplotlib.tri import Triangulation


def _validate_current_density(
    mesh,
    element_current_density: np.ndarray,
) -> np.ndarray:
    """
    Validate and return element-wise current-density magnitude.

    Parameters
    ----------
    mesh:
        Canonical FEM Mesh.

    element_current_density:
        Array with shape (N_elements, 2).

    Returns
    -------
    ndarray
        Current-density magnitude for every triangle.
    """

    J = np.asarray(
        element_current_density,
        dtype=float,
    )

    expected_shape = (
        mesh.element_count,
        2,
    )

    if J.shape != expected_shape:

        raise ValueError(
            "element_current_density must have shape "
            f"{expected_shape}, got {J.shape}."
        )

    if not np.all(
        np.isfinite(J)
    ):

        raise ValueError(
            "Current-density array contains "
            "non-finite values."
        )

    magnitude = np.linalg.norm(
        J,
        axis=1,
    )

    if np.any(
        magnitude < 0
    ):

        raise RuntimeError(
            "Current-density magnitude cannot be negative."
        )

    return magnitude


def plot_current_density(
    mesh,
    element_current_density: np.ndarray,
    *,
    terminal_current: float,
    nanowire_width: float,
    film_thickness: float,
    positive_terminal_nodes: np.ndarray | None = None,
    negative_terminal_nodes: np.ndarray | None = None,
    title: str = "SNSPD Current Density |J|",
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plot the FEM current-density distribution.

    Parameters
    ----------
    mesh:
        Canonical FEM mesh.

    element_current_density:
        FEM current density per triangular element [A/m^2].

    terminal_current:
        Transport current obtained from the FEM solution [A].

    nanowire_width:
        Nanowire width [m].

    film_thickness:
        Film thickness [m].

    positive_terminal_nodes:
        Optional positive-terminal node indices.

    negative_terminal_nodes:
        Optional negative-terminal node indices.

    title:
        Figure title.

    save_path:
        Optional output path.

    show:
        Whether to display the figure.
    """

    J_magnitude = _validate_current_density(
        mesh,
        element_current_density,
    )

    # --------------------------------------------------------
    # TRANSPORT CURRENT DENSITY
    # --------------------------------------------------------

    cross_sectional_area = (
        nanowire_width
        * film_thickness
    )

    if cross_sectional_area <= 0:

        raise ValueError(
            "Nanowire cross-sectional area must be positive."
        )

    if terminal_current <= 0:

        raise ValueError(
            "Terminal current must be positive."
        )

    J_transport = (
        terminal_current
        / cross_sectional_area
    )

    # Local normalized current density.
    local_crowding = (
        J_magnitude
        / J_transport
    )

    # --------------------------------------------------------
    # TRIANGULATION
    # --------------------------------------------------------

    triangulation = Triangulation(
        mesh.nodes[:, 0] * 1e6,
        mesh.nodes[:, 1] * 1e6,
        mesh.triangles,
    )

    # --------------------------------------------------------
    # HOTSPOT
    # --------------------------------------------------------

    hotspot_element = int(
        np.argmax(
            J_magnitude
        )
    )

    hotspot_J = float(
        J_magnitude[
            hotspot_element
        ]
    )

    hotspot_CJ = float(
        local_crowding[
            hotspot_element
        ]
    )

    hotspot_nodes = (
        mesh.triangles[
            hotspot_element
        ]
    )

    hotspot_center = np.mean(
        mesh.nodes[
            hotspot_nodes
        ],
        axis=0,
    )

    hotspot_x = (
        hotspot_center[0] * 1e6
    )

    hotspot_y = (
        hotspot_center[1] * 1e6
    )

    # --------------------------------------------------------
    # FIGURE
    # --------------------------------------------------------

    figure = plt.figure(
        figsize=(13, 8),
    )

    axes = figure.add_subplot(
        111
    )

    # --------------------------------------------------------
    # LOGARITHMIC CURRENT-DENSITY MAP
    # --------------------------------------------------------

    positive_values = (
        J_magnitude[
            J_magnitude > 0
        ]
    )

    if len(positive_values) == 0:

        raise RuntimeError(
            "All FEM current-density values are zero."
        )

    vmin = float(
        np.min(
            positive_values
        )
    )

    vmax = float(
        np.max(
            positive_values
        )
    )

    if np.isclose(
        vmin,
        vmax,
    ):

        # Avoid LogNorm failure for perfectly uniform fields.
        vmin = vmax * 0.999

    field = axes.tripcolor(
        triangulation,
        facecolors=J_magnitude,
        shading="flat",
        cmap="turbo",
        norm=LogNorm(
            vmin=vmin,
            vmax=vmax,
        ),
    )

    # --------------------------------------------------------
    # FEM MESH OVERLAY
    # --------------------------------------------------------

    axes.triplot(
        triangulation,
        color="black",
        linewidth=0.15,
        alpha=0.20,
    )

    # --------------------------------------------------------
    # TERMINAL NODES
    # --------------------------------------------------------

    if (
        positive_terminal_nodes is not None
        and len(positive_terminal_nodes) > 0
    ):

        positive = np.asarray(
            positive_terminal_nodes,
            dtype=int,
        )

        axes.scatter(
            mesh.nodes[
                positive,
                0,
            ] * 1e6,
            mesh.nodes[
                positive,
                1,
            ] * 1e6,
            s=4,
            marker="s",
            label="Positive terminal",
        )

    if (
        negative_terminal_nodes is not None
        and len(negative_terminal_nodes) > 0
    ):

        negative = np.asarray(
            negative_terminal_nodes,
            dtype=int,
        )

        axes.scatter(
            mesh.nodes[
                negative,
                0,
            ] * 1e6,
            mesh.nodes[
                negative,
                1,
            ] * 1e6,
            s=4,
            marker="s",
            label="Negative terminal",
        )

    # --------------------------------------------------------
    # HOTSPOT
    # --------------------------------------------------------

    axes.scatter(
        hotspot_x,
        hotspot_y,
        s=140,
        facecolors="none",
        edgecolors="white",
        linewidths=2.0,
        zorder=10,
    )

    axes.annotate(
        (
            f"MAX |J|\n"
            f"{hotspot_J:.4e} A/m²\n"
            f"C_J,local = {hotspot_CJ:.3f}"
        ),
        xy=(
            hotspot_x,
            hotspot_y,
        ),
        xytext=(
            12,
            12,
        ),
        textcoords="offset points",
        fontsize=9,
        bbox=dict(
            boxstyle="round",
            alpha=0.85,
        ),
        arrowprops=dict(
            arrowstyle="->",
        ),
    )

    # --------------------------------------------------------
    # COLORBAR
    # --------------------------------------------------------

    colorbar = figure.colorbar(
        field,
        ax=axes,
        pad=0.02,
    )

    colorbar.set_label(
        r"$|\mathbf{J}|$ [A/m$^2$]"
    )

    # --------------------------------------------------------
    # LABELS
    # --------------------------------------------------------

    axes.set_xlabel(
        "x [µm]"
    )

    axes.set_ylabel(
        "y [µm]"
    )

    axes.set_title(
        title
    )

    axes.set_aspect(
        "equal",
        adjustable="box",
    )

    axes.legend(
        loc="best"
    )

    axes.grid(
        alpha=0.15
    )

    # --------------------------------------------------------
    # INFORMATION BOX
    # --------------------------------------------------------

    information = (
        f"Transport current = "
        f"{terminal_current:.6e} A\n"
        f"J_transport = "
        f"{J_transport:.6e} A/m²\n"
        f"Maximum |J| = "
        f"{hotspot_J:.6e} A/m²\n"
        f"Maximum C_J = "
        f"{hotspot_CJ:.6f}\n"
        f"Hotspot = "
        f"({hotspot_x:.4f}, {hotspot_y:.4f}) µm"
    )

    axes.text(
        0.02,
        0.02,
        information,
        transform=axes.transAxes,
        verticalalignment="bottom",
        horizontalalignment="left",
        fontsize=9,
        bbox=dict(
            boxstyle="round",
            alpha=0.85,
        ),
    )

    figure.tight_layout()

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    if save_path is not None:

        save_path = Path(
            save_path
        )

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            f"\nCurrent-density visualization saved to:"
            f"\n{save_path}"
        )

    if show:

        plt.show()

    else:

        plt.close(
            figure
        )


def plot_crowding_factor(
    mesh,
    element_current_density: np.ndarray,
    *,
    terminal_current: float,
    nanowire_width: float,
    film_thickness: float,
    save_path: str | Path | None = None,
    show: bool = True,
) -> None:
    """
    Plot the normalized local current-crowding factor:

        C_J,local = |J| / J_transport

    where:

        J_transport = I / (w*t)

    This visualization is directly useful for comparing
    different SNSPD geometries.
    """

    J_magnitude = _validate_current_density(
        mesh,
        element_current_density,
    )

    cross_sectional_area = (
        nanowire_width
        * film_thickness
    )

    J_transport = (
        terminal_current
        / cross_sectional_area
    )

    crowding = (
        J_magnitude
        / J_transport
    )

    triangulation = Triangulation(
        mesh.nodes[:, 0] * 1e6,
        mesh.nodes[:, 1] * 1e6,
        mesh.triangles,
    )

    figure, axes = plt.subplots(
        figsize=(11, 8)
    )

    field = axes.tripcolor(
        triangulation,
        facecolors=crowding,
        shading="flat",
        cmap="turbo",
    )

    axes.triplot(
        triangulation,
        color="black",
        linewidth=0.15,
        alpha=0.20,
    )

    hotspot = int(
        np.argmax(
            crowding
        )
    )

    hotspot_nodes = (
        mesh.triangles[
            hotspot
        ]
    )

    center = np.mean(
        mesh.nodes[
            hotspot_nodes
        ],
        axis=0,
    )

    axes.scatter(
        center[0] * 1e6,
        center[1] * 1e6,
        s=150,
        facecolors="none",
        edgecolors="white",
        linewidths=2,
    )

    axes.annotate(
        (
            f"Maximum local crowding\n"
            f"$C_J$ = {crowding[hotspot]:.4f}"
        ),
        xy=(
            center[0] * 1e6,
            center[1] * 1e6,
        ),
        xytext=(
            12,
            12,
        ),
        textcoords="offset points",
        bbox=dict(
            boxstyle="round",
            alpha=0.85,
        ),
        arrowprops=dict(
            arrowstyle="->",
        ),
    )

    colorbar = figure.colorbar(
        field,
        ax=axes,
    )

    colorbar.set_label(
        r"Local crowding factor "
        r"$C_{J,\mathrm{local}} = |\mathbf{J}|/J_{\mathrm{transport}}$"
    )

    axes.set_xlabel(
        "x [µm]"
    )

    axes.set_ylabel(
        "y [µm]"
    )

    axes.set_title(
        "SNSPD LOCAL CURRENT-CROWDING FACTOR"
    )

    axes.set_aspect(
        "equal",
        adjustable="box",
    )

    axes.grid(
        alpha=0.15
    )

    figure.tight_layout()

    if save_path is not None:

        save_path = Path(
            save_path
        )

        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        figure.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight",
        )

        print(
            f"\nCrowding-factor visualization saved to:"
            f"\n{save_path}"
        )

    if show:

        plt.show()

    else:

        plt.close(
            figure
        )