# FILE: src/snspd/mesh/mesh.py
# PURPOSE:
# Defines the canonical FEM mesh representation used by the SNSPD simulator.
#
# The physics solvers operate on this representation and do not depend
# directly on Gmsh.
#
# Current supported element:
#     - 2D linear triangular element (P1)
#
# Coordinates are stored in SI units [m].

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Mesh:
    """
    Canonical finite-element mesh.

    Parameters
    ----------
    nodes:
        Nx2 array containing node coordinates [m].

    triangles:
        Mx3 integer array containing triangle node indices.

    physical_regions:
        Optional physical-region identifier for each triangle.

    physical_names:
        Mapping from physical-region ID to human-readable name.

    metadata:
        Additional mesh information.
    """

    nodes: np.ndarray

    triangles: np.ndarray

    physical_regions: np.ndarray | None = None

    physical_names: dict[int, str] = field(
        default_factory=dict
    )

    metadata: dict = field(
        default_factory=dict
    )

    def __post_init__(self):
        """
        Validate and normalize mesh arrays.
        """

        self.nodes = np.asarray(
            self.nodes,
            dtype=float,
        )

        self.triangles = np.asarray(
            self.triangles,
            dtype=int,
        )

        if self.nodes.ndim != 2:
            raise ValueError(
                "Mesh nodes must be a 2D array."
            )

        if self.nodes.shape[1] != 2:
            raise ValueError(
                "Mesh nodes must have shape (N, 2)."
            )

        if self.triangles.ndim != 2:
            raise ValueError(
                "Mesh triangles must be a 2D array."
            )

        if self.triangles.shape[1] != 3:
            raise ValueError(
                "Triangles must have shape (M, 3)."
            )

        if self.physical_regions is not None:

            self.physical_regions = np.asarray(
                self.physical_regions,
                dtype=int,
            )

            if len(self.physical_regions) != len(
                self.triangles
            ):
                raise ValueError(
                    "Number of physical-region IDs must "
                    "match number of triangles."
                )

    @property
    def node_count(self) -> int:
        """
        Number of mesh nodes.
        """

        return len(self.nodes)

    @property
    def element_count(self) -> int:
        """
        Number of triangular elements.
        """

        return len(self.triangles)

    @property
    def bounding_box(
        self,
    ) -> tuple[float, float, float, float]:
        """
        Return:

            xmin, ymin, xmax, ymax

        in meters.
        """

        if self.node_count == 0:
            raise ValueError(
                "Mesh contains no nodes."
            )

        xmin = np.min(
            self.nodes[:, 0]
        )

        ymin = np.min(
            self.nodes[:, 1]
        )

        xmax = np.max(
            self.nodes[:, 0]
        )

        ymax = np.max(
            self.nodes[:, 1]
        )

        return (
            xmin,
            ymin,
            xmax,
            ymax,
        )

    def triangle_coordinates(
        self,
        index: int,
    ) -> np.ndarray:
        """
        Return the three coordinates of a triangle.

        Returns
        -------
        ndarray
            Shape (3, 2), coordinates in meters.
        """

        node_indices = (
            self.triangles[index]
        )

        return self.nodes[
            node_indices
        ]

    def triangle_areas(self) -> np.ndarray:
        """
        Calculate the area of every triangle.

        Returns
        -------
        ndarray
            Triangle areas in m².
        """

        points = self.nodes[
            self.triangles
        ]

        x1 = points[:, 0, 0]
        y1 = points[:, 0, 1]

        x2 = points[:, 1, 0]
        y2 = points[:, 1, 1]

        x3 = points[:, 2, 0]
        y3 = points[:, 2, 1]

        areas = 0.5 * np.abs(
            (x2 - x1) * (y3 - y1)
            - (x3 - x1) * (y2 - y1)
        )

        return areas

    def total_area(self) -> float:
        """
        Return total mesh area in m².
        """

        return float(
            np.sum(
                self.triangle_areas()
            )
        )

    def validate(self) -> list[str]:
        """
        Validate basic mesh topology and geometry.

        Returns
        -------
        list[str]
            Detected mesh errors.
        """

        errors = []

        if self.node_count == 0:
            errors.append(
                "Mesh contains no nodes."
            )

        if self.element_count == 0:
            errors.append(
                "Mesh contains no elements."
            )

        if self.element_count > 0:

            areas = (
                self.triangle_areas()
            )

            if np.any(areas <= 0):

                bad_count = int(
                    np.sum(
                        areas <= 0
                    )
                )

                errors.append(
                    f"Mesh contains "
                    f"{bad_count} zero-area or "
                    f"negative-area triangles."
                )

        if self.element_count > 0:

            if np.any(
                self.triangles < 0
            ):

                errors.append(
                    "Mesh contains negative node indices."
                )

            if np.any(
                self.triangles
                >= self.node_count
            ):

                errors.append(
                    "Mesh contains node indices "
                    "outside the node array."
                )

        return errors

    def summary(self) -> str:
        """
        Generate a human-readable mesh summary.
        """

        xmin, ymin, xmax, ymax = (
            self.bounding_box
        )

        return (
            "\n"
            "SNSPD FEM MESH SUMMARY\n"
            "======================\n"
            f"Nodes            : {self.node_count}\n"
            f"Triangles        : {self.element_count}\n"
            f"Mesh area        : "
            f"{self.total_area() * 1e12:.6f} um²\n"
            f"Bounding box     :\n"
            f"    xmin = {xmin * 1e6:.6f} um\n"
            f"    ymin = {ymin * 1e6:.6f} um\n"
            f"    xmax = {xmax * 1e6:.6f} um\n"
            f"    ymax = {ymax * 1e6:.6f} um\n"
        )