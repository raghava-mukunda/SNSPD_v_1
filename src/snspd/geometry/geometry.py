# FILE: src/snsdp/geometry/geometry.py
# PURPOSE:
# Defines the canonical internal geometry representation used by the
# SNSPD digital twin.
#
# SVG, GDSII, DXF, and future geometry formats will all be converted
# into this representation before entering the meshing/physics pipeline.
#
# The physics solvers must not depend directly on any input-file format.

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from shapely.geometry import Polygon
from shapely.validation import explain_validity


@dataclass
class GeometryRegion:
    """
    Represents one physical 2D region of the device.

    Coordinates are stored internally in meters.
    """

    polygon: Polygon

    name: str = "region"

    material: str | None = None

    layer: int | None = None

    metadata: dict = field(default_factory=dict)

    @property
    def area(self) -> float:
        """
        Return region area in m².
        """
        return self.polygon.area

    @property
    def perimeter(self) -> float:
        """
        Return region perimeter in meters.
        """
        return self.polygon.length


@dataclass
class DeviceGeometry:
    """
    Canonical representation of the complete SNSPD geometry.

    All coordinates are stored in meters.
    """

    regions: List[GeometryRegion] = field(default_factory=list)

    source_format: str | None = None

    source_file: str | None = None

    metadata: dict = field(default_factory=dict)

    def add_region(self, region: GeometryRegion) -> None:
        """
        Add a geometry region to the device.
        """
        self.regions.append(region)

    @property
    def region_count(self) -> int:
        """
        Return the number of geometry regions.
        """
        return len(self.regions)

    @property
    def total_area(self) -> float:
        """
        Return the total area of all regions in m².

        Regions are assumed to be non-overlapping.
        Overlap detection is handled by validate().
        """
        return sum(
            region.area
            for region in self.regions
        )

    @property
    def bounding_box(self) -> tuple[float, float, float, float]:
        """
        Return the global geometry bounding box.

        Returns
        -------
        tuple
            (xmin, ymin, xmax, ymax) in meters.
        """

        if not self.regions:
            raise ValueError(
                "Geometry contains no regions."
            )

        bounds = [
            region.polygon.bounds
            for region in self.regions
        ]

        xmin = min(
            bound[0]
            for bound in bounds
        )

        ymin = min(
            bound[1]
            for bound in bounds
        )

        xmax = max(
            bound[2]
            for bound in bounds
        )

        ymax = max(
            bound[3]
            for bound in bounds
        )

        return xmin, ymin, xmax, ymax

    @property
    def width(self) -> float:
        """
        Return the overall bounding-box width in meters.
        """

        xmin, _, xmax, _ = self.bounding_box

        return xmax - xmin

    @property
    def height(self) -> float:
        """
        Return the overall bounding-box height in meters.
        """

        _, ymin, _, ymax = self.bounding_box

        return ymax - ymin

    def validate(self) -> list[str]:
        """
        Perform basic geometry and topology validation.

        Returns
        -------
        list[str]
            List of detected problems.

        An empty list means no problems were found.
        """

        errors = []

        if not self.regions:
            errors.append(
                "Geometry contains no regions."
            )

            return errors

        # Validate individual regions.
        for index, region in enumerate(self.regions):

            if region.polygon.is_empty:

                errors.append(
                    f"Region {index} "
                    f"('{region.name}') is empty."
                )

            if not region.polygon.is_valid:

                errors.append(
                    f"Region {index} "
                    f"('{region.name}') is geometrically invalid: "
                    f"{explain_validity(region.polygon)}"
                )

            if region.area <= 0:

                errors.append(
                    f"Region {index} "
                    f"('{region.name}') has zero or negative area."
                )

        # Check pairwise overlaps.
        for i in range(len(self.regions)):

            for j in range(i + 1, len(self.regions)):

                intersection = (
                    self.regions[i].polygon.intersection(
                        self.regions[j].polygon
                    )
                )

                if (
                    not intersection.is_empty
                    and intersection.area > 0
                ):

                    errors.append(
                        f"Regions {i} and {j} overlap."
                    )

        return errors

    def summary(self) -> str:
        """
        Generate a human-readable geometry summary.
        """

        xmin, ymin, xmax, ymax = (
            self.bounding_box
        )

        return (
            "\n"
            "SNSPD GEOMETRY SUMMARY\n"
            "======================\n"
            f"Source format : {self.source_format}\n"
            f"Source file   : {self.source_file}\n"
            f"Regions       : {self.region_count}\n"
            f"Width         : "
            f"{self.width * 1e6:.6f} um\n"
            f"Height        : "
            f"{self.height * 1e6:.6f} um\n"
            f"Area          : "
            f"{self.total_area * 1e12:.6f} um²\n"
            f"Bounding box  :\n"
            f"    xmin = {xmin * 1e6:.6f} um\n"
            f"    ymin = {ymin * 1e6:.6f} um\n"
            f"    xmax = {xmax * 1e6:.6f} um\n"
            f"    ymax = {ymax * 1e6:.6f} um\n"
        )