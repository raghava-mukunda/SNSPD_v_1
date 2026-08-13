# FILE: src/snspd/geometry/analyzer.py
# PURPOSE:
# Calculates purely geometric quantities from DeviceGeometry.
#
# This module does NOT calculate superconducting or electrical properties.
# Those quantities will emerge from later physics solvers.

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snspd.geometry.geometry import DeviceGeometry


@dataclass
class GeometryMetrics:
    """
    Basic geometric measurements.
    """

    width_m: float
    height_m: float
    area_m2: float
    perimeter_m: float

    region_count: int

    min_region_area_m2: float
    max_region_area_m2: float

    aspect_ratio: float


def analyze_geometry(
    geometry: DeviceGeometry,
) -> GeometryMetrics:
    """
    Calculate basic geometric metrics.

    Parameters
    ----------
    geometry:
        Canonical device geometry.

    Returns
    -------
    GeometryMetrics
    """

    if geometry.region_count == 0:

        raise ValueError(
            "Cannot analyze an empty geometry."
        )

    width = geometry.width
    height = geometry.height

    areas = [
        region.area
        for region in geometry.regions
    ]

    perimeter = sum(
        region.perimeter
        for region in geometry.regions
    )

    aspect_ratio = (
        width / height
        if height > 0
        else np.inf
    )

    return GeometryMetrics(
        width_m=width,
        height_m=height,
        area_m2=geometry.total_area,
        perimeter_m=perimeter,
        region_count=geometry.region_count,
        min_region_area_m2=min(areas),
        max_region_area_m2=max(areas),
        aspect_ratio=aspect_ratio,
    )


def format_metrics(
    metrics: GeometryMetrics,
) -> str:
    """
    Convert geometry metrics into a human-readable report.
    """

    return (
        "\n"
        "SNSPD GEOMETRY METRICS\n"
        "======================\n"
        f"Width            : "
        f"{metrics.width_m * 1e6:.6f} um\n"
        f"Height           : "
        f"{metrics.height_m * 1e6:.6f} um\n"
        f"Area             : "
        f"{metrics.area_m2 * 1e12:.6f} um²\n"
        f"Perimeter        : "
        f"{metrics.perimeter_m * 1e6:.6f} um\n"
        f"Regions          : "
        f"{metrics.region_count}\n"
        f"Minimum area     : "
        f"{metrics.min_region_area_m2 * 1e12:.6f} um²\n"
        f"Maximum area     : "
        f"{metrics.max_region_area_m2 * 1e12:.6f} um²\n"
        f"Aspect ratio     : "
        f"{metrics.aspect_ratio:.6f}\n"
    )