# FILE:
# src/snspd/geometry/nanowire_svg.py
#
# PURPOSE:
# Extract the physical nanowire geometry from a vector SVG.
#
# Pipeline:
#
#     SVG
#       |
#       v
#     SVG path objects
#       |
#       v
#     nanowire path selection
#       |
#       v
#     SVG path -> Shapely polygon
#       |
#       v
#     apply SVG translation
#       |
#       v
#     union connected regions
#       |
#       v
#     remove small artifacts
#       |
#       v
#     scale to SI units
#       |
#       v
#     DeviceGeometry
#
# IMPORTANT:
#
# The SVG coordinate system is NOT assumed to be SI.
# The caller must explicitly provide:
#
#     meters_per_svg_unit
#
# This prevents accidental physical-scale errors.


from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

from shapely.geometry import (
    Polygon,
    MultiPolygon,
    GeometryCollection,
)

from shapely.ops import unary_union

from svgpathtools import (
    parse_path,
)


from snspd.geometry.geometry import (
    DeviceGeometry,
    GeometryRegion,
)


# ============================================================
# CONFIGURATION
# ============================================================


@dataclass
class SVGExtractionConfig:
    """
    Configuration for SVG nanowire extraction.
    """

    # Physical scaling.
    #
    # Example:
    #
    # 1 SVG unit = 1 um
    #
    #     meters_per_svg_unit = 1e-6
    #
    meters_per_svg_unit: float

    # Minimum path area before scaling.
    # Used to reject tiny vector artifacts.
    minimum_area_svg_units2: float = 1.0

    # Number of samples per SVG path segment.
    samples_per_segment: int = 8

    # Select paths based on fill color.
    #
    # The uploaded SVG uses very pale-to-medium red/pink
    # fills for the repeated structure.
    #
    # Red-channel threshold:
    minimum_red: int = 220

    # Green/blue upper bound.
    maximum_green: int = 240
    maximum_blue: int = 240

    # Exclude nearly-white background.
    background_distance_threshold: int = 8

    # Remove disconnected components below this physical area.
    minimum_component_area_m2: float = 0.0


# ============================================================
# COLOR PARSING
# ============================================================


def _parse_hex_color(
    color: str | None,
) -> tuple[int, int, int] | None:
    """
    Parse a hexadecimal SVG color.

    Supports:

        #RRGGBB
        #RGB
    """

    if color is None:

        return None

    color = color.strip()

    if not color.startswith("#"):

        return None

    value = color[1:]

    if len(value) == 3:

        value = "".join(
            character * 2
            for character in value
        )

    if len(value) != 6:

        return None

    try:

        return (
            int(value[0:2], 16),
            int(value[2:4], 16),
            int(value[4:6], 16),
        )

    except ValueError:

        return None


def _get_fill_color(
    element: ET.Element,
) -> tuple[int, int, int] | None:
    """
    Obtain fill color from either:

        fill="..."

    or:

        style="...;fill:#..."
    """

    fill = element.attrib.get(
        "fill"
    )

    if fill:

        return _parse_hex_color(
            fill
        )

    style = element.attrib.get(
        "style",
        "",
    )

    match = re.search(
        r"fill\s*:\s*(#[0-9A-Fa-f]{3,6})",
        style,
    )

    if match:

        return _parse_hex_color(
            match.group(1)
        )

    return None


# ============================================================
# NANOWIRE COLOR SELECTION
# ============================================================


def _is_nanowire_color(
    color: tuple[int, int, int] | None,
    config: SVGExtractionConfig,
) -> bool:
    """
    Determine whether a fill color belongs to the
    nanowire structure.

    The uploaded SVG uses pink/red shades for the
    repeated geometry.
    """

    if color is None:

        return False

    red, green, blue = color

    # Require a sufficiently strong red component.
    if red < config.minimum_red:

        return False

    # Reject colors that are too close to white.
    distance_from_white = (
        (255 - red)
        + (255 - green)
        + (255 - blue)
    )

    if (
        distance_from_white
        < config.background_distance_threshold
    ):

        return False

    # Reject strongly non-red colors.
    if green > config.maximum_green:

        return False

    if blue > config.maximum_blue:

        return False

    return True


# ============================================================
# SVG TRANSFORM
# ============================================================


def _parse_translate(
    transform: str | None,
) -> tuple[float, float]:
    """
    Extract a simple SVG translate(x,y).

    The supplied SVG uses translation transforms for
    individual path objects.

    Rotation, scaling and matrix transforms are deliberately
    rejected rather than silently mishandled.
    """

    if not transform:

        return (
            0.0,
            0.0,
        )

    transform = transform.strip()

    match = re.fullmatch(
        r"translate\(\s*"
        r"([-+0-9.eE]+)"
        r"(?:\s*[, ]\s*"
        r"([-+0-9.eE]+))?"
        r"\s*\)",
        transform,
    )

    if match is None:

        raise ValueError(
            "Unsupported SVG transform: "
            f"{transform!r}. "
            "Only translate(x,y) is currently supported."
        )

    tx = float(
        match.group(1)
    )

    ty = float(
        match.group(2)
        if match.group(2)
        else 0.0
    )

    return (
        tx,
        ty,
    )


# ============================================================
# PATH -> POLYGON
# ============================================================


def _sample_svg_path(
    path_data: str,
    samples_per_segment: int,
) -> np.ndarray:
    """
    Convert an SVG path into sampled XY coordinates.

    Curved segments are sampled parametrically.
    """

    path = parse_path(
        path_data
    )

    points = []

    for segment in path:

        count = max(
            2,
            samples_per_segment,
        )

        parameters = np.linspace(
            0.0,
            1.0,
            count,
        )

        for parameter in parameters:

            value = segment.point(
                parameter
            )

            points.append(
                (
                    float(value.real),
                    float(value.imag),
                )
            )

    if len(points) < 3:

        raise ValueError(
            "SVG path contains fewer than "
            "three usable points."
        )

    # Remove consecutive duplicates.
    cleaned = [
        points[0]
    ]

    for point in points[1:]:

        if (
            not np.allclose(
                point,
                cleaned[-1],
            )
        ):

            cleaned.append(
                point
            )

    return np.asarray(
        cleaned,
        dtype=float,
    )


# ============================================================
# PATH -> SHAPELY
# ============================================================


def _path_to_polygon(
    path_data: str,
    transform: str | None,
    samples_per_segment: int,
) -> Polygon:
    """
    Convert one SVG closed path into a Shapely polygon.
    """

    points = _sample_svg_path(
        path_data,
        samples_per_segment,
    )

    tx, ty = _parse_translate(
        transform
    )

    points[:, 0] += tx
    points[:, 1] += ty

    # Close the polygon.
    if not np.allclose(
        points[0],
        points[-1],
    ):

        points = np.vstack(
            [
                points,
                points[0],
            ]
        )

    polygon = Polygon(
        points
    )

    if polygon.is_empty:

        raise ValueError(
            "SVG path produced an empty polygon."
        )

    # Repair minor numerical/topological defects.
    if not polygon.is_valid:

        polygon = polygon.buffer(
            0.0
        )

    return polygon


# ============================================================
# GEOMETRY EXTRACTION
# ============================================================


def extract_nanowire_geometry(
    svg_file: str | Path,
    config: SVGExtractionConfig,
) -> DeviceGeometry:
    """
    Extract nanowire geometry from an SVG.

    Parameters
    ----------
    svg_file:
        Path to SVG file.

    config:
        Extraction configuration.

    Returns
    -------
    DeviceGeometry
        Canonical SI-unit geometry.
    """

    svg_file = Path(
        svg_file
    )

    if not svg_file.exists():

        raise FileNotFoundError(
            f"SVG file not found: {svg_file}"
        )

    if (
        config.meters_per_svg_unit
        <= 0
    ):

        raise ValueError(
            "meters_per_svg_unit must be positive."
        )

    # --------------------------------------------------------
    # Parse XML
    # --------------------------------------------------------

    tree = ET.parse(
        svg_file
    )

    root = tree.getroot()

    namespace = (
        "{http://www.w3.org/2000/svg}"
    )

    path_elements = root.findall(
        ".//"
        + namespace
        + "path"
    )

    if not path_elements:

        raise RuntimeError(
            "SVG contains no <path> elements."
        )

    # --------------------------------------------------------
    # Extract candidate paths
    # --------------------------------------------------------

    polygons = []

    total_paths = len(
        path_elements
    )

    selected_paths = 0

    rejected_paths = 0

    for element in path_elements:

        path_data = element.attrib.get(
            "d"
        )

        if not path_data:

            continue

        color = _get_fill_color(
            element
        )

        if not _is_nanowire_color(
            color,
            config,
        ):

            rejected_paths += 1

            continue

        try:

            polygon = _path_to_polygon(
                path_data,
                element.attrib.get(
                    "transform"
                ),
                config.samples_per_segment,
            )

        except Exception:

            # Invalid decorative/vector path.
            rejected_paths += 1

            continue

        if (
            polygon.area
            < config.minimum_area_svg_units2
        ):

            rejected_paths += 1

            continue

        polygons.append(
            polygon
        )

        selected_paths += 1

    if not polygons:

        raise RuntimeError(
            "No nanowire paths were extracted."
        )

    # --------------------------------------------------------
    # Union
    # --------------------------------------------------------

    print(
        "\nSVG NANOWIRE EXTRACTION"
    )

    print(
        "-----------------------"
    )

    print(
        f"Total SVG paths       : "
        f"{total_paths}"
    )

    print(
        f"Selected nanowire paths: "
        f"{selected_paths}"
    )

    print(
        f"Rejected paths         : "
        f"{rejected_paths}"
    )

    print(
        "\nUnioning nanowire paths..."
    )

    unioned = unary_union(
        polygons
    )

    if unioned.is_empty:

        raise RuntimeError(
            "Nanowire union produced empty geometry."
        )

    # --------------------------------------------------------
    # Normalize geometry type
    # --------------------------------------------------------

    if isinstance(
        unioned,
        Polygon,
    ):

        components = [
            unioned
        ]

    elif isinstance(
        unioned,
        MultiPolygon,
    ):

        components = list(
            unioned.geoms
        )

    elif isinstance(
        unioned,
        GeometryCollection,
    ):

        components = [
            geometry
            for geometry in unioned.geoms
            if isinstance(
                geometry,
                Polygon,
            )
        ]

    else:

        raise RuntimeError(
            "Unsupported union geometry type: "
            f"{type(unioned)}"
        )

    # --------------------------------------------------------
    # Physical scaling
    # --------------------------------------------------------

    scale = (
        config.meters_per_svg_unit
    )

    scaled_polygons = []

    for polygon in components:

        coordinates = np.asarray(
            polygon.exterior.coords,
            dtype=float,
        )

        coordinates *= scale

        scaled = Polygon(
            coordinates
        )

        if (
            not scaled.is_empty
            and scaled.area > 0
        ):

            scaled_polygons.append(
                scaled
            )

    if not scaled_polygons:

        raise RuntimeError(
            "No valid physical nanowire regions remain."
        )

    # --------------------------------------------------------
    # Remove tiny physical components
    # --------------------------------------------------------

    if (
        config.minimum_component_area_m2
        > 0
    ):

        scaled_polygons = [
            polygon
            for polygon in scaled_polygons
            if polygon.area
            >= config.minimum_component_area_m2
        ]

    if not scaled_polygons:

        raise RuntimeError(
            "All extracted components were removed "
            "by the minimum-area filter."
        )

    # --------------------------------------------------------
    # Build canonical geometry
    # --------------------------------------------------------

    geometry = DeviceGeometry(
        source_format="SVG",
        source_file=str(
            svg_file.resolve()
        ),
        metadata={
            "svg_unit_scale_m": scale,
            "total_svg_paths": total_paths,
            "selected_nanowire_paths": selected_paths,
            "rejected_paths": rejected_paths,
        },
    )

    for index, polygon in enumerate(
        scaled_polygons
    ):

        geometry.add_region(
            GeometryRegion(
                polygon=polygon,
                name=f"nanowire_{index}",
                material="superconductor",
                metadata={
                    "source": "svg",
                },
            )
        )

    return geometry