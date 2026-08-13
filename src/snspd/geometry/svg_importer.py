# FILE: src/snspd/geometry/svg_importer.py
# PURPOSE:
# Imports SVG vector geometry and converts it into the canonical
# DeviceGeometry representation used by the SNSPD digital twin.
#
# The importer distinguishes between:
#
#   1. Filled SVG regions
#   2. SVG nanowire centerlines represented using stroke + stroke-width
#
# Nanowire centerlines are converted into physical wire regions using
# geometric buffering:
#
#       centerline -> LineString -> buffer(width / 2) -> Polygon
#
# All canonical geometry coordinates are stored in meters.
#
# IMPORTANT:
# This is still the Stage-1 geometry engine. High-precision handling of
# SVG transforms, exact Bezier geometry, clipping, nested groups, etc.
# will be added before the geometry is considered production-ready.

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from shapely.geometry import (
    LineString,
    Polygon,
)
from shapely.ops import unary_union
from svgpathtools import parse_path

from snspd.geometry.geometry import (
    DeviceGeometry,
    GeometryRegion,
)


# ------------------------------------------------------------
# SVG UNIT CONVERSION
# ------------------------------------------------------------

UNIT_TO_M = {
    "m": 1.0,
    "cm": 1e-2,
    "mm": 1e-3,
    "um": 1e-6,
    "µm": 1e-6,
    "nm": 1e-9,
    "in": 0.0254,
    "pt": 0.0254 / 72.0,
    "px": 0.0254 / 96.0,
}


# ------------------------------------------------------------
# LENGTH PARSER
# ------------------------------------------------------------

def parse_length(
    value: str | None,
    default_unit: str = "px",
) -> float:
    """
    Parse an SVG length and convert it to meters.

    Examples
    --------
    '80nm' -> 80e-9 m
    '2um'  -> 2e-6 m
    '1mm'  -> 1e-3 m
    """

    if value is None:
        raise ValueError(
            "SVG length cannot be None."
        )

    value = value.strip()

    match = re.fullmatch(
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eE][-+]?\d+)?)"
        r"\s*([a-zA-Zµ]*)",
        value,
    )

    if not match:
        raise ValueError(
            f"Invalid SVG length: '{value}'"
        )

    number = float(
        match.group(1)
    )

    unit = (
        match.group(2)
        or default_unit
    )

    if unit not in UNIT_TO_M:
        raise ValueError(
            f"Unsupported SVG unit '{unit}' "
            f"in value '{value}'."
        )

    return number * UNIT_TO_M[unit]


# ------------------------------------------------------------
# SVG POINT PARSER
# ------------------------------------------------------------

def parse_points(
    points_string: str,
) -> list[tuple[float, float]]:
    """
    Parse an SVG points attribute.

    Example
    -------
    '0,0 10,0 10,5 0,5'
    """

    values = re.findall(
        r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
        r"(?:[eE][-+]?\d+)?",
        points_string,
    )

    if len(values) % 2 != 0:
        raise ValueError(
            "Invalid SVG points definition: "
            f"'{points_string}'"
        )

    coordinates = [
        float(value)
        for value in values
    ]

    return [
        (
            coordinates[i],
            coordinates[i + 1],
        )
        for i in range(
            0,
            len(coordinates),
            2,
        )
    ]


# ------------------------------------------------------------
# SVG PATH -> CENTERLINE
# ------------------------------------------------------------

def path_to_linestring(
    path_string: str,
    samples_per_segment: int = 100,
) -> LineString:
    """
    Convert an SVG path into a Shapely LineString.

    The path is treated as a CENTERLINE rather than a filled polygon.

    Curved segments are sampled at this stage. Exact Bezier geometry
    preservation will be introduced in the high-precision geometry engine.
    """

    path = parse_path(
        path_string
    )

    points = []

    for segment in path:

        t_values = np.linspace(
            0.0,
            1.0,
            samples_per_segment,
            endpoint=False,
        )

        for t in t_values:

            point = segment.point(t)

            points.append(
                (
                    point.real,
                    point.imag,
                )
            )

    # Add the final point of the final segment.
    if len(path) > 0:

        final_point = path[-1].point(1.0)

        points.append(
            (
                final_point.real,
                final_point.imag,
            )
        )

    if len(points) < 2:
        raise ValueError(
            "SVG path does not contain enough points "
            "to construct a centerline."
        )

    return LineString(
        points
    )


# ------------------------------------------------------------
# CENTERLINE -> PHYSICAL NANOWIRE
# ------------------------------------------------------------

def centerline_to_wire(
    centerline: LineString,
    width_m: float,
    resolution: int = 32,
) -> Polygon:
    """
    Convert an SNSPD nanowire centerline into a physical wire region.

    Parameters
    ----------
    centerline:
        Nanowire centerline in meters.

    width_m:
        Physical nanowire width in meters.

    resolution:
        Number of segments used to approximate round buffer joins/caps.

    Returns
    -------
    Polygon
        Physical nanowire region.

    Mathematical operation
    ----------------------
    The wire is represented as the Minkowski sum:

        wire = centerline ⊕ disk(width / 2)

    which is equivalent to a geometric buffer of width / 2.
    """

    if width_m <= 0:
        raise ValueError(
            "Nanowire width must be positive."
        )

    if centerline.is_empty:
        raise ValueError(
            "Nanowire centerline is empty."
        )

    if centerline.length <= 0:
        raise ValueError(
            "Nanowire centerline has zero length."
        )

    wire = centerline.buffer(
        width_m / 2.0,
        resolution=resolution,
        cap_style=2,
        join_style=2,
    )

    if wire.is_empty:
        raise ValueError(
            "Buffering the nanowire centerline "
            "produced an empty region."
        )

    # Buffer can theoretically produce MultiPolygon geometry
    # if the centerline topology is problematic.
    #
    # For a connected SNSPD nanowire we require one physical region.
    if wire.geom_type == "MultiPolygon":

        wire = unary_union(
            wire
        )

    if wire.geom_type != "Polygon":

        raise ValueError(
            "Nanowire centerline did not produce "
            "a single connected Polygon. "
            f"Result type: {wire.geom_type}"
        )

    return wire


# ------------------------------------------------------------
# SVG IMPORTER
# ------------------------------------------------------------

def import_svg(
    filename: str | Path,
) -> DeviceGeometry:
    """
    Import an SVG file into DeviceGeometry.

    Supported initial primitives:

        <rect>
        <polygon>
        <polyline>
        <path>

    For <path>:

        fill != none
            -> interpreted as filled geometry

        fill="none" + stroke
            -> interpreted as a centerline

    For an SNSPD, the second form is preferred because the physical
    nanowire width is explicitly represented by stroke-width.
    """

    filename = Path(
        filename
    )

    if not filename.exists():

        raise FileNotFoundError(
            f"SVG file not found: {filename}"
        )

    tree = ET.parse(
        filename
    )

    root = tree.getroot()

    geometry = DeviceGeometry(
        source_format="SVG",
        source_file=str(filename),
    )

    # --------------------------------------------------------
    # Determine default SVG unit.
    # --------------------------------------------------------

    width_attribute = (
        root.get("width")
    )

    if width_attribute:

        width_match = re.search(
            r"[a-zA-Zµ]+",
            width_attribute,
        )

        default_unit = (
            width_match.group(0)
            if width_match
            else "px"
        )

    else:

        default_unit = "px"

    # --------------------------------------------------------
    # Process SVG elements.
    # --------------------------------------------------------

    for element in root.iter():

        tag = element.tag

        # Remove XML namespace.
        if "}" in tag:

            tag = tag.split(
                "}",
                1,
            )[1]

        try:

            # =================================================
            # RECTANGLE
            # =================================================

            if tag == "rect":

                x = parse_length(
                    element.get(
                        "x",
                        "0",
                    ),
                    default_unit,
                )

                y = parse_length(
                    element.get(
                        "y",
                        "0",
                    ),
                    default_unit,
                )

                width = parse_length(
                    element.get(
                        "width"
                    ),
                    default_unit,
                )

                height = parse_length(
                    element.get(
                        "height"
                    ),
                    default_unit,
                )

                polygon = Polygon(
                    [
                        (x, y),
                        (x + width, y),
                        (
                            x + width,
                            y + height,
                        ),
                        (
                            x,
                            y + height,
                        ),
                    ]
                )

                geometry.add_region(
                    GeometryRegion(
                        polygon=polygon,
                        name=element.get(
                            "id",
                            f"region_{geometry.region_count}",
                        ),
                    )
                )

            # =================================================
            # POLYGON
            # =================================================

            elif tag == "polygon":

                points = parse_points(
                    element.get(
                        "points",
                        "",
                    )
                )

                scale = UNIT_TO_M[
                    default_unit
                ]

                points = [
                    (
                        x * scale,
                        y * scale,
                    )
                    for x, y in points
                ]

                polygon = Polygon(
                    points
                )

                geometry.add_region(
                    GeometryRegion(
                        polygon=polygon,
                        name=element.get(
                            "id",
                            f"region_{geometry.region_count}",
                        ),
                    )
                )

            # =================================================
            # POLYLINE
            # =================================================

            elif tag == "polyline":

                points = parse_points(
                    element.get(
                        "points",
                        "",
                    )
                )

                scale = UNIT_TO_M[
                    default_unit
                ]

                points = [
                    (
                        x * scale,
                        y * scale,
                    )
                    for x, y in points
                ]

                # Polyline with a stroke is treated as a centerline.
                stroke_width = element.get(
                    "stroke-width"
                )

                if (
                    stroke_width is not None
                    and element.get(
                        "stroke",
                        "none",
                    ).lower()
                    != "none"
                ):

                    width_m = parse_length(
                        stroke_width,
                        default_unit,
                    )

                    centerline = LineString(
                        points
                    )

                    polygon = centerline_to_wire(
                        centerline,
                        width_m,
                    )

                else:

                    # A closed polyline may represent a filled region.
                    if points[0] != points[-1]:
                        points.append(
                            points[0]
                        )

                    polygon = Polygon(
                        points
                    )

                geometry.add_region(
                    GeometryRegion(
                        polygon=polygon,
                        name=element.get(
                            "id",
                            f"region_{geometry.region_count}",
                        ),
                    )
                )

            # =================================================
            # PATH
            # =================================================

            elif tag == "path":

                path_string = element.get(
                    "d",
                    "",
                )

                if not path_string.strip():
                    continue

                fill = element.get(
                    "fill",
                    "black",
                ).lower()

                stroke = element.get(
                    "stroke",
                    "none",
                ).lower()

                stroke_width_attribute = (
                    element.get(
                        "stroke-width"
                    )
                )

                # ------------------------------------------------
                # SNSPD CENTERLINE
                # ------------------------------------------------

                if (
                    fill == "none"
                    and stroke != "none"
                    and stroke_width_attribute is not None
                ):

                    centerline = path_to_linestring(
                        path_string
                    )

                    # SVG numerical coordinates are converted
                    # using the document's default physical unit.
                    scale = UNIT_TO_M[
                        default_unit
                    ]

                    centerline = LineString(
                        [
                            (
                                x * scale,
                                y * scale,
                            )
                            for x, y
                            in centerline.coords
                        ]
                    )

                    width_m = parse_length(
                        stroke_width_attribute,
                        default_unit,
                    )

                    polygon = centerline_to_wire(
                        centerline,
                        width_m,
                    )

                    geometry.add_region(
                        GeometryRegion(
                            polygon=polygon,
                            name=element.get(
                                "id",
                                f"nanowire_{geometry.region_count}",
                            ),
                            metadata={
                                "source_type": "nanowire_centerline",
                                "width_m": width_m,
                                "centerline_length_m":
                                    centerline.length,
                            },
                        )
                    )

                # ------------------------------------------------
                # FILLED PATH
                # ------------------------------------------------

                else:

                    raise NotImplementedError(
                        "Filled SVG <path> geometry is not yet "
                        "implemented in the high-precision importer. "
                        "Use polygon geometry or a nanowire "
                        "centerline with fill='none' and stroke-width."
                    )

        except (
            ValueError,
            KeyError,
        ) as exc:

            raise ValueError(
                f"Failed to parse SVG "
                f"element '{tag}': {exc}"
            ) from exc

    return geometry