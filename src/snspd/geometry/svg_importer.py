# FILE: src/snspd/geometry/svg_importer.py
#
# PURPOSE:
# High-precision SVG geometry importer for the SNSPD digital twin.
#
# Supported geometry:
#
#   <rect>
#   <polygon>
#   <polyline>
#   <path>
#
# SVG paths support:
#
#   M m
#   L l
#   H h
#   V v
#   C c
#   S s
#   Q q
#   T t
#   A a
#   Z z
#
# Filled paths are converted into Shapely polygons.
#
# Centerline paths:
#
#     fill="none"
#     stroke != "none"
#     stroke-width="..."
#
# are interpreted as nanowire centerlines and buffered into
# physical nanowire regions.
#
# All canonical geometry coordinates are stored in meters.


from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from shapely.geometry import (
    LineString,
    Polygon,
    MultiPolygon,
    GeometryCollection,
    Point,
)

from shapely.ops import (
    unary_union,
    polygonize,
)

from svgpathtools import parse_path

from snspd.geometry.geometry import (
    DeviceGeometry,
    GeometryRegion,
)


# ============================================================
# SVG UNIT CONVERSION
# ============================================================


UNIT_TO_M = {
    "m": 1.0,
    "cm": 1e-2,
    "mm": 1e-3,
    "um": 1e-6,
    "µm": 1e-6,
    "nm": 1e-9,
    "in": 0.0254,
    "pt": 0.0254 / 72.0,
    "pc": 0.0254 / 6.0,
    "px": 0.0254 / 96.0,
}


# ============================================================
# LENGTH PARSER
# ============================================================


def parse_length(
    value: str | None,
    default_unit: str = "px",
) -> float:
    """
    Parse an SVG length and convert it to meters.

    Examples
    --------
    80nm -> 80e-9 m
    2um  -> 2e-6 m
    1mm  -> 1e-3 m
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


# ============================================================
# SVG POINT PARSER
# ============================================================


def parse_points(
    points_string: str,
) -> list[tuple[float, float]]:
    """
    Parse an SVG points attribute.

    Example
    -------
    0,0 10,0 10,5 0,5
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


# ============================================================
# SVG PATH SAMPLING
# ============================================================


def _split_path_subpaths(
    path,
):
    """
    Split an svgpathtools Path into continuous subpaths.

    A new subpath is detected whenever the start of a segment
    is discontinuous from the end of the previous segment.
    """

    subpaths = []

    current = []

    previous_end = None

    for segment in path:

        start = complex(segment.start)

        end = complex(segment.end)

        if (
            previous_end is not None
            and abs(start - previous_end) > 1e-12
        ):

            if current:

                subpaths.append(
                    current
                )

            current = []

        current.append(
            segment
        )

        previous_end = end

    if current:

        subpaths.append(
            current
        )

    return subpaths


def _sample_path_subpath(
    segments,
    samples_per_segment: int = 64,
) -> list[tuple[float, float]]:
    """
    Sample one SVG path subpath.

    Curved segments are numerically sampled.
    """

    points = []

    for segment in segments:

        t_values = np.linspace(
            0.0,
            1.0,
            samples_per_segment,
            endpoint=False,
        )

        for t in t_values:

            point = segment.point(
                float(t)
            )

            points.append(
                (
                    float(point.real),
                    float(point.imag),
                )
            )

    if segments:

        final_point = segments[-1].point(
            1.0
        )

        points.append(
            (
                float(final_point.real),
                float(final_point.imag),
            )
        )

    return points


def path_to_rings(
    path_string: str,
    samples_per_segment: int = 64,
) -> list[LineString]:
    """
    Convert an SVG path into closed Shapely rings.

    The function supports paths containing multiple subpaths.

    Each closed subpath becomes a LineString ring.
    """

    path = parse_path(
        path_string
    )

    subpaths = _split_path_subpaths(
        path
    )

    rings = []

    for segments in subpaths:

        points = _sample_path_subpath(
            segments,
            samples_per_segment,
        )

        if len(points) < 3:

            continue

        first = np.asarray(
            points[0]
        )

        last = np.asarray(
            points[-1]
        )

        if np.linalg.norm(
            first - last
        ) > 1e-12:

            # SVG closed geometry should end where it started.
            # For a filled path, close the contour explicitly.
            points.append(
                points[0]
            )

        if len(points) < 4:

            continue

        ring = LineString(
            points
        )

        if ring.length <= 0:

            continue

        rings.append(
            ring
        )

    if not rings:

        raise ValueError(
            "SVG path contains no valid closed geometry."
        )

    return rings


# ============================================================
# PATH -> POLYGON
# ============================================================


def _polygonize_rings(
    rings: list[LineString],
) -> list[Polygon]:
    """
    Convert closed rings into polygon candidates.

    This uses Shapely polygonization rather than simply assuming
    that the first ring is the exterior and all remaining rings
    are holes.

    This is important because exported SVG files can contain:

        exterior
        hole
        hole
        exterior
        ...

    in arbitrary ordering.
    """

    if not rings:

        return []

    network = unary_union(
        rings
    )

    polygons = list(
        polygonize(
            network
        )
    )

    return [
        polygon
        for polygon in polygons
        if polygon.area > 0
    ]


def _evenodd_from_rings(
    rings: list[LineString],
) -> Polygon | MultiPolygon | GeometryCollection:
    """
    Construct filled geometry using SVG even-odd fill semantics.

    A point is inside the filled geometry when it is enclosed by
    an odd number of path rings.

    This matches:

        fill-rule="evenodd"
    """

    candidates = _polygonize_rings(
        rings
    )

    if not candidates:

        return GeometryCollection()

    selected = []

    for candidate in candidates:

        point = candidate.representative_point()

        crossings = 0

        for ring in rings:

            ring_polygon = Polygon(
                ring
            )

            if ring_polygon.contains(
                point
            ):

                crossings += 1

        if crossings % 2 == 1:

            selected.append(
                candidate
            )

    if not selected:

        return GeometryCollection()

    return unary_union(
        selected
    )


def _nonzero_from_rings(
    rings: list[LineString],
) -> Polygon | MultiPolygon | GeometryCollection:
    """
    Construct geometry for SVG's nonzero fill rule.

    For normal exported polygon geometry, a union of the
    polygonized regions is sufficient.

    Explicit holes are subsequently represented naturally
    by the resulting topology.
    """

    candidates = _polygonize_rings(
        rings
    )

    if not candidates:

        return GeometryCollection()

    return unary_union(
        candidates
    )


def path_to_filled_geometry(
    path_string: str,
    fill_rule: str = "nonzero",
    samples_per_segment: int = 64,
):
    """
    Convert a filled SVG path into Shapely geometry.

    Parameters
    ----------
    path_string:
        SVG path 'd' attribute.

    fill_rule:
        SVG fill rule:

            nonzero
            evenodd

    samples_per_segment:
        Number of samples used per curved SVG segment.
    """

    rings = path_to_rings(
        path_string,
        samples_per_segment=samples_per_segment,
    )

    fill_rule = (
        fill_rule
        .strip()
        .lower()
    )

    if fill_rule == "evenodd":

        geometry = _evenodd_from_rings(
            rings
        )

    else:

        geometry = _nonzero_from_rings(
            rings
        )

    if geometry.is_empty:

        raise ValueError(
            "Filled SVG path produced empty geometry."
        )

    if not geometry.is_valid:

        geometry = geometry.buffer(
            0
        )

    return geometry


# ============================================================
# CENTERLINE PATH
# ============================================================


def path_to_linestring(
    path_string: str,
    samples_per_segment: int = 100,
) -> LineString:
    """
    Convert an SVG path into a Shapely LineString.

    Used for SNSPD centerline geometry.
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

            point = segment.point(
                float(t)
            )

            points.append(
                (
                    float(point.real),
                    float(point.imag),
                )
            )

    if len(path) > 0:

        final_point = path[-1].point(
            1.0
        )

        points.append(
            (
                float(final_point.real),
                float(final_point.imag),
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


# ============================================================
# CENTERLINE -> PHYSICAL NANOWIRE
# ============================================================


def centerline_to_wire(
    centerline: LineString,
    width_m: float,
    resolution: int = 32,
) -> Polygon:
    """
    Convert an SNSPD centerline into a physical nanowire.

    Mathematically:

        Wire = Centerline ⊕ Disk(width/2)

    i.e. a Minkowski sum / geometric buffer.
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


# ============================================================
# GEOMETRY HELPERS
# ============================================================


def _geometry_to_regions(
    geometry,
):
    """
    Convert arbitrary Shapely polygonal geometry into individual
    Polygon objects.
    """

    if geometry.is_empty:

        return []

    if geometry.geom_type == "Polygon":

        return [geometry]

    if geometry.geom_type == "MultiPolygon":

        return list(
            geometry.geoms
        )

    if geometry.geom_type == "GeometryCollection":

        regions = []

        for item in geometry.geoms:

            if item.geom_type == "Polygon":

                regions.append(
                    item
                )

            elif item.geom_type == "MultiPolygon":

                regions.extend(
                    list(
                        item.geoms
                    )
                )

        return regions

    raise ValueError(
        "Unsupported Shapely geometry type: "
        f"{geometry.geom_type}"
    )


def _element_style(
    element,
    attribute: str,
    default: str,
) -> str:
    """
    Read an SVG presentation attribute.

    Supports both:

        fill="..."

    and:

        style="fill:..."
    """

    value = element.get(
        attribute
    )

    if value is not None:

        return value.strip()

    style = element.get(
        "style",
        "",
    )

    for item in style.split(";"):

        if ":" not in item:

            continue

        key, val = item.split(
            ":",
            1,
        )

        if key.strip().lower() == attribute:

            return val.strip()

    return default


# ============================================================
# SVG IMPORTER
# ============================================================


def import_svg(
    filename: str | Path,
) -> DeviceGeometry:
    """
    Import an SVG file into DeviceGeometry.

    Supported primitives:

        <rect>
        <polygon>
        <polyline>
        <path>

    For <path>:

        fill="none" + stroke
            -> SNSPD centerline

        filled path
            -> physical polygonal geometry

    Filled SVG paths are fully supported, including paths with
    multiple contours and holes using fill-rule="evenodd".
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

    # ========================================================
    # DEFAULT SVG UNIT
    # ========================================================

    width_attribute = root.get(
        "width"
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

    scale = UNIT_TO_M[
        default_unit
    ]

    # ========================================================
    # PROCESS ELEMENTS
    # ========================================================

    for element in root.iter():

        tag = element.tag

        if "}" in tag:

            tag = tag.split(
                "}",
                1,
            )[1]

        try:

            # ==================================================
            # RECTANGLE
            # ==================================================

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

            # ==================================================
            # POLYGON
            # ==================================================

            elif tag == "polygon":

                points = parse_points(
                    element.get(
                        "points",
                        "",
                    )
                )

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

            # ==================================================
            # POLYLINE
            # ==================================================

            elif tag == "polyline":

                points = parse_points(
                    element.get(
                        "points",
                        "",
                    )
                )

                points = [
                    (
                        x * scale,
                        y * scale,
                    )
                    for x, y in points
                ]

                stroke_width = element.get(
                    "stroke-width"
                )

                stroke = _element_style(
                    element,
                    "stroke",
                    "none",
                ).lower()

                if (
                    stroke_width is not None
                    and stroke != "none"
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

                    geometry.add_region(
                        GeometryRegion(
                            polygon=polygon,
                            name=element.get(
                                "id",
                                f"nanowire_{geometry.region_count}",
                            ),
                            metadata={
                                "source_type":
                                    "nanowire_centerline",
                                "width_m":
                                    width_m,
                                "centerline_length_m":
                                    centerline.length,
                            },
                        )
                    )

                else:

                    if (
                        len(points) >= 3
                        and points[0] != points[-1]
                    ):

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

            # ==================================================
            # PATH
            # ==================================================

            elif tag == "path":

                path_string = element.get(
                    "d",
                    "",
                )

                if not path_string.strip():

                    continue

                fill = _element_style(
                    element,
                    "fill",
                    "black",
                ).lower()

                stroke = _element_style(
                    element,
                    "stroke",
                    "none",
                ).lower()

                stroke_width_attribute = (
                    element.get(
                        "stroke-width"
                    )
                )

                fill_rule = _element_style(
                    element,
                    "fill-rule",
                    "nonzero",
                ).lower()

                # ==================================================
                # CENTERLINE PATH
                # ==================================================

                if (
                    fill == "none"
                    and stroke != "none"
                    and stroke_width_attribute
                    is not None
                ):

                    centerline = path_to_linestring(
                        path_string
                    )

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
                                "source_type":
                                    "nanowire_centerline",
                                "width_m":
                                    width_m,
                                "centerline_length_m":
                                    centerline.length,
                            },
                        )
                    )

                # ==================================================
                # FILLED PATH
                # ==================================================

                else:

                    filled_geometry = (
                        path_to_filled_geometry(
                            path_string,
                            fill_rule=fill_rule,
                            samples_per_segment=64,
                        )
                    )

                    # Convert SVG coordinates into meters.
                    filled_geometry = (
                        _scale_shapely_geometry(
                            filled_geometry,
                            scale,
                        )
                    )

                    regions = (
                        _geometry_to_regions(
                            filled_geometry
                        )
                    )

                    if not regions:

                        raise ValueError(
                            "Filled SVG path produced "
                            "no polygonal regions."
                        )

                    for region_index, polygon in enumerate(
                        regions
                    ):

                        geometry.add_region(
                            GeometryRegion(
                                polygon=polygon,
                                name=element.get(
                                    "id",
                                    f"region_{geometry.region_count}",
                                )
                                if len(regions) == 1
                                else (
                                    element.get(
                                        "id",
                                        "region",
                                    )
                                    + f"_{region_index}"
                                ),
                                metadata={
                                    "source_type":
                                        "filled_svg_path",
                                    "fill_rule":
                                        fill_rule,
                                },
                            )
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


# ============================================================
# SHAPELY SCALING
# ============================================================


def _scale_shapely_geometry(
    geometry,
    scale: float,
):
    """
    Scale a Shapely geometry from SVG coordinate units
    into meters.

    Implemented explicitly instead of relying on an additional
    affine-transformation dependency.
    """

    if geometry.is_empty:

        return geometry

    if geometry.geom_type == "Polygon":

        exterior = [
            (
                x * scale,
                y * scale,
            )
            for x, y
            in geometry.exterior.coords
        ]

        interiors = []

        for interior in geometry.interiors:

            interiors.append(
                [
                    (
                        x * scale,
                        y * scale,
                    )
                    for x, y
                    in interior.coords
                ]
            )

        return Polygon(
            exterior,
            interiors,
        )

    if geometry.geom_type == "MultiPolygon":

        return MultiPolygon(
            [
                _scale_shapely_geometry(
                    polygon,
                    scale,
                )
                for polygon in geometry.geoms
            ]
        )

    if geometry.geom_type == "GeometryCollection":

        scaled = []

        for item in geometry.geoms:

            scaled_item = (
                _scale_shapely_geometry(
                    item,
                    scale,
                )
            )

            if not scaled_item.is_empty:

                scaled.append(
                    scaled_item
                )

        return GeometryCollection(
            scaled
        )

    raise ValueError(
        "Unsupported geometry type for scaling: "
        f"{geometry.geom_type}"
    )