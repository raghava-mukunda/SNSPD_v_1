#!/usr/bin/env python3

"""
SNSPD SVG PREPROCESSOR
======================

Converts complex/illustrative SVG files into a clean polygon-based
SNSPD geometry compatible with the high-precision SVG importer.

Pipeline:

    SVG
      |
      v
    CairoSVG rasterization
      |
      v
    RGB mask extraction
      |
      v
    red/pink geometry detection
      |
      v
    connected-component filtering
      |
      v
    single-conductor enforcement
      |
      v
    morphological cleanup
      |
      v
    contour extraction
      |
      v
    polygon SVG
      |
      v
    FEM-compatible geometry

The default assumption is that the SNSPD nanowire is represented by
red/pink regions on a white/light background.

Physical scaling is controlled explicitly.

Example:

    python3 examples/geometry/preprocess_svg.py \
        examples/new_geometry.svg \
        -o examples/new_geometry_processed.svg

For a known physical width:

    python3 examples/geometry/preprocess_svg.py \
        examples/new_geometry.svg \
        -o examples/new_geometry_processed.svg \
        --width-um 50

For a known nanowire width:

    python3 examples/geometry/preprocess_svg.py \
        examples/new_geometry.svg \
        -o examples/new_geometry_processed.svg \
        --wire-width-nm 10

"""

from __future__ import annotations

import argparse
import os
import sys
import math

import numpy as np
from PIL import Image

try:
    import cairosvg
except ImportError:
    print("\nERROR: cairosvg is not installed.")
    print("Install with:")
    print("    pip install cairosvg")
    sys.exit(1)

try:
    from scipy import ndimage
except ImportError:
    print("\nERROR: scipy is not installed.")
    print("Install with:")
    print("    pip install scipy")
    sys.exit(1)

try:
    from skimage import measure
except ImportError:
    print("\nERROR: scikit-image is not installed.")
    print("Install with:")
    print("    pip install scikit-image")
    sys.exit(1)


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_RENDER_SCALE = 3.0

# Minimum connected component area in rendered pixels.
DEFAULT_MIN_COMPONENT_AREA = 100

# Polygon simplification tolerance in pixels.
DEFAULT_SIMPLIFY = 1.5

# Default physical width of the entire geometry.
DEFAULT_WIDTH_UM = 50.0


# ============================================================
# SVG RENDERING
# ============================================================

def render_svg(svg_path: str, scale: float) -> Image.Image:

    print("\nRendering SVG...")

    png_bytes = cairosvg.svg2png(
        url=svg_path,
        scale=scale,
        output_width=None,
        output_height=None,
        background_color="white",
    )

    from io import BytesIO

    image = Image.open(BytesIO(png_bytes)).convert("RGB")

    print(
        f"Rendered image : "
        f"{image.width} x {image.height} pixels"
    )

    return image


# ============================================================
# RED/PINK MASK
# ============================================================

# ============================================================
# BLUE MASK
# ============================================================

def create_red_mask(
    image: Image.Image,
    blue_threshold: int = 100,
    color_difference: int = 30,
) -> np.ndarray:

    """
    Detect blue/cyan geometry.

    Condition:

        B > threshold
        B - R > difference
        B - G > difference

    This rejects white/light-gray background.
    """

    rgb = np.asarray(image).astype(np.int16)

    r = rgb[:, :, 0]
    g = rgb[:, :, 1]
    b = rgb[:, :, 2]

    mask = (
        (b >= blue_threshold)
        &
        ((b - r) >= color_difference)
        &
        ((b - g) >= color_difference)
    )

    return mask

# ============================================================
# MASK CLEANUP
# ============================================================

def clean_mask(
    mask: np.ndarray,
    min_component_area: int,
    min_component_fraction: float = 0.01,
) -> np.ndarray:

    print("\nCleaning geometry mask...")

    # Remove isolated noise
    mask = ndimage.binary_opening(
        mask,
        structure=np.ones((3, 3), dtype=bool),
    )

    # Close small gaps
    mask = ndimage.binary_closing(
        mask,
        structure=np.ones((5, 5), dtype=bool),
    )

    # Fill holes inside wire regions
    mask = ndimage.binary_fill_holes(mask)

    # Connected components
    labels, num = ndimage.label(mask)

    if num == 0:
        raise RuntimeError(
            "No red/pink geometry was detected in the SVG."
        )

    components = []

    for label in range(1, num + 1):

        area = np.count_nonzero(labels == label)

        if area >= min_component_area:
            components.append((label, area))

    if not components:
        raise RuntimeError(
            "Red/pink pixels were found, but no sufficiently large "
            "connected component survived filtering."
        )

    components.sort(
        key=lambda x: x[1],
        reverse=True,
    )

    print(f"Connected components found : {num}")
    print(
        f"Components retained        : "
        f"{len(components)}"
    )

    # --------------------------------------------------------
    # ELECTRICAL CONNECTIVITY CHECK
    # --------------------------------------------------------
    #
    # For the electrical FEM model the SNSPD conductor must be
    # ONE connected conducting domain.
    #
    # Rasterization can leave tiny disconnected red/pink
    # fragments near the electrical lead. Keeping those fragments
    # creates multiple conducting regions and can make the FEM
    # matrix singular.
    #
    # Therefore:
    #   1. Keep the largest connected component.
    #   2. Discard only components smaller than the configured
    #      fraction of the largest component.
    #   3. Refuse to silently discard a second large component.
    #
    # This is deliberately conservative: we do not merge distant
    # pieces because that would invent physical conductor geometry.
    # --------------------------------------------------------

    largest_label, largest_area = components[0]

    if not (0.0 < min_component_fraction <= 1.0):
        raise ValueError(
            "min_component_fraction must be between 0 and 1."
        )

    min_relative_area = min_component_fraction

    large_components = [
        (label, area)
        for label, area in components
        if area >= min_relative_area * largest_area
    ]

    if len(large_components) > 1:
        details = ", ".join(
            f"{area}px² ({100.0 * area / largest_area:.2f}% of largest)"
            for _, area in large_components
        )
        raise RuntimeError(
            "\nMultiple substantial disconnected SNSPD conductor "
            "components were detected.\n"
            f"Largest component : {largest_area}px²\n"
            f"Other substantial components: {details}\n"
            "The geometry must be physically connected before FEM "
            "electrical analysis."
        )

    discarded = [
        (label, area)
        for label, area in components
        if label != largest_label
    ]

    if discarded:
        print(
            "\nDiscarding small disconnected raster artifacts:"
        )
        for _, area in discarded:
            print(
                f"    {area}px² "
                f"({100.0 * area / largest_area:.4f}% of largest)"
            )

    cleaned = labels == largest_label

    # Final connectivity check.
    _, final_num = ndimage.label(cleaned)
    if final_num != 1:
        raise RuntimeError(
            f"Internal error: cleaned SNSPD geometry has "
            f"{final_num} connected components."
        )

    print(
        "Final electrical conductor components : 1"
    )

    return cleaned


# ============================================================
# BOUNDING BOX
# ============================================================

def get_mask_bbox(mask: np.ndarray):

    ys, xs = np.where(mask)

    if len(xs) == 0:
        raise RuntimeError("Geometry mask is empty.")

    xmin = int(xs.min())
    xmax = int(xs.max())
    ymin = int(ys.min())
    ymax = int(ys.max())

    return xmin, ymin, xmax, ymax


# ============================================================
# CONTOUR EXTRACTION
# ============================================================

def extract_contours(
    mask: np.ndarray,
    simplify_tolerance: float,
):

    print("\nExtracting geometry contours...")

    contours = measure.find_contours(
        mask.astype(float),
        level=0.5,
    )

    if not contours:
        raise RuntimeError(
            "No geometry contour could be extracted."
        )

    polygons = []

    for contour in contours:

        if len(contour) < 10:
            continue

        # skimage returns:
        #
        #   contour[:, 0] = y
        #   contour[:, 1] = x

        xy = np.column_stack(
            (
                contour[:, 1],
                contour[:, 0],
            )
        )

        # ----------------------------------------------------
        # Remove very small contours
        # ----------------------------------------------------

        area = polygon_area(xy)

        if abs(area) < 20:
            continue

        # ----------------------------------------------------
        # Simplify polygon
        # ----------------------------------------------------

        xy = simplify_closed_polygon(
            xy,
            simplify_tolerance,
        )

        if len(xy) >= 3:
            polygons.append(xy)

    if not polygons:
        raise RuntimeError(
            "No valid polygon geometry was extracted."
        )

    print(
        f"Contours extracted : {len(polygons)}"
    )

    return polygons


# ============================================================
# POLYGON AREA
# ============================================================

def polygon_area(points):

    x = points[:, 0]
    y = points[:, 1]

    return 0.5 * np.sum(
        x * np.roll(y, -1)
        -
        y * np.roll(x, -1)
    )


# ============================================================
# RDP SIMPLIFICATION
# ============================================================

def perpendicular_distance(
    point,
    line_start,
    line_end,
):

    x, y = point
    x1, y1 = line_start
    x2, y2 = line_end

    dx = x2 - x1
    dy = y2 - y1

    if dx == 0 and dy == 0:
        return math.hypot(
            x - x1,
            y - y1,
        )

    numerator = abs(
        dy * x
        -
        dx * y
        +
        x2 * y1
        -
        y2 * x1
    )

    denominator = math.sqrt(
        dx * dx + dy * dy
    )

    return numerator / denominator


def rdp(points, epsilon):

    if len(points) < 3:
        return points

    start = points[0]
    end = points[-1]

    distances = np.array(
        [
            perpendicular_distance(
                p,
                start,
                end,
            )
            for p in points[1:-1]
        ]
    )

    if len(distances) == 0:
        return np.array([start, end])

    max_index = int(np.argmax(distances))
    max_distance = distances[max_index]

    if max_distance > epsilon:

        split_index = max_index + 1

        left = rdp(
            points[: split_index + 1],
            epsilon,
        )

        right = rdp(
            points[split_index:],
            epsilon,
        )

        return np.vstack(
            (
                left[:-1],
                right,
            )
        )

    return np.array(
        [
            start,
            end,
        ]
    )


def simplify_closed_polygon(
    points,
    epsilon,
):

    # Ensure closed contour
    if not np.allclose(
        points[0],
        points[-1],
    ):
        points = np.vstack(
            (
                points,
                points[0],
            )
        )

    simplified = rdp(
        points,
        epsilon,
    )

    if len(simplified) < 4:
        return points

    return simplified[:-1]


# ============================================================
# PHYSICAL SCALING
# ============================================================

def scale_polygons(
    polygons,
    target_width_um,
):

    all_points = np.vstack(polygons)

    xmin = np.min(all_points[:, 0])
    xmax = np.max(all_points[:, 0])

    ymin = np.min(all_points[:, 1])
    ymax = np.max(all_points[:, 1])

    pixel_width = xmax - xmin
    pixel_height = ymax - ymin

    if pixel_width <= 0:
        raise RuntimeError(
            "Invalid geometry width."
        )

    scale = target_width_um / pixel_width

    print("\nPhysical scaling")
    print("----------------")
    print(
        f"Raster width      : "
        f"{pixel_width:.3f} px"
    )
    print(
        f"Raster height     : "
        f"{pixel_height:.3f} px"
    )
    print(
        f"Target width      : "
        f"{target_width_um:.6f} um"
    )
    print(
        f"Scale             : "
        f"{scale:.9f} um/px"
    )
    print(
        f"Resulting height  : "
        f"{pixel_height * scale:.6f} um"
    )

    scaled = []

    for polygon in polygons:

        p = polygon.copy()

        p[:, 0] = (
            p[:, 0] - xmin
        ) * scale

        p[:, 1] = (
            p[:, 1] - ymin
        ) * scale

        scaled.append(p)

    return scaled


# ============================================================
# SVG WRITER
# ============================================================

def write_polygon_svg(
    polygons,
    output_path,
    width_um,
    height_um,
):

    print("\nWriting processed SVG...")

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            '<?xml version="1.0" '
            'encoding="UTF-8"?>\n'
        )

        f.write(
            '<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{width_um:.9f}um" '
            f'height="{height_um:.9f}um" '
            f'viewBox="0 0 {width_um:.9f} '
            f'{height_um:.9f}">\n'
        )

        f.write(
            '  <g fill="#000000" '
            'stroke="none">\n'
        )

        for polygon in polygons:

            points = " ".join(
                f"{x:.9f},{y:.9f}"
                for x, y in polygon
            )

            f.write(
                f'    <polygon points="{points}"/>\n'
            )

        f.write("  </g>\n")
        f.write("</svg>\n")

    print(
        f"Output file : {output_path}"
    )


# ============================================================
# PREVIEW
# ============================================================

def write_preview(
    image,
    mask,
    preview_path,
):

    print(
        f"\nWriting preview : {preview_path}"
    )

    preview = np.zeros(
        (
            mask.shape[0],
            mask.shape[1],
            3,
        ),
        dtype=np.uint8,
    )

    # White background
    preview[:, :, :] = 255

    # Red detected geometry
    preview[mask] = [
        220,
        30,
        30,
    ]

    Image.fromarray(
        preview
    ).save(
        preview_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Convert complex SNSPD SVG "
            "artwork into FEM-compatible "
            "polygon geometry."
        )
    )

    parser.add_argument(
        "input_svg",
        help="Input SVG file",
    )

    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output processed SVG",
    )

    parser.add_argument(
        "--preview",
        default=None,
        help="Optional PNG preview",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_RENDER_SCALE,
        help=(
            "Rasterization scale factor "
            "(default: 3)"
        ),
    )

    parser.add_argument(
        "--width-um",
        type=float,
        default=DEFAULT_WIDTH_UM,
        help=(
            "Physical width of resulting "
            "geometry in micrometers "
            "(default: 50 um)"
        ),
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=DEFAULT_MIN_COMPONENT_AREA,
        help=(
            "Minimum connected-component "
            "area in raster pixels."
        ),
    )

    parser.add_argument(
        "--min-component-fraction",
        type=float,
        default=0.01,
        help=(
            "Small disconnected components below this fraction "
            "of the largest component are treated as raster "
            "artifacts. Default: 0.01 (1%%)."
        ),
    )

    parser.add_argument(
        "--simplify",
        type=float,
        default=DEFAULT_SIMPLIFY,
        help=(
            "Polygon simplification tolerance "
            "in pixels."
        ),
    )

    args = parser.parse_args()

    input_svg = os.path.abspath(
        args.input_svg
    )

    if not os.path.isfile(input_svg):

        raise FileNotFoundError(
            f"Input SVG not found:\n{input_svg}"
        )

    if args.output is None:

        base, ext = os.path.splitext(
            input_svg
        )

        output_path = (
            base
            + "_processed"
            + ".svg"
        )

    else:

        output_path = os.path.abspath(
            args.output
        )

    if args.preview is None:

        base, _ = os.path.splitext(
            output_path
        )

        preview_path = (
            base
            + "_preview.png"
        )

    else:

        preview_path = os.path.abspath(
            args.preview
        )

    print(
        "\n"
        "===================================================="
    )

    print(
        "SNSPD SVG PREPROCESSOR"
    )

    print(
        "===================================================="
    )

    print(
        f"\nInput SVG : {input_svg}"
    )

    print(
        f"Output SVG: {output_path}"
    )

    # --------------------------------------------------------
    # 1. Render
    # --------------------------------------------------------

    image = render_svg(
        input_svg,
        args.scale,
    )

    # --------------------------------------------------------
    # 2. Detect red/pink geometry
    # --------------------------------------------------------

    print(
        "\nDetecting red/pink SNSPD geometry..."
    )

    mask = create_red_mask(
        image
    )

    detected_pixels = np.count_nonzero(
        mask
    )

    total_pixels = mask.size

    print(
        f"Detected pixels : "
        f"{detected_pixels}"
    )

    print(
        f"Coverage        : "
        f"{100 * detected_pixels / total_pixels:.4f}%"
    )

    if detected_pixels == 0:

        raise RuntimeError(
            "\nNo red/pink geometry detected.\n"
            "The SVG may use a different color "
            "representation."
        )

    # --------------------------------------------------------
    # 3. Clean
    # --------------------------------------------------------

    mask = clean_mask(
        mask,
        args.min_area,
        args.min_component_fraction,
    )

    # --------------------------------------------------------
    # 4. Final connectivity verification
    # --------------------------------------------------------

    final_labels, final_components = ndimage.label(mask)

    if final_components != 1:
        raise RuntimeError(
            f"Electrical SNSPD geometry is not connected: "
            f"{final_components} components remain after cleanup."
        )

    print(
        f"\nElectrical geometry connectivity : PASS "
        f"(1 connected conductor)"
    )

    # --------------------------------------------------------
    # 5. Preview
    # --------------------------------------------------------

    write_preview(
        image,
        mask,
        preview_path,
    )

    # --------------------------------------------------------
    # 6. Extract contours
    # --------------------------------------------------------

    polygons = extract_contours(
        mask,
        args.simplify,
    )

    # --------------------------------------------------------
    # 7. Scale
    # --------------------------------------------------------

    polygons = scale_polygons(
        polygons,
        args.width_um,
    )

    # --------------------------------------------------------
    # Determine final dimensions
    # --------------------------------------------------------

    all_points = np.vstack(
        polygons
    )

    width_um = (
        np.max(all_points[:, 0])
        -
        np.min(all_points[:, 0])
    )

    height_um = (
        np.max(all_points[:, 1])
        -
        np.min(all_points[:, 1])
    )

    # --------------------------------------------------------
    # 8. Write clean SVG
    # --------------------------------------------------------

    write_polygon_svg(
        polygons,
        output_path,
        width_um,
        height_um,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\n"
        "===================================================="
    )

    print(
        "PREPROCESSING COMPLETE"
    )

    print(
        "===================================================="
    )

    print(
        f"\nInput geometry"
    )

    print(
        f"    SVG size          : "
        f"{image.width} × {image.height} px"
    )

    print(
        f"    Detected contours : "
        f"{len(polygons)}"
    )

    print(
        f"\nPhysical geometry"
    )

    print(
        f"    Width             : "
        f"{width_um:.6f} um"
    )

    print(
        f"    Height            : "
        f"{height_um:.6f} um"
    )

    print(
        f"\nGenerated files"
    )

    print(
        f"    SVG               : "
        f"{output_path}"
    )

    print(
        f"    Preview           : "
        f"{preview_path}"
    )

    print(
        "\nNext step:"
    )

    print(
        "    Open the preview PNG first."
    )

    print(
        "    If the detected region is correct,"
    )

    print(
        "    run the processed SVG through"
    )

    print(
        "    run_geometry.py."
    )

    print(
        "\n====================================================\n"
    )


if __name__ == "__main__":
    main()