# FILE: src/snspd/mesh/quality.py
# PURPOSE:
# Calculates numerical quality metrics for the SNSPD FEM mesh.
#
# Poor-quality elements can produce inaccurate or unstable FEM solutions.
# This module therefore forms part of the numerical verification pipeline.
#
# Current metrics:
#     - minimum triangle angle
#     - maximum triangle angle
#     - aspect ratio
#     - minimum area
#     - maximum area
#     - mean area

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from snspd.mesh.mesh import Mesh


@dataclass
class MeshQuality:
    """
    Mesh quality statistics.
    """

    minimum_angle_deg: float

    maximum_angle_deg: float

    minimum_area_m2: float

    maximum_area_m2: float

    mean_area_m2: float

    maximum_aspect_ratio: float

    mean_aspect_ratio: float


def _triangle_angles(
    points: np.ndarray,
) -> np.ndarray:
    """
    Calculate the three internal angles of a triangle.

    Parameters
    ----------
    points:
        Array with shape (3, 2).

    Returns
    -------
    ndarray
        Three angles in degrees.
    """

    a = np.linalg.norm(
        points[1] - points[2]
    )

    b = np.linalg.norm(
        points[0] - points[2]
    )

    c = np.linalg.norm(
        points[0] - points[1]
    )

    sides = np.array(
        [a, b, c],
        dtype=float,
    )

    if np.any(sides <= 0):

        return np.array(
            [0.0, 0.0, 0.0]
        )

    # Cosine rule.
    cos_A = (
        b**2
        + c**2
        - a**2
    ) / (
        2 * b * c
    )

    cos_B = (
        a**2
        + c**2
        - b**2
    ) / (
        2 * a * c
    )

    cos_C = (
        a**2
        + b**2
        - c**2
    ) / (
        2 * a * b
    )

    cosines = np.clip(
        [cos_A, cos_B, cos_C],
        -1.0,
        1.0,
    )

    return np.degrees(
        np.arccos(
            cosines
        )
    )


def _triangle_aspect_ratio(
    points: np.ndarray,
) -> float:
    """
    Calculate a simple triangle aspect ratio:

        longest side / shortest altitude

    A value near 1 represents a well-shaped triangle.
    Large values indicate a poorly shaped triangle.
    """

    side_lengths = np.array(
        [
            np.linalg.norm(
                points[1] - points[0]
            ),
            np.linalg.norm(
                points[2] - points[1]
            ),
            np.linalg.norm(
                points[0] - points[2]
            ),
        ]
    )

    if np.min(side_lengths) <= 0:
        return np.inf

    return float(
        np.max(side_lengths)
        / np.min(side_lengths)
    )


def analyze_mesh_quality(
    mesh: Mesh,
) -> MeshQuality:
    """
    Analyze the quality of the complete FEM mesh.
    """

    if mesh.element_count == 0:

        raise ValueError(
            "Cannot analyze an empty mesh."
        )

    areas = mesh.triangle_areas()

    minimum_angles = []
    maximum_angles = []
    aspect_ratios = []

    for triangle_index in range(
        mesh.element_count
    ):

        points = (
            mesh.triangle_coordinates(
                triangle_index
            )
        )

        angles = _triangle_angles(
            points
        )

        minimum_angles.append(
            np.min(angles)
        )

        maximum_angles.append(
            np.max(angles)
        )

        aspect_ratios.append(
            _triangle_aspect_ratio(
                points
            )
        )

    return MeshQuality(
        minimum_angle_deg=float(
            np.min(
                minimum_angles
            )
        ),
        maximum_angle_deg=float(
            np.max(
                maximum_angles
            )
        ),
        minimum_area_m2=float(
            np.min(areas)
        ),
        maximum_area_m2=float(
            np.max(areas)
        ),
        mean_area_m2=float(
            np.mean(areas)
        ),
        maximum_aspect_ratio=float(
            np.max(
                aspect_ratios
            )
        ),
        mean_aspect_ratio=float(
            np.mean(
                aspect_ratios
            )
        ),
    )


def format_mesh_quality(
    quality: MeshQuality,
) -> str:
    """
    Format mesh-quality statistics.
    """

    return (
        "\n"
        "SNSPD FEM MESH QUALITY\n"
        "======================\n"
        f"Minimum angle      : "
        f"{quality.minimum_angle_deg:.6f} deg\n"
        f"Maximum angle      : "
        f"{quality.maximum_angle_deg:.6f} deg\n"
        f"Minimum area       : "
        f"{quality.minimum_area_m2 * 1e12:.6e} um²\n"
        f"Maximum area       : "
        f"{quality.maximum_area_m2 * 1e12:.6e} um²\n"
        f"Mean area          : "
        f"{quality.mean_area_m2 * 1e12:.6e} um²\n"
        f"Maximum aspect     : "
        f"{quality.maximum_aspect_ratio:.6f}\n"
        f"Mean aspect        : "
        f"{quality.mean_aspect_ratio:.6f}\n"
    )