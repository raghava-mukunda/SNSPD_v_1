# FILE: tests/test_geometry.py
# PURPOSE:
# Tests the fundamental geometry representation and SVG import pipeline.
#
# These tests establish the first automated verification layer of the
# SNSPD digital-twin software.

from pathlib import Path

import pytest

from snsdp.geometry.svg_importer import (
    import_svg,
)


EXAMPLE_GEOMETRY = (
    Path(__file__).parents[1]
    / "examples"
    / "simple_meander.svg"
)


def test_svg_import():

    geometry = import_svg(
        EXAMPLE_GEOMETRY
    )

    assert (
        geometry.region_count > 0
    )


def test_geometry_is_valid():

    geometry = import_svg(
        EXAMPLE_GEOMETRY
    )

    errors = geometry.validate()

    assert errors == []


def test_geometry_has_positive_area():

    geometry = import_svg(
        EXAMPLE_GEOMETRY
    )

    assert (
        geometry.total_area > 0
    )


def test_geometry_has_positive_dimensions():

    geometry = import_svg(
        EXAMPLE_GEOMETRY
    )

    assert geometry.width > 0
    assert geometry.height > 0