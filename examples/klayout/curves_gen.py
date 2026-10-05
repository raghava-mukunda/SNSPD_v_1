import math
import klayout.db as kdb


# ============================================================
# DESIGN PARAMETERS
# ============================================================

# Everything below is in micrometers
# unless explicitly stated otherwise.

WIRE_WIDTH = 0.200       # um

RADIUS = 5.0             # centerline radius [um]

NUMBER_OF_POINTS = 128   # smoothness of curves

LAYER = 1
DATATYPE = 0

OUTPUT = "results/curves.gds"


# ============================================================
# CREATE LAYOUT
# ============================================================

layout = kdb.Layout()

# 1 database unit = 1 nm
layout.dbu = 0.001


# ============================================================
# CREATE CELL
# ============================================================

cell = layout.create_cell("CURVES")


# ============================================================
# CREATE LAYER
# ============================================================

layer = layout.layer(
    LAYER,
    DATATYPE
)


# ============================================================
# 1. FULL CIRCLE / CURVED WIRE
# ============================================================

circle_points = []

for i in range(NUMBER_OF_POINTS):

    theta = (
        2.0
        * math.pi
        * i
        / NUMBER_OF_POINTS
    )

    x = (
        10.0
        + RADIUS * math.cos(theta)
    )

    y = (
        10.0
        + RADIUS * math.sin(theta)
    )

    circle_points.append(
        kdb.DPoint(x, y)
    )


circle = kdb.DPath(
    circle_points,
    WIRE_WIDTH
)

cell.shapes(layer).insert(
    circle
)


# ============================================================
# 2. FULL DONUT
# ============================================================
#
# A donut is simply:
#
#       outer circle
#       -
#       inner circle
#
# The difference between them is the conductor.
#
# ============================================================

OUTER_RADIUS = 5.0
INNER_RADIUS = 4.0

outer_points = []
inner_points = []


for i in range(NUMBER_OF_POINTS):

    theta = (
        2.0
        * math.pi
        * i
        / NUMBER_OF_POINTS
    )

    # Outer circle
    outer_points.append(
        kdb.DPoint(
            30.0
            + OUTER_RADIUS * math.cos(theta),

            10.0
            + OUTER_RADIUS * math.sin(theta)
        )
    )

    # Inner circle
    inner_points.append(
        kdb.DPoint(
            30.0
            + INNER_RADIUS * math.cos(theta),

            10.0
            + INNER_RADIUS * math.sin(theta)
        )
    )


outer = kdb.DPolygon(
    outer_points
)

inner = kdb.DPolygon(
    inner_points
)


# Convert to regions so we can perform
# a Boolean subtraction.

outer_region = kdb.Region(
    outer
)

inner_region = kdb.Region(
    inner
)


donut = (
    outer_region
    - inner_region
)


cell.shapes(layer).insert(
    donut
)


# ============================================================
# 3. HALF DONUT
# ============================================================
#
# Here we explicitly generate only:
#
#       0 <= theta <= pi
#
# which gives the upper half of the donut.
#
# ============================================================

HALF_OUTER_RADIUS = 5.0
HALF_INNER_RADIUS = 4.0

half_outer_points = []
half_inner_points = []


for i in range(
    NUMBER_OF_POINTS // 2 + 1
):

    theta = (
        math.pi
        * i
        / (NUMBER_OF_POINTS // 2)
    )

    # Outer semicircle
    half_outer_points.append(
        kdb.DPoint(
            50.0
            + HALF_OUTER_RADIUS
            * math.cos(theta),

            10.0
            + HALF_OUTER_RADIUS
            * math.sin(theta)
        )
    )

    # Inner semicircle
    half_inner_points.append(
        kdb.DPoint(
            50.0
            + HALF_INNER_RADIUS
            * math.cos(theta),

            10.0
            + HALF_INNER_RADIUS
            * math.sin(theta)
        )
    )


# ------------------------------------------------------------
# Join outer and inner boundaries.
#
# Outer boundary:
#
#       left -> right
#
# Inner boundary:
#
#       right -> left
#
# ------------------------------------------------------------

half_donut_points = (
    half_outer_points
    + list(
        reversed(
            half_inner_points
        )
    )
)


half_donut = kdb.DPolygon(
    half_donut_points
)


cell.shapes(layer).insert(
    half_donut
)


# ============================================================
# 4. HALF DONUT - OTHER DIRECTION
# ============================================================
#
# This one demonstrates that we can simply change
# the angular range.
#
# Here:
#
#       pi <= theta <= 2*pi
#
# ============================================================

half_outer_points_2 = []
half_inner_points_2 = []


for i in range(
    NUMBER_OF_POINTS // 2 + 1
):

    theta = (
        math.pi
        + math.pi
        * i
        / (NUMBER_OF_POINTS // 2)
    )

    half_outer_points_2.append(
        kdb.DPoint(
            70.0
            + HALF_OUTER_RADIUS
            * math.cos(theta),

            10.0
            + HALF_OUTER_RADIUS
            * math.sin(theta)
        )
    )

    half_inner_points_2.append(
        kdb.DPoint(
            70.0
            + HALF_INNER_RADIUS
            * math.cos(theta),

            10.0
            + HALF_INNER_RADIUS
            * math.sin(theta)
        )
    )


half_donut_points_2 = (
    half_outer_points_2
    + list(
        reversed(
            half_inner_points_2
        )
    )
)


half_donut_2 = kdb.DPolygon(
    half_donut_points_2
)


cell.shapes(layer).insert(
    half_donut_2
)


# ============================================================
# SAVE GDS
# ============================================================

layout.write(
    OUTPUT
)


print()
print("Curves written to:")
print(OUTPUT)