import klayout.db as kdb

import math
# ============================================================
# DESIGN PARAMETERS
# Change these values to change the box
# ============================================================

WIDTH = 10.0       # um
HEIGHT = 5.0       # um

LAYER = 1
DATATYPE = 0

OUTPUT = "results/box.gds"


# ============================================================
# CREATE LAYOUT
# ============================================================

layout = kdb.Layout()

# Database unit = 1 nm
layout.dbu = 0.001


# ============================================================
# CREATE CELL
# ============================================================

cell = layout.create_cell("BOX")


# ============================================================
# CREATE LAYER
# ============================================================

layer = layout.layer(
    LAYER,
    DATATYPE
)


# ============================================================
# DRAW BOX
# ============================================================
NUMBER_OF_Meander = 49

gap = 0.4
for i in range(NUMBER_OF_Meander):

    box2 = kdb.DBox(
        0.0,       # x1 [um]
        0.0+(i*gap),       # y1 [um]

        17,     # x2 [um]
        0.22 + (i*gap)    # y2 [um]
    )
    cell.shapes(layer).insert(box2)
for k in range(25):
    HALF_OUTER_RADIUS = 0.62/2
    HALF_INNER_RADIUS = 0.18/2
    NUMBER_OF_POINTS = 50
    half_outer_points = []
    half_inner_points = []


    for i in range(
        NUMBER_OF_POINTS // 2 + 1
    ):

        theta = (math.pi * i/ (NUMBER_OF_POINTS // 2))

        # Outer semicircle
        half_outer_points.append(kdb.DPoint(
            0+ HALF_OUTER_RADIUS* math.cos(theta+(math.pi/2)),

                (0.22+(0.18/2))+(2*k*gap)
                + HALF_OUTER_RADIUS
                * math.sin(theta+(math.pi/2))
            )
        )

        # Inner semicircle
        half_inner_points.append(
            kdb.DPoint(
                0
                + HALF_INNER_RADIUS
                * math.cos(theta+(math.pi/2)),

                (0.22+(0.18/2))+(2*k*gap)
                + HALF_INNER_RADIUS
                * math.sin(theta+(math.pi/2))
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
    # SAVE GDS
    # ============================================================

    layout.write(OUTPUT)



for k in range(25):
    HALF_OUTER_RADIUS = 0.62/2
    HALF_INNER_RADIUS = 0.18/2
    NUMBER_OF_POINTS = 50
    half_outer_points = []
    half_inner_points = []


    for i in range(
        NUMBER_OF_POINTS // 2 + 1
    ):

        theta = (math.pi * i/ (NUMBER_OF_POINTS // 2))

        # Outer semicircle
        half_outer_points.append(kdb.DPoint(
            17+ HALF_OUTER_RADIUS* math.cos(theta-(math.pi/2))*(3),

                (-(0.18/2))+(2*k*gap)
                + HALF_OUTER_RADIUS
                * math.sin(theta-(math.pi/2))
            )
        )

        # Inner semicircle
        half_inner_points.append(
            kdb.DPoint(
                17
                + HALF_INNER_RADIUS
                * math.cos(theta-(math.pi/2)),

                (-(0.18/2))+(2*k*gap)
                + HALF_INNER_RADIUS
                * math.sin(theta-(math.pi/2))
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
    # SAVE GDS
    # ============================================================

    layout.write(OUTPUT)

print("Created box:")
print(f"Width  = {WIDTH} um")
print(f"Height = {HEIGHT} um")
print(f"Output = {OUTPUT}")