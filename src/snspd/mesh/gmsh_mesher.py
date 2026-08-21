# FILE: src/snspd/mesh/gmsh_mesher.py
# PURPOSE:
# Converts canonical SNSPD geometry into a finite-element mesh using Gmsh.
#
# The rest of the simulator sees only the canonical Mesh class.
# Gmsh is treated strictly as the mesh-generation backend.
#
# Current implementation:
#     - 2D triangular mesh
#     - Linear P1 elements
#     - Physical-region tagging
#     - Shapely Polygon -> Gmsh Plane Surface
#
# Important design principle:
#     The physics solvers must not depend directly on Gmsh.
#
# Future extensions:
#     - adaptive refinement
#     - boundary-layer meshes
#     - higher-order elements
#     - anisotropic refinement
#     - physics-driven refinement
#     - curved/high-order geometry
#     - boundary-condition tagging
#     - 3D multilayer meshing


from __future__ import annotations


import gmsh
import numpy as np


from snspd.geometry.geometry import (
    DeviceGeometry,
    GeometryRegion,
)


from snspd.mesh.mesh import Mesh


class GmshMesher:
    """
    Gmsh-based 2D finite-element mesh generator.

    The mesher converts the canonical DeviceGeometry representation
    into a canonical Mesh representation.

    Gmsh is deliberately isolated inside this class so that later
    physics solvers do not depend on Gmsh's internal data structures.
    """

    def __init__(
        self,
        characteristic_length: float,
        corner_refinement_length: float | None = None,
        corner_refinement_radius: float | None = None,
    ):
        """
        Parameters
        ----------
        characteristic_length:
            Target characteristic element size in meters.

        Example
        -------
        0.25e-6 -> 0.25 µm
        """

        if characteristic_length <= 0:

            raise ValueError(
                "Characteristic mesh length "
                "must be positive."
            )

        self.characteristic_length = (
            float(characteristic_length)
        )

        if corner_refinement_length is not None:
            if corner_refinement_length <= 0.0:
                raise ValueError(
                    "Corner refinement length must be positive."
                )
            if corner_refinement_length > self.characteristic_length:
                raise ValueError(
                    "Corner refinement length cannot exceed "
                    "the global characteristic length."
                )

        if corner_refinement_radius is not None:
            if corner_refinement_radius <= 0.0:
                raise ValueError(
                    "Corner refinement radius must be positive."
                )

        self.corner_refinement_length = (
            None
            if corner_refinement_length is None
            else float(corner_refinement_length)
        )

        self.corner_refinement_radius = (
            None
            if corner_refinement_radius is None
            else float(corner_refinement_radius)
        )

    # ============================================================
    # POLYGON -> GMSH SURFACE
    # ============================================================

    def _add_polygon(
        self,
        region: GeometryRegion,
    ) -> int:
        """
        Add one Shapely Polygon to the Gmsh geometry kernel.

        Parameters
        ----------
        region:
            Canonical physical geometry region.

        Returns
        -------
        int
            Gmsh surface tag.

        Notes
        -----
        The polygon may contain interior holes.

        The coordinates are already stored internally in SI units
        [m], so no unit conversion is performed here.
        """

        polygon = region.polygon

        # --------------------------------------------------------
        # Basic validation
        # --------------------------------------------------------

        if polygon.is_empty:

            raise ValueError(
                f"Region '{region.name}' is empty."
            )

        if polygon.geom_type != "Polygon":

            raise ValueError(
                f"Region '{region.name}' must be a Polygon. "
                f"Got {polygon.geom_type}."
            )

        if not polygon.is_valid:

            raise ValueError(
                f"Region '{region.name}' is invalid: "
                f"{polygon}"
            )

        # ========================================================
        # EXTERIOR BOUNDARY
        # ========================================================

        exterior = list(
            polygon.exterior.coords
        )

        # Shapely repeats the first coordinate at the end.
        # Gmsh does not require this duplicate point.
        if (
            len(exterior) > 1
            and exterior[0] == exterior[-1]
        ):

            exterior = exterior[:-1]

        if len(exterior) < 3:

            raise ValueError(
                f"Region '{region.name}' exterior "
                "boundary contains fewer than 3 points."
            )

        point_tags = []

        for x, y in exterior:

            point_tag = (
                gmsh.model.geo.addPoint(
                    float(x),
                    float(y),
                    0.0,
                    self.characteristic_length,
                )
            )

            point_tags.append(
                point_tag
            )

        # --------------------------------------------------------
        # Exterior boundary lines
        # --------------------------------------------------------

        line_tags = []

        for i in range(
            len(point_tags)
        ):

            start = point_tags[i]

            end = point_tags[
                (i + 1) % len(point_tags)
            ]

            line_tag = (
                gmsh.model.geo.addLine(
                    start,
                    end,
                )
            )

            line_tags.append(
                line_tag
            )

        exterior_loop = (
            gmsh.model.geo.addCurveLoop(
                line_tags
            )
        )

        # ========================================================
        # INTERIOR HOLES
        # ========================================================

        hole_loops = []

        for interior in polygon.interiors:

            coordinates = list(
                interior.coords
            )

            if (
                len(coordinates) > 1
                and coordinates[0]
                == coordinates[-1]
            ):

                coordinates = (
                    coordinates[:-1]
                )

            if len(coordinates) < 3:

                raise ValueError(
                    f"Region '{region.name}' contains "
                    "an invalid interior boundary."
                )

            hole_point_tags = []

            for x, y in coordinates:

                point_tag = (
                    gmsh.model.geo.addPoint(
                        float(x),
                        float(y),
                        0.0,
                        self.characteristic_length,
                    )
                )

                hole_point_tags.append(
                    point_tag
                )

            hole_line_tags = []

            for i in range(
                len(hole_point_tags)
            ):

                start = (
                    hole_point_tags[i]
                )

                end = hole_point_tags[
                    (i + 1)
                    % len(hole_point_tags)
                ]

                line_tag = (
                    gmsh.model.geo.addLine(
                        start,
                        end,
                    )
                )

                hole_line_tags.append(
                    line_tag
                )

            hole_loop = (
                gmsh.model.geo.addCurveLoop(
                    hole_line_tags
                )
            )

            hole_loops.append(
                hole_loop
            )

        # ========================================================
        # PLANE SURFACE
        # ========================================================

        surface_tag = (
            gmsh.model.geo.addPlaneSurface(
                [
                    exterior_loop,
                    *hole_loops,
                ]
            )
        )

        return int(
            surface_tag
        )

    # ============================================================
    # PHYSICS-DRIVEN LOCAL CORNER REFINEMENT
    # ============================================================

    def _add_corner_refinement_fields(
        self,
        geometry: DeviceGeometry,
    ) -> None:
        """
        Add Gmsh Distance/Threshold fields around sharp boundary
        vertices.

        The refinement is purely geometric: it does not alter the
        FEM equations or fabricate sub-element physics.

        If corner_refinement_length is None, the legacy uniform mesh
        is retained.

        The field is applied around polygon vertices.  For an SNSPD
        meander, this provides the fine mesh required to resolve the
        local current-density singular/asymptotic region near turns.

        Parameters
        ----------
        geometry:
            Canonical device geometry.
        """

        if (
            self.corner_refinement_length is None
            or self.corner_refinement_radius is None
        ):
            return

        point_tags = []

        # Reconstructing the Gmsh point tags from geometry would be
        # unreliable because tags are local to _add_polygon().  Instead,
        # query all 0-D entities created by the CAD kernel and use their
        # coordinates.  The distance field is therefore applied to every
        # CAD vertex, which is conservative and remains geometry-driven.
        entities = gmsh.model.getEntities(0)

        for _, tag in entities:
            point_tags.append(int(tag))

        if not point_tags:
            return

        distance_field = gmsh.model.mesh.field.add("Distance")

        gmsh.model.mesh.field.setNumbers(
            distance_field,
            "NodesList",
            point_tags,
        )

        threshold_field = gmsh.model.mesh.field.add("Threshold")

        gmsh.model.mesh.field.setNumber(
            threshold_field,
            "InField",
            distance_field,
        )

        gmsh.model.mesh.field.setNumber(
            threshold_field,
            "SizeMin",
            self.corner_refinement_length,
        )

        gmsh.model.mesh.field.setNumber(
            threshold_field,
            "SizeMax",
            self.characteristic_length,
        )

        gmsh.model.mesh.field.setNumber(
            threshold_field,
            "DistMin",
            0.0,
        )

        gmsh.model.mesh.field.setNumber(
            threshold_field,
            "DistMax",
            self.corner_refinement_radius,
        )

        gmsh.model.mesh.field.setAsBackgroundMesh(
            threshold_field
        )

    # ============================================================
    # MESH GENERATION
    # ============================================================

    def generate(
        self,
        geometry: DeviceGeometry,
    ) -> Mesh:
        """
        Generate a 2D triangular FEM mesh.

        Parameters
        ----------
        geometry:
            Canonical SNSPD DeviceGeometry.

        Returns
        -------
        Mesh
            Canonical FEM mesh containing:

                nodes
                triangles
                physical_regions
                physical_names
                metadata
        """

        # ========================================================
        # GEOMETRY VALIDATION
        # ========================================================

        validation_errors = (
            geometry.validate()
        )

        if validation_errors:

            raise ValueError(
                "Cannot mesh invalid geometry:\n"
                + "\n".join(
                    validation_errors
                )
            )

        if geometry.region_count == 0:

            raise ValueError(
                "Cannot mesh empty geometry."
            )

        # ========================================================
        # START GMSH
        # ========================================================

        gmsh.initialize()

        try:

            # ----------------------------------------------------
            # Terminal output
            # ----------------------------------------------------

            gmsh.option.setNumber(
                "General.Terminal",
                1,
            )

            # ----------------------------------------------------
            # Create Gmsh model
            # ----------------------------------------------------

            gmsh.model.add(
                "snspd_device"
            )

            # ====================================================
            # CREATE CAD SURFACES
            # ====================================================

            surface_tags = []

            for region in geometry.regions:

                surface_tag = (
                    self._add_polygon(
                        region
                    )
                )

                surface_tags.append(
                    surface_tag
                )

            # ====================================================
            # SYNCHRONIZE CAD GEOMETRY
            # ====================================================
            #
            # IMPORTANT:
            #
            # Physical groups must be created AFTER the geometry
            # kernel has been synchronized.
            #
            # Creating them before synchronization caused:
            #
            #     Unknown entity of dimension 2
            #
            # in the previous implementation.
            #
            # ====================================================

            gmsh.model.geo.synchronize()

            # ====================================================
            # CREATE PHYSICAL GROUPS
            # ====================================================
            #
            # Each physical region receives an explicit ID:
            #
            #     region 0 -> physical ID 1
            #     region 1 -> physical ID 2
            #     ...
            #
            # These IDs will later be used by the FEM solver to
            # determine material properties and equations.
            #
            # ====================================================

            physical_names = {}

            for index, (
                region,
                surface_tag,
            ) in enumerate(
                zip(
                    geometry.regions,
                    surface_tags,
                )
            ):

                physical_tag = (
                    index + 1
                )

                gmsh.model.addPhysicalGroup(
                    2,
                    [surface_tag],
                    physical_tag,
                )

                gmsh.model.setPhysicalName(
                    2,
                    physical_tag,
                    region.name,
                )

                physical_names[
                    physical_tag
                ] = region.name

            # ====================================================
            # MESH CONTROL
            # ====================================================

            # The global characteristic length remains the fallback.
            # Local Distance/Threshold fields below override it around
            # geometry vertices when requested.
            gmsh.option.setNumber(
                "Mesh.CharacteristicLengthMin",
                min(
                    self.characteristic_length,
                    self.corner_refinement_length
                    if self.corner_refinement_length is not None
                    else self.characteristic_length,
                ),
            )

            gmsh.option.setNumber(
                "Mesh.CharacteristicLengthMax",
                self.characteristic_length,
            )

            if (
                self.corner_refinement_length is not None
                and self.corner_refinement_radius is not None
            ):
                self._add_corner_refinement_fields(
                    geometry
                )

            # ----------------------------------------------------
            # Frontal-Delaunay 2D algorithm.
            # ----------------------------------------------------

            gmsh.option.setNumber(
                "Mesh.Algorithm",
                6,
            )

            # ====================================================
            # GENERATE 2D MESH
            # ====================================================

            gmsh.model.mesh.generate(
                2
            )

            # ====================================================
            # EXTRACT NODES
            # ====================================================

            (
                node_tags,
                node_coordinates,
                _,
            ) = gmsh.model.mesh.getNodes()

            node_tags = np.asarray(
                node_tags,
                dtype=int,
            )

            node_coordinates = np.asarray(
                node_coordinates,
                dtype=float,
            )

            if len(node_tags) == 0:

                raise RuntimeError(
                    "Gmsh generated no mesh nodes."
                )

            # Gmsh returns XYZ coordinates even for a 2D mesh.
            #
            # Convert:
            #
            #     [x1,y1,z1,x2,y2,z2,...]
            #
            # into:
            #
            #     [[x1,y1],
            #      [x2,y2],
            #      ...]
            #

            nodes = (
                node_coordinates
                .reshape(
                    -1,
                    3,
                )[:, :2]
            )

            # ----------------------------------------------------
            # Gmsh node tags are not guaranteed to be:
            #
            #     0, 1, 2, 3, ...
            #
            # so construct an explicit mapping.
            # ----------------------------------------------------

            node_map = {
                int(tag): index
                for index, tag
                in enumerate(
                    node_tags
                )
            }

            # ====================================================
            # EXTRACT TRIANGULAR ELEMENTS
            # ====================================================

            (
                element_types,
                element_tags,
                element_nodes,
            ) = gmsh.model.mesh.getElements(
                2
            )

            triangles = []

            physical_regions = []

            # ====================================================
            # BUILD ELEMENT -> PHYSICAL REGION MAP
            # ====================================================
            #
            # This is the important correction to the previous
            # implementation.
            #
            # We DO NOT use:
            #
            #     gmsh.model.mesh.getElement(...)
            #
            # to try to recover the geometric entity.
            #
            # Instead, because we know which surface belongs to
            # which physical region, we directly ask Gmsh for the
            # elements belonging to each surface.
            #
            # Therefore:
            #
            #     element tag -> physical region
            #
            # is constructed explicitly and reliably.
            #
            # ====================================================

            element_physical_region = {}

            for region_index, surface_tag in enumerate(
                surface_tags
            ):

                physical_tag = (
                    region_index + 1
                )

                (
                    region_element_types,
                    region_element_tags,
                    _,
                ) = gmsh.model.mesh.getElements(
                    2,
                    surface_tag,
                )

                for (
                    region_element_type,
                    region_tags,
                ) in zip(
                    region_element_types,
                    region_element_tags,
                ):

                    # Gmsh element type 2:
                    #
                    # 3-node linear triangle
                    #
                    if (
                        region_element_type
                        != 2
                    ):
                        continue

                    for element_tag in region_tags:

                        element_physical_region[
                            int(element_tag)
                        ] = physical_tag

            # ====================================================
            # PROCESS TRIANGLES
            # ====================================================

            for (
                element_type,
                tags,
                connectivity,
            ) in zip(
                element_types,
                element_tags,
                element_nodes,
            ):

                # ------------------------------------------------
                # Only retain 3-node triangles.
                # ------------------------------------------------

                if element_type != 2:

                    continue

                connectivity = np.asarray(
                    connectivity,
                    dtype=int,
                ).reshape(
                    -1,
                    3,
                )

                tags = np.asarray(
                    tags,
                    dtype=int,
                )

                for (
                    element_index,
                    element,
                ) in enumerate(
                    connectivity
                ):

                    element_tag = int(
                        tags[
                            element_index
                        ]
                    )

                    # ------------------------------------------------
                    # Convert Gmsh node IDs to our zero-based
                    # canonical node indices.
                    # ------------------------------------------------

                    try:

                        triangle = [
                            node_map[
                                int(
                                    element[0]
                                )
                            ],
                            node_map[
                                int(
                                    element[1]
                                )
                            ],
                            node_map[
                                int(
                                    element[2]
                                )
                            ],
                        ]

                    except KeyError as exc:

                        raise RuntimeError(
                            "Gmsh triangle references "
                            f"unknown node tag: {exc}"
                        ) from exc

                    triangles.append(
                        triangle
                    )

                    # ------------------------------------------------
                    # Recover physical region.
                    # ------------------------------------------------

                    if (
                        element_tag
                        not in element_physical_region
                    ):

                        raise RuntimeError(
                            f"Triangle element "
                            f"{element_tag} has no "
                            "physical-region assignment."
                        )

                    physical_regions.append(
                        element_physical_region[
                            element_tag
                        ]
                    )

            # ====================================================
            # FINALIZE ELEMENT ARRAYS
            # ====================================================

            if not triangles:

                raise RuntimeError(
                    "Gmsh generated no triangular elements."
                )

            triangles = np.asarray(
                triangles,
                dtype=int,
            )

            physical_regions = np.asarray(
                physical_regions,
                dtype=int,
            )

            # ====================================================
            # SANITY CHECK
            # ====================================================

            if len(triangles) != len(
                physical_regions
            ):

                raise RuntimeError(
                    "Triangle count does not match "
                    "physical-region count."
                )

            # ====================================================
            # CREATE CANONICAL MESH
            # ====================================================

            mesh = Mesh(
                nodes=nodes,
                triangles=triangles,
                physical_regions=(
                    physical_regions
                ),
                physical_names=(
                    physical_names
                ),
                metadata={
                    "backend": "gmsh",
                    "element_type": (
                        "triangle3"
                    ),
                    "element_order": 1,
                    "dimension": 2,
                    "characteristic_length_m": (
                        self.characteristic_length
                    ),
                    "corner_refinement_length_m": (
                        self.corner_refinement_length
                    ),
                    "corner_refinement_radius_m": (
                        self.corner_refinement_radius
                    ),
                },
            )

            # ====================================================
            # FINAL MESH VALIDATION
            # ====================================================

            mesh_errors = (
                mesh.validate()
            )

            if mesh_errors:

                raise RuntimeError(
                    "Generated mesh failed "
                    "canonical mesh validation:\n"
                    + "\n".join(
                        mesh_errors
                    )
                )

            return mesh

        finally:

            # ----------------------------------------------------
            # Always shut down Gmsh, including when an exception
            # occurs.
            # ----------------------------------------------------

            gmsh.finalize()