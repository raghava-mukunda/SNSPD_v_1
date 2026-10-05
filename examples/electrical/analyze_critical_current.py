"""
SNSPD Clem-Berggren critical-current analysis.

This script is the Stage-4 interface between the validated FEM result
and the Clem-Berggren vortex-entry calculation.

The FEM supplies:
    J(x,y) [A/m^2]

The Clem-Berggren model uses:
    K(x,y) = d J(x,y)

The critical current is obtained from the local geometry-dependent
Gibbs-barrier criterion.

IMPORTANT:
------------
This script does NOT replace or weaken the Clem-Berggren physics.

The following are still handled by src/snspd/physics/clem_berggren.py:

    Lambda = 2 lambda^2 / d
    K = d J
    p = pi/alpha - 1
    G(delta) = E_self - W_I
    G(delta_c) = 0

The analyzer only handles:
    - input validation
    - material parameter selection
    - FEM loading
    - progress reporting
    - result reporting
    - result serialization
    - optional heatmap generation
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np


# ============================================================
# MATERIAL DATABASE
# ============================================================

# IMPORTANT:
#
# These are explicit model inputs.
#
# Do not silently infer lambda/xi from the FEM.
#
# If a material is not present, the user must supply:
#
#     --lambda-nm
#     --xi-nm
#
# This keeps the superconducting parameters explicit.

MATERIAL_DATABASE = {
    "NbTiN": {
        "lambda_nm": 450.0,
        "xi_nm": 5.0,
    },
}


# ============================================================
# ARGUMENT PARSER
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "SNSPD Clem-Berggren vortex-entry "
            "critical-current analysis."
        )
    )

    # --------------------------------------------------------
    # FEM input
    # --------------------------------------------------------

    parser.add_argument(
        "fem_file",
        type=str,
        help="Validated FEM .npz result file.",
    )

    # --------------------------------------------------------
    # Material
    # --------------------------------------------------------

    parser.add_argument(
        "--material",
        type=str,
        default="unspecified",
        help=(
            "Superconducting material. "
            "Currently supported built-in material: NbTiN."
        ),
    )

    # --------------------------------------------------------
    # Explicit superconducting parameters
    #
    # These remain available even when --material is supplied.
    # --------------------------------------------------------

    parser.add_argument(
        "--lambda-nm",
        type=float,
        default=None,
        help=(
            "London penetration depth at operating "
            "temperature [nm]."
        ),
    )

    parser.add_argument(
        "--xi-nm",
        type=float,
        default=None,
        help=(
            "Ginzburg-Landau coherence length at "
            "operating temperature [nm]."
        ),
    )

    parser.add_argument(
        "--temperature-k",
        type=float,
        required=True,
        help="Operating temperature [K].",
    )

    # --------------------------------------------------------
    # Compatibility with master pipeline
    #
    # --jc is accepted for pipeline compatibility but is NOT
    # used to override the Clem-Berggren critical-current model.
    # --------------------------------------------------------

    parser.add_argument(
        "--jc",
        type=float,
        default=None,
        help=(
            "Optional nominal depairing/current-density scale "
            "supplied by the master pipeline. It is recorded "
            "for provenance but does not replace the "
            "Clem-Berggren calculation."
        ),
    )

    # --------------------------------------------------------
    # Output paths
    # --------------------------------------------------------

    parser.add_argument(
        "--output",
        type=str,
        default=(
            "results/"
            "critical_current_heatmap.png"
        ),
        help=(
            "Output critical-current/current-density "
            "heatmap PNG."
        ),
    )

    parser.add_argument(
        "--output-npz",
        type=str,
        default=(
            "results/"
            "critical_current_clem_berggren.npz"
        ),
        help="Output numerical result .npz file.",
    )

    # Backward-compatible alias.
    parser.add_argument(
        "--output-legacy",
        type=str,
        default=None,
        help=argparse.SUPPRESS,
    )

    return parser.parse_args()


# ============================================================
# MATERIAL RESOLUTION
# ============================================================

def resolve_material_parameters(args):

    material_key = str(
        args.material
    ).strip()

    lambda_nm = args.lambda_nm
    xi_nm = args.xi_nm

    # --------------------------------------------------------
    # Explicit values take priority.
    # --------------------------------------------------------

    if (
        lambda_nm is not None
        and xi_nm is not None
    ):

        return (
            float(lambda_nm),
            float(xi_nm),
            "explicit",
        )

    # --------------------------------------------------------
    # Otherwise use material database.
    # --------------------------------------------------------

    if material_key in MATERIAL_DATABASE:

        material = MATERIAL_DATABASE[
            material_key
        ]

        if lambda_nm is None:

            lambda_nm = (
                material["lambda_nm"]
            )

        if xi_nm is None:

            xi_nm = (
                material["xi_nm"]
            )

        return (
            float(lambda_nm),
            float(xi_nm),
            "material database",
        )

    # --------------------------------------------------------
    # Missing material parameters.
    # --------------------------------------------------------

    raise ValueError(
        "\n"
        "Cannot determine superconducting parameters.\n\n"
        "Provide either:\n"
        "    --lambda-nm <value> --xi-nm <value>\n\n"
        "or use a supported material such as:\n"
        "    --material NbTiN\n"
    )


# ============================================================
# PROGRESS DISPLAY
# ============================================================

class ProgressDisplay:

    def __init__(self):

        self.start_time = time.perf_counter()
        self.last_message = ""

    def callback(
        self,
        done: int,
        total: int,
        message: str,
    ):

        elapsed = (
            time.perf_counter()
            - self.start_time
        )

        if total > 0:

            fraction = (
                done
                /
                total
            )

        else:

            fraction = 0.0

        fraction = max(
            0.0,
            min(
                1.0,
                fraction,
            ),
        )

        width = 40

        filled = int(
            width * fraction
        )

        bar = (
            "=" * filled
            + ">"
            + " " * max(
                0,
                width - filled - 1,
            )
        )

        # ----------------------------------------------------
        # ETA
        # ----------------------------------------------------

        if done > 0 and fraction > 0.0:

            estimated_total = (
                elapsed
                /
                fraction
            )

            eta = max(
                0.0,
                estimated_total
                - elapsed,
            )

        else:

            eta = 0.0

        if eta >= 60.0:

            eta_text = (
                f"{eta / 60.0:.1f} min"
            )

        else:

            eta_text = (
                f"{eta:.1f} s"
            )

        line = (
            f"\r"
            f"Clem-Berggren "
            f"[{bar}] "
            f"{done:4d}/{total:<4d} "
            f"{fraction * 100:6.2f}% "
            f"ETA {eta_text:<9} "
            f"{message:<20}"
        )

        print(
            line,
            end="",
            flush=True,
        )

        self.last_message = message

    def finish(self):

        print()


# ============================================================
# FEM LOADING
# ============================================================

def load_fem_result(
    fem_path: Path,
):

    if not fem_path.exists():

        raise FileNotFoundError(
            f"FEM result file not found:\n"
            f"{fem_path}"
        )

    print()
    print(
        "Loading validated FEM result..."
    )

    start = time.perf_counter()

    data = np.load(
        fem_path
    )

    required = [
        "nodes_m",
        "triangles",
        "triangle_centers_m",
        "element_J_magnitude_A_per_m2",
        "fem_current_A",
        "wire_width_m",
        "film_thickness_m",
    ]

    missing = [
        name
        for name in required
        if name not in data
    ]

    if missing:

        raise RuntimeError(
            "FEM result is missing required fields:\n"
            + "\n".join(
                f"    {name}"
                for name in missing
            )
        )

    nodes = np.asarray(
        data["nodes_m"],
        dtype=float,
    )

    triangles = np.asarray(
        data["triangles"],
        dtype=np.int64,
    )

    triangle_centers = np.asarray(
        data["triangle_centers_m"],
        dtype=float,
    )

    element_J = np.asarray(
        data[
            "element_J_magnitude_A_per_m2"
        ],
        dtype=float,
    )

    fem_current = float(
        data["fem_current_A"]
    )

    wire_width = float(
        data["wire_width_m"]
    )

    film_thickness = float(
        data["film_thickness_m"]
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    print(
        f"Loaded FEM result in "
        f"{elapsed:.2f} s"
    )

    return (
        data,
        nodes,
        triangles,
        triangle_centers,
        element_J,
        fem_current,
        wire_width,
        film_thickness,
    )


# ============================================================
# HEATMAP
# ============================================================

def generate_heatmap(
    fem_data,
    nodes,
    triangles,
    element_J,
    film_thickness,
    limiting_x,
    limiting_y,
    output_path,
):

    try:

        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri

    except ImportError:

        print(
            "\nWARNING:"
        )

        print(
            "matplotlib is not installed."
        )

        print(
            "Skipping heatmap generation."
        )

        return False

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Convert FEM volume current density to sheet current.
    # --------------------------------------------------------

    K = (
        element_J
        * film_thickness
    )

    triangulation = (
        mtri.Triangulation(
            nodes[:, 0],
            nodes[:, 1],
            triangles,
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 8)
    )

    # --------------------------------------------------------
    # Plot sheet current density.
    # --------------------------------------------------------

    field = ax.tripcolor(
        triangulation,
        K,
        shading="flat",
    )

    cbar = fig.colorbar(
        field,
        ax=ax,
    )

    cbar.set_label(
        "Sheet current density K [A/m]"
    )

    # --------------------------------------------------------
    # Limiting location.
    # --------------------------------------------------------

    ax.scatter(
        limiting_x,
        limiting_y,
        marker="x",
        s=100,
        linewidths=2,
    )

    ax.set_xlabel(
        "x [m]"
    )

    ax.set_ylabel(
        "y [m]"
    )

    ax.set_title(
        "SNSPD FEM Sheet Current Density\n"
        "Clem-Berggren Limiting Location"
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=200,
    )

    plt.close(
        fig
    )

    print()
    print(
        "Critical-current heatmap saved to:"
    )
    print(
        output_path.resolve()
    )

    return True


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_arguments()

    fem_path = Path(
        args.fem_file
    )

    (
        lambda_nm,
        xi_nm,
        parameter_source,
    ) = resolve_material_parameters(
        args
    )

    (
        fem_data,
        nodes,
        triangles,
        triangle_centers,
        element_J,
        fem_current,
        wire_width,
        film_thickness,
    ) = load_fem_result(
        fem_path
    )

    # --------------------------------------------------------
    # Validate numerical inputs.
    # --------------------------------------------------------

    if lambda_nm <= 0.0:

        raise ValueError(
            "lambda must be positive."
        )

    if xi_nm <= 0.0:

        raise ValueError(
            "xi must be positive."
        )

    if args.temperature_k < 0.0:

        raise ValueError(
            "Temperature cannot be negative."
        )

    # --------------------------------------------------------
    # Convert to SI.
    # --------------------------------------------------------

    lambda_m = (
        lambda_nm
        * 1e-9
    )

    xi_m = (
        xi_nm
        * 1e-9
    )

    # --------------------------------------------------------
    # Import physics model.
    # --------------------------------------------------------

    from snspd.physics.clem_berggren import (
        ClemBerggrenParameters,
        analyze_clem_berggren,
        format_result,
    )

    params = ClemBerggrenParameters(
        wire_width_m=wire_width,
        film_thickness_m=film_thickness,
        penetration_depth_m=lambda_m,
        coherence_length_m=xi_m,
        temperature_k=args.temperature_k,
        material=args.material,
    )

    # --------------------------------------------------------
    # Header.
    # --------------------------------------------------------

    print()
    print(
        "=" * 72
    )

    print(
        "STAGE 4 / CLEM-BERGGREN "
        "CRITICAL CURRENT"
    )

    print(
        "=" * 72
    )

    print()
    print(
        "INPUT"
    )
    print(
        "-----"
    )

    print(
        f"FEM result file          : "
        f"{fem_path.resolve()}"
    )

    print(
        f"Material                 : "
        f"{args.material}"
    )

    print(
        f"Temperature              : "
        f"{args.temperature_k:.6f} K"
    )

    print(
        f"Parameter source         : "
        f"{parameter_source}"
    )

    print(
        f"Lambda                   : "
        f"{lambda_nm:.6f} nm"
    )

    print(
        f"Xi                       : "
        f"{xi_nm:.6f} nm"
    )

    print(
        f"Wire width               : "
        f"{wire_width * 1e9:.6f} nm"
    )

    print(
        f"Film thickness           : "
        f"{film_thickness * 1e9:.6f} nm"
    )

    print(
        f"FEM transport current    : "
        f"{fem_current:.12e} A"
    )

    if args.jc is not None:

        print(
            f"Pipeline Jc reference    : "
            f"{args.jc:.12e} A/m²"
        )

        print(
            "NOTE: --jc is recorded for "
            "pipeline compatibility only."
        )

        print(
            "      It does NOT replace the "
            "Clem-Berggren calculation."
        )

    print(
        f"Pearl length             : "
        f"{params.pearl_length_m * 1e6:.6f} um"
    )

    # --------------------------------------------------------
    # FEM statistics.
    # --------------------------------------------------------

    print()
    print(
        "FEM MESH"
    )
    print(
        "--------"
    )

    print(
        f"Nodes                    : "
        f"{len(nodes):,}"
    )

    print(
        f"Triangles                : "
        f"{len(triangles):,}"
    )

    print(
        f"Maximum |J|              : "
        f"{np.nanmax(element_J):.6e} A/m²"
    )

    print(
        f"Mean |J|                 : "
        f"{np.nanmean(element_J):.6e} A/m²"
    )

    # --------------------------------------------------------
    # Physics validity.
    # --------------------------------------------------------

    print()
    print(
        "CLEM-BERGGREN VALIDITY"
    )
    print(
        "----------------------"
    )

    print(
        f"d / lambda               : "
        f"{params.thickness_to_lambda_ratio:.6e}"
    )

    print(
        f"W / Lambda               : "
        f"{params.width_to_pearl_ratio:.6e}"
    )

    print(
        f"xi / W                   : "
        f"{params.coherence_to_width_ratio:.6e}"
    )

    print(
        f"d << lambda              : "
        f"{'PASS' if params.thickness_to_lambda_ratio < 0.1 else 'CHECK'}"
    )

    print(
        f"W << Lambda              : "
        f"{'PASS' if params.width_to_pearl_ratio < 0.1 else 'CHECK'}"
    )

    print(
        f"xi << W                  : "
        f"{'PASS' if params.coherence_to_width_ratio < 0.1 else 'CHECK'}"
    )

    # --------------------------------------------------------
    # Progress.
    # --------------------------------------------------------

    progress = ProgressDisplay()

    print()
    print(
        "Running Clem-Berggren local "
        "corner analysis..."
    )

    analysis_start = time.perf_counter()

    result = analyze_clem_berggren(
        nodes_m=nodes,
        triangles=triangles,
        triangle_centers_m=triangle_centers,
        element_J_magnitude_A_per_m2=element_J,
        fem_current_A=fem_current,
        params=params,
        progress_callback=progress.callback,
    )

    progress.finish()

    analysis_time = (
        time.perf_counter()
        - analysis_start
    )

    # --------------------------------------------------------
    # Result.
    # --------------------------------------------------------

    print(
        format_result(
            result
        )
    )

    # --------------------------------------------------------
    # Corner statistics.
    # --------------------------------------------------------

    valid_corners = [
        c
        for c in result.corners
        if c.accepted
        and np.isfinite(
            c.critical_current_A
        )
    ]

    invalid_corners = [
        c
        for c in result.corners
        if not c.accepted
    ]

    print()
    print(
        "NUMERICAL SUMMARY"
    )
    print(
        "-----------------"
    )

    print(
        f"Nodes                    : "
        f"{len(nodes):,}"
    )

    print(
        f"Triangles                : "
        f"{len(triangles):,}"
    )

    print(
        f"Re-entrant corners       : "
        f"{len(result.corners):,}"
    )

    print(
        f"Valid corners            : "
        f"{len(valid_corners):,}"
    )

    print(
        f"Invalid corners          : "
        f"{len(invalid_corners):,}"
    )

    print(
        f"Analysis time            : "
        f"{analysis_time:.3f} s"
    )

    print()
    print(
        "LIMITING CORNER"
    )
    print(
        "---------------"
    )

    print(
        f"Ic                       : "
        f"{result.critical_current_A:.12e} A"
    )

    print(
        f"Ic                       : "
        f"{result.critical_current_A * 1e6:.6f} uA"
    )

    print(
        f"x                        : "
        f"{result.limiting_x_m * 1e6:.6f} um"
    )

    print(
        f"y                        : "
        f"{result.limiting_y_m * 1e6:.6f} um"
    )

    print(
        f"Interior angle           : "
        f"{result.limiting_angle_deg:.6f} deg"
    )

    print(
        f"K0 reference             : "
        f"{result.limiting_K0_reference_A_per_m_power:.6e} A/m"
    )

    print(
        f"K0 critical              : "
        f"{result.limiting_K0_critical_A_per_m_power:.6e} A/m"
    )

    # --------------------------------------------------------
    # Corner R² statistics.
    # --------------------------------------------------------

    if valid_corners:

        r2_values = np.array(
            [
                c.fit_r2
                for c in valid_corners
                if np.isfinite(c.fit_r2)
            ],
            dtype=float,
        )

        Ic_values = np.array(
            [
                c.critical_current_A
                for c in valid_corners
            ],
            dtype=float,
        )

        print()
        print(
            "VALID CORNER STATISTICS"
        )
        print(
            "-----------------------"
        )

        print(
            f"Minimum R²              : "
            f"{np.min(r2_values):.6f}"
        )

        print(
            f"Mean R²                 : "
            f"{np.mean(r2_values):.6f}"
        )

        print(
            f"Maximum R²              : "
            f"{np.max(r2_values):.6f}"
        )

        print(
            f"Minimum Ic              : "
            f"{np.min(Ic_values) * 1e6:.6f} uA"
        )

        print(
            f"Maximum Ic              : "
            f"{np.max(Ic_values) * 1e6:.6f} uA"
        )

    # --------------------------------------------------------
    # Save NPZ.
    # --------------------------------------------------------

    output_npz = Path(
        args.output_npz
    )

    output_npz.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if result.corners:

        corner_indices = np.array(
            [
                c.vertex_index
                for c in result.corners
            ],
            dtype=np.int64,
        )

        corner_x = np.array(
            [
                c.x_m
                for c in result.corners
            ],
            dtype=float,
        )

        corner_y = np.array(
            [
                c.y_m
                for c in result.corners
            ],
            dtype=float,
        )

        corner_angles = np.array(
            [
                c.interior_angle_rad
                for c in result.corners
            ],
            dtype=float,
        )

        corner_Ic = np.array(
            [
                c.critical_current_A
                for c in result.corners
            ],
            dtype=float,
        )

        corner_r2 = np.array(
            [
                c.fit_r2
                for c in result.corners
            ],
            dtype=float,
        )

        corner_accepted = np.array(
            [
                c.accepted
                for c in result.corners
            ],
            dtype=bool,
        )

    else:

        corner_indices = np.empty(
            0,
            dtype=np.int64,
        )

        corner_x = np.empty(
            0,
            dtype=float,
        )

        corner_y = np.empty(
            0,
            dtype=float,
        )

        corner_angles = np.empty(
            0,
            dtype=float,
        )

        corner_Ic = np.empty(
            0,
            dtype=float,
        )

        corner_r2 = np.empty(
            0,
            dtype=float,
        )

        corner_accepted = np.empty(
            0,
            dtype=bool,
        )

    np.savez_compressed(

        output_npz,

        # ----------------------------------------------------
        # Core result
        # ----------------------------------------------------

        critical_current_A=float(
            result.critical_current_A
        ),

        straight_strip_critical_current_A=float(
            result.straight_strip_critical_current_A
        ),

        straight_strip_critical_sheet_current_A_per_m=float(
            result.straight_strip_critical_sheet_current_A_per_m
        ),

        # ----------------------------------------------------
        # Physical scales
        # ----------------------------------------------------

        pearl_length_m=float(
            result.pearl_length_m
        ),

        penetration_depth_m=float(
            result.penetration_depth_m
        ),

        coherence_length_m=float(
            result.coherence_length_m
        ),

        wire_width_m=float(
            wire_width
        ),

        film_thickness_m=float(
            film_thickness
        ),

        temperature_k=float(
            args.temperature_k
        ),

        # ----------------------------------------------------
        # FEM provenance
        # ----------------------------------------------------

        fem_current_A=float(
            fem_current
        ),

        fem_nodes=int(
            len(nodes)
        ),

        fem_triangles=int(
            len(triangles)
        ),

        # ----------------------------------------------------
        # Limiting corner
        # ----------------------------------------------------

        limiting_x_m=float(
            result.limiting_x_m
        ),

        limiting_y_m=float(
            result.limiting_y_m
        ),

        limiting_angle_deg=float(
            result.limiting_angle_deg
        ),

        limiting_current_A=float(
            result.limiting_current_A
        ),

        limiting_K0_reference_A_per_m_power=float(
            result.limiting_K0_reference_A_per_m_power
        ),

        limiting_K0_critical_A_per_m_power=float(
            result.limiting_K0_critical_A_per_m_power
        ),

        # ----------------------------------------------------
        # Validity
        # ----------------------------------------------------

        width_to_pearl_ratio=float(
            result.width_to_pearl_ratio
        ),

        coherence_to_width_ratio=float(
            result.coherence_to_width_ratio
        ),

        thickness_to_lambda_ratio=float(
            result.thickness_to_lambda_ratio
        ),

        validity_w_over_lambda=bool(
            result.validity_w_over_lambda
        ),

        validity_xi_over_w=bool(
            result.validity_xi_over_w
        ),

        validity_d_over_lambda=bool(
            result.validity_d_over_lambda
        ),

        # ----------------------------------------------------
        # Pipeline provenance
        # ----------------------------------------------------

        material=str(
            args.material
        ),

        lambda_nm=float(
            lambda_nm
        ),

        xi_nm=float(
            xi_nm
        ),

        pipeline_jc_A_per_m2=(
            np.nan
            if args.jc is None
            else float(args.jc)
        ),

        # ----------------------------------------------------
        # Corner arrays
        # ----------------------------------------------------

        corner_vertex_indices=corner_indices,

        corner_x_m=corner_x,

        corner_y_m=corner_y,

        corner_angles_rad=corner_angles,

        corner_critical_currents_A=corner_Ic,

        corner_fit_R2=corner_r2,

        corner_accepted=corner_accepted,
    )

    print()
    print(
        "Clem-Berggren numerical result saved to:"
    )

    print(
        output_npz.resolve()
    )

    # --------------------------------------------------------
    # Heatmap
    # --------------------------------------------------------

    heatmap_ok = generate_heatmap(
        fem_data=fem_data,
        nodes=nodes,
        triangles=triangles,
        element_J=element_J,
        film_thickness=film_thickness,
        limiting_x=result.limiting_x_m,
        limiting_y=result.limiting_y_m,
        output_path=Path(
            args.output
        ),
    )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    print()
    print(
        "=" * 72
    )

    print(
        "STAGE 4 / CLEM-BERGGREN "
        "CRITICAL CURRENT : PASS"
    )

    print(
        "=" * 72
    )

    print(
        f"Ic = "
        f"{result.critical_current_A * 1e6:.6f} uA"
    )

    print(
        f"Valid corners = "
        f"{len(valid_corners)} / "
        f"{len(result.corners)}"
    )

    if not heatmap_ok:

        print(
            "WARNING: numerical analysis passed, "
            "but heatmap generation was skipped."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
        print(
            "\nClem-Berggren analysis interrupted."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print(
            "Clem-Berggren analysis FAILED:"
        )

        print(
            f"{type(exc).__name__}: {exc}"
        )

        raise