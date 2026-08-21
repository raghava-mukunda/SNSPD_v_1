#!/usr/bin/env python3
"""
SNSPD END-TO-END MASTER SIMULATION
===================================

Run the complete geometry -> FEM -> critical-current pipeline:

    Input SVG
        |
        v
    1. SVG preprocessing
        |
        v
    2. Geometry import / validation / metrics
        |
        v
    3. Stationary electrical FEM
        |
        v
    4. Current-density / current-crowding analysis
        |
        v
    5. Clem-Berggren critical-current analysis
        |
        v
    Final geometry, J-map, Ic-map, and numerical results

CORE INPUTS
-----------
    input SVG
    nanowire width

SUPERCONDUCTING INPUTS
----------------------
    film thickness
    penetration depth lambda
    coherence length xi
    operating temperature
    material label

IMPORTANT
---------
The preprocessing stage uses the overall device width
(--device-width-um) to establish physical SVG scale.

This is NOT the nanowire width.

The nanowire width is passed independently to the
electrical FEM stage.

CRITICAL CURRENT MODEL
----------------------
The critical-current stage uses the Clem-Berggren
vortex-nucleation framework.

The old phenomenological Jc/C_J scaling is NOT used
by Stage 5.

Clem-Berggren inputs:

    lambda
    xi
    temperature
    material
    FEM current-density field

DEFAULTS
--------
    nanowire width       = 100 nm
    film thickness       = 10 nm
    device width         = 50 um
    lambda               = 450 nm
    xi                   = 5 nm
    material             = NbTiN
    temperature          = 3.1 K

Example
-------
    python3 examples/master_snspd_simulation.py \
        examples/meander.svg \
        --wire-width-nm 100

Example with explicit superconducting parameters
-------------------------------------------------
    python3 examples/master_snspd_simulation.py \
        examples/meander.svg \
        --wire-width-nm 100 \
        --thickness-nm 10 \
        --lambda-nm 450 \
        --xi-nm 5 \
        --material NbTiN \
        --temperature-k 3.1

Override the physical overall device width used by preprocessing
---------------------------------------------------------------
    --device-width-um 50
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_DEVICE_WIDTH_UM = 50.0

DEFAULT_THICKNESS_NM = 10.0

DEFAULT_LAMBDA_NM = 450.0

DEFAULT_XI_NM = 5.0

DEFAULT_MATERIAL = "NbTiN"

DEFAULT_TEMPERATURE_K = 3.1


# Legacy parameter retained only for backwards-compatible
# command-line parsing. It is NOT used by the Clem-Berggren
# critical-current stage.
DEFAULT_JC = 2.16e10


DEFAULT_RENDER_SCALE = 3.0

DEFAULT_MIN_AREA = 100

DEFAULT_MIN_COMPONENT_FRACTION = 0.01

DEFAULT_SIMPLIFY = 1.5


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Run the complete SNSPD SVG -> geometry -> FEM -> "
            "current-crowding -> Clem-Berggren critical-current "
            "pipeline."
        )
    )

    # --------------------------------------------------------
    # INPUT SVG
    # --------------------------------------------------------

    parser.add_argument(
        "input_svg",
        help="Input SNSPD SVG geometry.",
    )

    # --------------------------------------------------------
    # DEVICE GEOMETRY
    # --------------------------------------------------------

    parser.add_argument(
        "--wire-width-nm",
        type=float,
        required=True,
        help="Nanowire width in nm.",
    )

    parser.add_argument(
        "--thickness-nm",
        type=float,
        default=DEFAULT_THICKNESS_NM,
        help=(
            "Superconducting film thickness in nm "
            f"(default: {DEFAULT_THICKNESS_NM:g})."
        ),
    )

    parser.add_argument(
        "--device-width-um",
        type=float,
        default=DEFAULT_DEVICE_WIDTH_UM,
        help=(
            "Physical overall geometry width used by the SVG "
            f"preprocessor (default: {DEFAULT_DEVICE_WIDTH_UM:g} um)."
        ),
    )

    # --------------------------------------------------------
    # CLEM-BERGGREN PARAMETERS
    # --------------------------------------------------------

    parser.add_argument(
        "--lambda-nm",
        type=float,
        default=DEFAULT_LAMBDA_NM,
        help=(
            "London penetration depth lambda in nm "
            f"(default: {DEFAULT_LAMBDA_NM:g})."
        ),
    )

    parser.add_argument(
        "--xi-nm",
        type=float,
        default=DEFAULT_XI_NM,
        help=(
            "Superconducting coherence length xi in nm "
            f"(default: {DEFAULT_XI_NM:g})."
        ),
    )

    parser.add_argument(
        "--material",
        default=DEFAULT_MATERIAL,
        help=(
            "Material label "
            f"(default: {DEFAULT_MATERIAL})."
        ),
    )

    parser.add_argument(
        "--temperature-k",
        type=float,
        default=DEFAULT_TEMPERATURE_K,
        help=(
            "Operating temperature in K "
            f"(default: {DEFAULT_TEMPERATURE_K:g})."
        ),
    )

    # --------------------------------------------------------
    # LEGACY JC
    # --------------------------------------------------------
    #
    # Kept so old scripts/commands do not immediately break.
    #
    # IMPORTANT:
    # This parameter is NOT passed to the current
    # Clem-Berggren critical-current solver.
    #

    parser.add_argument(
        "--jc",
        type=float,
        default=DEFAULT_JC,
        help=(
            "Legacy critical-current density parameter. "
            "Retained for compatibility only and NOT used "
            "by the Clem-Berggren model."
        ),
    )

    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    parser.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_RENDER_SCALE,
        help=(
            "SVG rasterization scale "
            f"(default: {DEFAULT_RENDER_SCALE:g})."
        ),
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=DEFAULT_MIN_AREA,
        help=(
            "Preprocessor minimum component area "
            f"(default: {DEFAULT_MIN_AREA})."
        ),
    )

    parser.add_argument(
        "--min-component-fraction",
        type=float,
        default=DEFAULT_MIN_COMPONENT_FRACTION,
        help=(
            "Preprocessor minimum connected-component fraction "
            f"(default: {DEFAULT_MIN_COMPONENT_FRACTION:g})."
        ),
    )

    parser.add_argument(
        "--simplify",
        type=float,
        default=DEFAULT_SIMPLIFY,
        help=(
            "Polygon simplification tolerance in pixels "
            f"(default: {DEFAULT_SIMPLIFY:g})."
        ),
    )

    # --------------------------------------------------------
    # PIPELINE OPTIONS
    # --------------------------------------------------------

    parser.add_argument(
        "--skip-geometry-plot",
        action="store_true",
        help=(
            "Run geometry validation/metrics but do not require "
            "its interactive plot to remain open."
        ),
    )

    parser.add_argument(
        "--keep-going",
        action="store_true",
        help=(
            "Attempt later stages even if the geometry-plot stage "
            "returns a non-zero status. FEM/current-crowding failures "
            "still stop the pipeline."
        ),
    )

    return parser.parse_args()


# ============================================================
# COMMAND RUNNER
# ============================================================

def run_stage(
    title: str,
    command: list[str],
    cwd: Path,
    *,
    allow_failure: bool = False,
) -> None:

    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
    print()

    print(
        "$ "
        + " ".join(
            command
        )
    )

    print()

    env = os.environ.copy()

    # Prevent matplotlib from opening an interactive GUI
    # when the master pipeline is running automatically.
    env.setdefault(
        "MPLBACKEND",
        "Agg",
    )

    result = subprocess.run(
        command,
        cwd=str(cwd),
        env=env,
    )

    if result.returncode != 0:

        message = (
            f"\nStage failed: {title}\n"
            f"Return code: {result.returncode}\n"
        )

        if allow_failure:

            print(message)

            print(
                "Continuing because this stage "
                "was marked optional."
            )

            return

        raise RuntimeError(
            message
        )


# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> None:

    args = parse_arguments()

    # ========================================================
    # INPUT VALIDATION
    # ========================================================

    input_svg = (
        Path(
            args.input_svg
        )
        .expanduser()
        .resolve()
    )

    if not input_svg.exists():

        raise FileNotFoundError(
            f"Input SVG does not exist:\n"
            f"{input_svg}"
        )

    if input_svg.suffix.lower() != ".svg":

        raise ValueError(
            f"Expected an SVG file, got: "
            f"{input_svg}"
        )

    if args.wire_width_nm <= 0:

        raise ValueError(
            "--wire-width-nm must be positive."
        )

    if args.thickness_nm <= 0:

        raise ValueError(
            "--thickness-nm must be positive."
        )

    if args.device_width_um <= 0:

        raise ValueError(
            "--device-width-um must be positive."
        )

    if args.lambda_nm <= 0:

        raise ValueError(
            "--lambda-nm must be positive."
        )

    if args.xi_nm <= 0:

        raise ValueError(
            "--xi-nm must be positive."
        )

    if args.temperature_k <= 0:

        raise ValueError(
            "--temperature-k must be positive."
        )

    # Legacy Jc validation only.
    if args.jc <= 0:

        raise ValueError(
            "--jc must be positive."
        )

    # ========================================================
    # PROJECT ROOT
    # ========================================================

    # This master file is intended to live in:
    #
    #     <repo>/examples/master_snspd_simulation.py
    #
    # Therefore:
    #
    #     repo_root = examples/..
    #

    script_dir = (
        Path(__file__)
        .resolve()
        .parent
    )

    repo_root = (
        script_dir.parent
    )

    # ========================================================
    # PIPELINE SCRIPTS
    # ========================================================

    preprocess_script = (
        script_dir
        / "geometry"
        / "preprocess_svg.py"
    )

    geometry_script = (
        script_dir
        / "run_geometry.py"
    )

    crowding_script = (
        script_dir
        / "electrical"
        / "analyze_current_crowding.py"
    )

    critical_script = (
        script_dir
        / "electrical"
        / "analyze_critical_current.py"
    )

    required_scripts = [
        preprocess_script,
        geometry_script,
        crowding_script,
        critical_script,
    ]

    for script in required_scripts:

        if not script.exists():

            raise FileNotFoundError(
                "Required pipeline script not found:\n"
                f"{script}"
            )

    # ========================================================
    # OUTPUT DIRECTORIES
    # ========================================================

    results_dir = (
        repo_root
        / "results"
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Processed SVG
    # --------------------------------------------------------

    processed_svg = (
        input_svg.parent
        / f"{input_svg.stem}_processed.svg"
    )

    # --------------------------------------------------------
    # Preprocessing preview
    # --------------------------------------------------------

    preview_png = (
        results_dir
        / f"{input_svg.stem}_preprocess_preview.png"
    )

    # --------------------------------------------------------
    # FEM result
    # --------------------------------------------------------

    fem_npz = (
        results_dir
        / "current_crowding_fem.npz"
    )

    # --------------------------------------------------------
    # Standardized master critical-current output
    # --------------------------------------------------------

    critical_npz = (
        results_dir
        / "critical_current_field.npz"
    )

    critical_png = (
        results_dir
        / "critical_current_heatmap.png"
    )

    # --------------------------------------------------------
    # Authoritative output produced by current
    # Clem-Berggren implementation
    # --------------------------------------------------------

    clem_berggren_npz = (
        results_dir
        / "critical_current_clem_berggren.npz"
    )

    # --------------------------------------------------------
    # Current-density heatmap
    # --------------------------------------------------------

    current_density_png = (
        results_dir
        / "current_density_heatmap.png"
    )

    # ========================================================
    # HEADER
    # ========================================================

    print()

    print(
        "=" * 72
    )

    print(
        "SNSPD END-TO-END MASTER SIMULATION"
    )

    print(
        "=" * 72
    )

    print()

    print(
        f"Input SVG                 : "
        f"{input_svg}"
    )

    print(
        f"Nanowire width            : "
        f"{args.wire_width_nm:.6f} nm"
    )

    print(
        f"Film thickness            : "
        f"{args.thickness_nm:.6f} nm"
    )

    print(
        f"Device width for scaling  : "
        f"{args.device_width_um:.6f} um"
    )

    print(
        f"Material                  : "
        f"{args.material}"
    )

    print(
        f"Lambda                    : "
        f"{args.lambda_nm:.6f} nm"
    )

    print(
        f"Xi                        : "
        f"{args.xi_nm:.6f} nm"
    )

    print(
        f"Temperature               : "
        f"{args.temperature_k:.6f} K"
    )

    print()

    print(
        "Critical-current model    : "
        "Clem-Berggren"
    )

    print()

    print(
        "Pipeline:"
    )

    print(
        "  1. SVG preprocessing"
    )

    print(
        "  2. Geometry validation / metrics"
    )

    print(
        "  3. Stationary electrical FEM"
    )

    print(
        "  4. Current-density / crowding analysis"
    )

    print(
        "  5. Clem-Berggren critical-current analysis"
    )

    print()

    # ========================================================
    # STAGE 1
    # SVG PREPROCESSING
    # ========================================================

    run_stage(
        "STAGE 1 / SVG PREPROCESSING",
        [
            sys.executable,

            str(
                preprocess_script
            ),

            str(
                input_svg
            ),

            "-o",

            str(
                processed_svg
            ),

            "--preview",

            str(
                preview_png
            ),

            "--scale",

            str(
                args.scale
            ),

            "--width-um",

            str(
                args.device_width_um
            ),

            "--min-area",

            str(
                args.min_area
            ),

            "--min-component-fraction",

            str(
                args.min_component_fraction
            ),

            "--simplify",

            str(
                args.simplify
            ),
        ],

        repo_root,
    )

    if not processed_svg.exists():

        raise RuntimeError(
            "Preprocessing reported success, "
            "but output was not found:\n"
            f"{processed_svg}"
        )

    # ========================================================
    # STAGE 2
    # GEOMETRY
    # ========================================================

    run_stage(
        "STAGE 2 / GEOMETRY VALIDATION + METRICS",
        [
            sys.executable,

            str(
                geometry_script
            ),

            str(
                processed_svg
            ),
        ],

        repo_root,

        allow_failure=(
            args.keep_going
        ),
    )

    # ========================================================
    # STAGE 3
    # STATIONARY ELECTRICAL FEM
    # ========================================================

    run_stage(
        "STAGE 3 / STATIONARY ELECTRICAL FEM + CURRENT CROWDING",
        [
            sys.executable,

            str(
                crowding_script
            ),

            str(
                processed_svg
            ),

            "--wire-width-nm",

            str(
                args.wire_width_nm
            ),

            "--thickness-nm",

            str(
                args.thickness_nm
            ),

            "--fem-output",

            str(
                fem_npz
            ),
        ],

        repo_root,
    )

    if not fem_npz.exists():

        raise RuntimeError(
            "Current-crowding stage completed "
            "but FEM result was not found:\n"
            f"{fem_npz}"
        )

    # ========================================================
    # STAGE 4
    # CLEM-BERGGREN CRITICAL CURRENT
    # ========================================================

    #
    # IMPORTANT:
    #
    # The old implementation expected:
    #
    #     --jc
    #     --output-npz
    #
    # Those arguments belong to the previous phenomenological
    # critical-current model.
    #
    # The current implementation expects:
    #
    #     --lambda-nm
    #     --xi-nm
    #     --temperature-k
    #     --material
    #     --output
    #
    # The current implementation writes its authoritative
    # numerical NPZ as:
    #
    #     results/critical_current_clem_berggren.npz
    #
    # We therefore normalize that output to:
    #
    #     results/critical_current_field.npz
    #
    # for the master pipeline.
    #

    # Remove stale output so that a failed/new run cannot
    # accidentally be interpreted as a fresh result.

    if clem_berggren_npz.exists():

        clem_berggren_npz.unlink()

    if critical_npz.exists():

        critical_npz.unlink()

    run_stage(
        "STAGE 4 / CLEM-BERGGREN CRITICAL CURRENT",
        [
            sys.executable,

            str(
                critical_script
            ),

            str(
                fem_npz
            ),

            "--lambda-nm",

            str(
                args.lambda_nm
            ),

            "--xi-nm",

            str(
                args.xi_nm
            ),

            "--temperature-k",

            str(
                args.temperature_k
            ),

            "--material",

            str(
                args.material
            ),

            "--output",

            str(
                critical_png
            ),
        ],

        repo_root,
    )

    # ========================================================
    # VALIDATE CLEM-BERGGREN NUMERICAL OUTPUT
    # ========================================================

    if not clem_berggren_npz.exists():

        raise RuntimeError(
            "Clem-Berggren stage completed but its "
            "authoritative result was not found:\n"
            f"{clem_berggren_npz}"
        )

    # ========================================================
    # NORMALIZE OUTPUT NAME
    # ========================================================

    shutil.copy2(
        clem_berggren_npz,
        critical_npz,
    )

    print()

    print(
        "Clem-Berggren numerical result copied to:"
    )

    print(
        critical_npz
    )

    # ========================================================
    # VALIDATE HEATMAP
    # ========================================================

    if not critical_png.exists():

        raise RuntimeError(
            "Clem-Berggren stage completed but "
            "critical-current heatmap was not found:\n"
            f"{critical_png}"
        )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print()

    print(
        "=" * 72
    )

    print(
        "SNSPD MASTER SIMULATION COMPLETE"
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
        f"SVG                       : "
        f"{input_svg}"
    )

    print(
        f"Wire width                : "
        f"{args.wire_width_nm:.6f} nm"
    )

    print(
        f"Thickness                 : "
        f"{args.thickness_nm:.6f} nm"
    )

    print(
        f"Material                  : "
        f"{args.material}"
    )

    print(
        f"Lambda                    : "
        f"{args.lambda_nm:.6f} nm"
    )

    print(
        f"Xi                        : "
        f"{args.xi_nm:.6f} nm"
    )

    print(
        f"Temperature               : "
        f"{args.temperature_k:.6f} K"
    )

    print()

    print(
        "CRITICAL CURRENT MODEL"
    )

    print(
        "----------------------"
    )

    print(
        "Clem-Berggren vortex-nucleation framework"
    )

    print()

    print(
        "OUTPUTS"
    )

    print(
        "-------"
    )

    print(
        f"Processed SVG             : "
        f"{processed_svg}"
    )

    print(
        f"Preprocess preview        : "
        f"{preview_png}"
    )

    print(
        f"FEM result                : "
        f"{fem_npz}"
    )

    print(
        f"Current-density heatmap   : "
        f"{current_density_png}"
    )

    print(
        f"Clem-Berggren result     : "
        f"{clem_berggren_npz}"
    )

    print(
        f"Critical-current field    : "
        f"{critical_npz}"
    )

    print(
        f"Critical-current heatmap  : "
        f"{critical_png}"
    )

    print()

    print(
        "The numerical results above are the authoritative "
        "outputs from the individual FEM and critical-current stages."
    )

    print()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()