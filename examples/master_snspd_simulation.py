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
    5. Critical-current Phase A + B + C
        |
        v
    Final geometry, J-map, Ic-map, and numerical results

CORE INPUTS
-----------
    input SVG
    nanowire width

The remaining physical parameters have explicit defaults but can
be overridden from the command line.

IMPORTANT
---------
The preprocessing stage currently uses the overall device width
(--device-width-um) to establish physical SVG scale. This is NOT
the nanowire width.

The nanowire width is passed independently to the electrical FEM
stage.

Default physical assumptions:
    nanowire width   = 100 nm
    film thickness   = 10 nm
    device width     = 50 um
    Jc               = 2.16e10 A/m^2
    material         = NbTiN
    temperature      = 3.1 K

Example
-------
    python3 examples/master_snspd_simulation.py \
        examples/meander.svg \
        --wire-width-nm 100

Override thickness and critical-current model:
    python3 examples/master_snspd_simulation.py \
        examples/meander.svg \
        --wire-width-nm 100 \
        --thickness-nm 10 \
        --jc 2.16e10 \
        --material NbTiN \
        --temperature-k 3.1

Override the physical overall device width used by preprocessing:
    --device-width-um 50
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys


# ============================================================
# DEFAULTS
# ============================================================

DEFAULT_DEVICE_WIDTH_UM = 50.0
DEFAULT_THICKNESS_NM = 10.0
DEFAULT_JC = 2.16e10
DEFAULT_MATERIAL = "NbTiN"
DEFAULT_TEMPERATURE_K = 3.1

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
            "current-crowding -> critical-current pipeline."
        )
    )

    parser.add_argument(
        "input_svg",
        help="Input SNSPD SVG geometry.",
    )

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
            f"Superconducting film thickness in nm "
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

    parser.add_argument(
        "--jc",
        type=float,
        default=DEFAULT_JC,
        help=(
            f"Critical current density in A/m^2 "
            f"(default: {DEFAULT_JC:.3e})."
        ),
    )

    parser.add_argument(
        "--material",
        default=DEFAULT_MATERIAL,
        help=f"Material label (default: {DEFAULT_MATERIAL}).",
    )

    parser.add_argument(
        "--temperature-k",
        type=float,
        default=DEFAULT_TEMPERATURE_K,
        help=(
            f"Operating temperature in K "
            f"(default: {DEFAULT_TEMPERATURE_K:g})."
        ),
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_RENDER_SCALE,
        help=f"SVG rasterization scale (default: {DEFAULT_RENDER_SCALE:g}).",
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=DEFAULT_MIN_AREA,
        help=f"Preprocessor minimum component area (default: {DEFAULT_MIN_AREA}).",
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
        help=f"Polygon simplification tolerance in pixels (default: {DEFAULT_SIMPLIFY:g}).",
    )

    parser.add_argument(
        "--skip-geometry-plot",
        action="store_true",
        help=(
            "Run geometry validation/metrics but do not require its "
            "interactive plot to remain open."
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
    print("$ " + " ".join(command))
    print()

    env = os.environ.copy()

    # Prevent matplotlib from opening an interactive GUI when the
    # master pipeline is running geometry diagnostics.
    #
    # The electrical and critical-current scripts explicitly save
    # their figures. The geometry stage is therefore safest in a
    # non-interactive backend during a fully automated run.
    env.setdefault("MPLBACKEND", "Agg")

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
            print("Continuing because this stage was marked optional.")
            return

        raise RuntimeError(message)


# ============================================================
# MAIN PIPELINE
# ============================================================

def main() -> None:
    args = parse_arguments()

    input_svg = Path(args.input_svg).expanduser().resolve()

    if not input_svg.exists():
        raise FileNotFoundError(
            f"Input SVG does not exist:\n{input_svg}"
        )

    if input_svg.suffix.lower() != ".svg":
        raise ValueError(
            f"Expected an SVG file, got: {input_svg}"
        )

    if args.wire_width_nm <= 0:
        raise ValueError("--wire-width-nm must be positive.")

    if args.thickness_nm <= 0:
        raise ValueError("--thickness-nm must be positive.")

    if args.device_width_um <= 0:
        raise ValueError("--device-width-um must be positive.")

    if args.jc <= 0:
        raise ValueError("--jc must be positive.")

    if args.temperature_k <= 0:
        raise ValueError("--temperature-k must be positive.")

    # ------------------------------------------------------------
    # PROJECT ROOT
    # ------------------------------------------------------------

    # This master file is intended to live in:
    #
    #     <repo>/examples/master_snspd_simulation.py
    #
    # Therefore:
    #
    #     repo_root = examples/..
    #
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent

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
                f"Required pipeline script not found:\n{script}"
            )

    # ------------------------------------------------------------
    # OUTPUT DIRECTORIES
    # ------------------------------------------------------------

    results_dir = repo_root / "results"
    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed_svg = (
        input_svg.parent
        / f"{input_svg.stem}_processed.svg"
    )

    preview_png = (
        results_dir
        / f"{input_svg.stem}_preprocess_preview.png"
    )

    fem_npz = (
        results_dir
        / "current_crowding_fem.npz"
    )

    critical_npz = (
        results_dir
        / "critical_current_field.npz"
    )

    critical_png = (
        results_dir
        / "critical_current_heatmap.png"
    )

    # Current-crowding script already uses this default output name
    # unless explicitly overridden.
    current_density_png = (
        results_dir
        / "current_density_heatmap.png"
    )

    # ------------------------------------------------------------
    # HEADER
    # ------------------------------------------------------------

    print()
    print("=" * 72)
    print("SNSPD END-TO-END MASTER SIMULATION")
    print("=" * 72)
    print()
    print(f"Input SVG                 : {input_svg}")
    print(f"Nanowire width            : {args.wire_width_nm:.6f} nm")
    print(f"Film thickness            : {args.thickness_nm:.6f} nm")
    print(f"Device width for scaling  : {args.device_width_um:.6f} um")
    print(f"Material                  : {args.material}")
    print(f"Jc                        : {args.jc:.6e} A/m²")
    print(f"Temperature               : {args.temperature_k:.6f} K")
    print()
    print("Pipeline:")
    print("  1. SVG preprocessing")
    print("  2. Geometry validation / metrics")
    print("  3. Stationary electrical FEM")
    print("  4. Current-density / crowding analysis")
    print("  5. Critical-current Phase A + B + C")
    print()

    # ------------------------------------------------------------
    # STAGE 1: PREPROCESS
    # ------------------------------------------------------------

    run_stage(
        "STAGE 1 / SVG PREPROCESSING",
        [
            sys.executable,
            str(preprocess_script),
            str(input_svg),
            "-o",
            str(processed_svg),
            "--preview",
            str(preview_png),
            "--scale",
            str(args.scale),
            "--width-um",
            str(args.device_width_um),
            "--min-area",
            str(args.min_area),
            "--min-component-fraction",
            str(args.min_component_fraction),
            "--simplify",
            str(args.simplify),
        ],
        repo_root,
    )

    if not processed_svg.exists():
        raise RuntimeError(
            f"Preprocessing reported success, but output was not found:\n"
            f"{processed_svg}"
        )

    # ------------------------------------------------------------
    # STAGE 2: GEOMETRY
    # ------------------------------------------------------------

    # run_geometry.py is a diagnostic/visualization stage. We run it
    # with a non-interactive matplotlib backend so that the master
    # pipeline can continue automatically.
    run_stage(
        "STAGE 2 / GEOMETRY VALIDATION + METRICS",
        [
            sys.executable,
            str(geometry_script),
            str(processed_svg),
        ],
        repo_root,
        allow_failure=args.keep_going,
    )

    # ------------------------------------------------------------
    # STAGE 3: CURRENT CROWDING FEM
    # ------------------------------------------------------------

    run_stage(
        "STAGE 3 / STATIONARY ELECTRICAL FEM + CURRENT CROWDING",
        [
            sys.executable,
            str(crowding_script),
            str(processed_svg),
            "--wire-width-nm",
            str(args.wire_width_nm),
            "--thickness-nm",
            str(args.thickness_nm),
            "--fem-output",
            str(fem_npz),
        ],
        repo_root,
    )

    if not fem_npz.exists():
        raise RuntimeError(
            "Current-crowding stage completed but FEM result was not found:\n"
            f"{fem_npz}"
        )

    # ------------------------------------------------------------
    # STAGE 4: CRITICAL CURRENT
    # ------------------------------------------------------------

    run_stage(
        "STAGE 4 / CRITICAL CURRENT PHASE A + B + C",
        [
            sys.executable,
            str(critical_script),
            str(fem_npz),
            "--jc",
            str(args.jc),
            "--material",
            str(args.material),
            "--temperature-k",
            str(args.temperature_k),
            "--output",
            str(critical_png),
            "--output-npz",
            str(critical_npz),
        ],
        repo_root,
    )

    if not critical_npz.exists():
        raise RuntimeError(
            "Critical-current stage completed but field result was not found:\n"
            f"{critical_npz}"
        )

    if not critical_png.exists():
        raise RuntimeError(
            "Critical-current stage completed but heatmap was not found:\n"
            f"{critical_png}"
        )

    # ------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------

    print()
    print("=" * 72)
    print("SNSPD MASTER SIMULATION COMPLETE")
    print("=" * 72)
    print()
    print("INPUT")
    print("-----")
    print(f"SVG                       : {input_svg}")
    print(f"Wire width                : {args.wire_width_nm:.6f} nm")
    print(f"Thickness                 : {args.thickness_nm:.6f} nm")
    print(f"Jc                        : {args.jc:.6e} A/m²")
    print(f"Material                  : {args.material}")
    print(f"Temperature               : {args.temperature_k:.6f} K")
    print()
    print("OUTPUTS")
    print("-------")
    print(f"Processed SVG             : {processed_svg}")
    print(f"Preprocess preview        : {preview_png}")
    print(f"FEM result                : {fem_npz}")
    print(f"Current-density heatmap   : {current_density_png}")
    print(f"Critical-current field    : {critical_npz}")
    print(f"Critical-current heatmap  : {critical_png}")
    print()
    print("The numerical results above are the authoritative outputs")
    print("from the individual FEM and critical-current stages.")
    print()


if __name__ == "__main__":
    main()
