# FILE: src/snspd/fem/__init__.py
# PURPOSE:
# Initializes the finite-element-method subsystem of the SNSPD simulator.
#
# The FEM subsystem provides:
#
#     - element-level mathematical operators
#     - global matrix assembly
#     - boundary-condition handling
#     - linear-system solution
#
# This subsystem is intentionally independent of SNSPD-specific physics.