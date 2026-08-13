# FILE: src/snspd/physics/materials/__init__.py
# PURPOSE:
# Material-model package for the SNSPD physics engine.
#
# Material models are deliberately separated from geometry
# and FEM code.
#
# A material model is responsible for providing constitutive
# quantities such as:
#
#     epsilon
#     mu
#     conductivity
#     superconducting gap
#     penetration depth
#     critical temperature
#     thermal properties
#     quasiparticle properties
#
# Numerical values must be traceable to either:
#
#     1. a fundamental physical model,
#     2. a validated material database,
#     3. or explicitly supplied experimental data.