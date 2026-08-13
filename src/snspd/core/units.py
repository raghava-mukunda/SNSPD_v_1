# FILE: src/snspd/core/units.py
# PURPOSE:
# Provides the centralized physical-unit system for the SNSPD simulator.
#
# User-facing quantities may use nm, um, GHz, K, uA, etc.
# Numerical solvers will convert quantities to SI when required.

from pint import UnitRegistry


# Single unit registry for the entire SNSPD application.
ureg = UnitRegistry()


# Common length units.
nm = ureg.nanometer
um = ureg.micrometer
mm = ureg.millimeter


# Temperature.
K = ureg.kelvin


# Current.
uA = ureg.microampere
mA = ureg.milliampere
A = ureg.ampere


# Frequency.
Hz = ureg.hertz
kHz = ureg.kilohertz
MHz = ureg.megahertz
GHz = ureg.gigahertz


# Voltage.
V = ureg.volt
mV = ureg.millivolt


# Resistance and inductance.
ohm = ureg.ohm
pH = ureg.picohenry
nH = ureg.nanohenry


def to_si(quantity):
    """
    Convert a Pint quantity into SI base units.

    Parameters
    ----------
    quantity:
        Pint Quantity.

    Returns
    -------
    pint.Quantity
        Quantity converted to SI base units.
    """

    return quantity.to_base_units()