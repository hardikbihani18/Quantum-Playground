import math
import numpy as np

from errors import InvalidPhysicsInputError
from physics.base import QuantumSystem
from physics.constants import HBAR, E_CHARGE, ELECTRON_MASS


class Barrier(QuantumSystem):
    def __init__(self, energy_eV=1.0, barrier_eV=3.0, mass_kg=ELECTRON_MASS):
        values = [energy_eV, barrier_eV, mass_kg]
        if not all(math.isfinite(float(value)) for value in values):
            raise InvalidPhysicsInputError("Enter finite energy and mass values.")
        if energy_eV <= 0 or barrier_eV <= energy_eV or mass_kg <= 0:
            raise InvalidPhysicsInputError("Use 0 < E < V0 and a positive mass.")
        self.energy_eV = float(energy_eV)
        self.barrier_eV = float(barrier_eV)
        self.mass_kg = float(mass_kg)

    def energy_levels(self, count=1):
        return [self.energy_eV]

    def transmission(self, width_nm=0.5):
        if not math.isfinite(float(width_nm)) or width_nm < 0:
            raise InvalidPhysicsInputError("Barrier width cannot be negative.")
        difference = (self.barrier_eV - self.energy_eV) * E_CHARGE
        kappa = math.sqrt(2 * self.mass_kg * difference) / HBAR
        value = kappa * width_nm * 1e-9
        if value > 350:
            return 0.0
        sinh_value = math.sinh(value)
        ratio = self.barrier_eV ** 2 * sinh_value ** 2
        ratio = ratio / (4 * self.energy_eV * (self.barrier_eV - self.energy_eV))
        return 1 / (1 + ratio)


def tunneling_curve(energy_eV=1.0, barrier_eV=3.0, width_nm=0.5,
                    mass_kg=ELECTRON_MASS, points=100):
    system = Barrier(energy_eV, barrier_eV, mass_kg)
    if not math.isfinite(float(width_nm)) or width_nm < 0:
        raise InvalidPhysicsInputError("Barrier width must be finite and non-negative.")
    widths = np.linspace(0, max(2, width_nm * 2), points)
    transmissions = [system.transmission(width) for width in widths]
    return {
        "energy_eV": energy_eV,
        "barrier_eV": barrier_eV,
        "width_nm": width_nm,
        "transmission": system.transmission(width_nm),
        "curve_width_nm": widths.tolist(),
        "curve_transmission": transmissions,
    }
