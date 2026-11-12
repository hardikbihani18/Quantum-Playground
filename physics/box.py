import math
import numpy as np

from errors import InvalidPhysicsInputError, InvalidQuantumNumberError
from physics.base import QuantumSystem
from physics.constants import PLANCK, E_CHARGE, ELECTRON_MASS
from physics.numerics import integrate


class ParticleInBox(QuantumSystem):
    def __init__(self, length_nm=1.0, mass_kg=ELECTRON_MASS):
        if not math.isfinite(length_nm) or not math.isfinite(mass_kg) or length_nm <= 0 or mass_kg <= 0:
            raise InvalidPhysicsInputError("Length and mass must be positive.")
        self.length_nm = float(length_nm)
        self.length_m = self.length_nm * 1e-9
        self.mass_kg = float(mass_kg)

    def energy_eV(self, n):
        if int(n) != n or n < 1:
            raise InvalidQuantumNumberError("n must be a positive integer.")
        energy_j = n ** 2 * PLANCK ** 2 / (8 * self.mass_kg * self.length_m ** 2)
        return energy_j / E_CHARGE

    def energy_levels(self, count=5):
        return [self.energy_eV(n) for n in range(1, count + 1)]

    def wavefunction(self, n=1, points=300):
        if int(n) != n or n < 1:
            raise InvalidQuantumNumberError("n must be a positive integer.")
        x = np.linspace(0, self.length_nm, points)
        psi = np.sqrt(2 / self.length_nm) * np.sin(n * np.pi * x / self.length_nm)
        density = psi ** 2
        normalization = integrate(density, x)
        return {
            "x_nm": x.tolist(),
            "psi": psi.tolist(),
            "density": density.tolist(),
            "energy_eV": self.energy_eV(n),
            "normalization": float(normalization),
            "n": int(n),
        }


def get_mass(mass_ratio=1.0):
    if not math.isfinite(float(mass_ratio)) or mass_ratio <= 0:
        raise InvalidPhysicsInputError("Mass ratio must be positive.")
    return float(mass_ratio) * ELECTRON_MASS


def level_generator(system, count=6):
    for n in range(1, count + 1):
        yield {"n": n, "energy_eV": system.energy_eV(n)}
