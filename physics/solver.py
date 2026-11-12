import math
import numpy as np

from errors import InvalidPhysicsInputError
from physics.constants import KINETIC_COEFF, HBAR, E_CHARGE, ELECTRON_MASS
from physics.base import QuantumSystem
from physics.numerics import integrate

POTENTIALS = {
    "finite_well": "Finite square well",
    "harmonic": "Harmonic oscillator",
    "double_well": "Double well",
    "custom": "Custom potential",
}


class HarmonicOscillator(QuantumSystem):
    def __init__(self, mass_ratio=1.0, curvature=1.0):
        self.mass = mass_ratio * ELECTRON_MASS
        self.curvature = curvature
        if self.mass <= 0 or self.curvature <= 0:
            raise InvalidPhysicsInputError("Mass and curvature must be positive.")

    def energy_levels(self, count=5):
        omega = np.sqrt(self.curvature * E_CHARGE * 1e18 / self.mass)
        return [float((n + 0.5) * HBAR * omega / E_CHARGE) for n in range(count)]


def potential_profile(x, kind, width, barrier, curvature, separation, amplitude):
    if kind == "finite_well":
        return np.where(np.abs(x) < width / 2, 0, barrier)
    if kind == "harmonic":
        return 0.5 * curvature * x ** 2
    if kind == "double_well":
        return barrier * ((x / separation) ** 2 - 1) ** 2
    if kind == "custom":
        edge = max(abs(x).max(), 0.0001)
        return 0.5 * curvature * x ** 2 + amplitude * np.sin(np.pi * x / edge)
    raise InvalidPhysicsInputError("Choose a listed potential.")


def solve_schrodinger(kind="finite_well", mass_ratio=1.0, x_max_nm=2.0,
                      grid_points=161, levels=4, width=1.0, barrier=0.6,
                      curvature=1.0, separation=0.6, amplitude=0.25):
    if kind not in POTENTIALS:
        raise InvalidPhysicsInputError("Choose a listed potential.")
    numbers = [mass_ratio, x_max_nm, width, barrier, curvature, separation, amplitude]
    if not all(math.isfinite(float(value)) for value in numbers):
        raise InvalidPhysicsInputError("Enter finite potential values.")
    if mass_ratio <= 0 or x_max_nm <= 0.2 or width <= 0 or separation <= 0 or barrier < 0 or curvature < 0:
        raise InvalidPhysicsInputError("Potential dimensions and mass must be positive.")
    if grid_points < 61 or grid_points > 301 or levels < 1 or levels > 8:
        raise InvalidPhysicsInputError("Grid or energy-level count is outside its allowed range.")

    full_x = np.linspace(-x_max_nm, x_max_nm, grid_points + 2)
    x = full_x[1:-1]
    dx = full_x[1] - full_x[0]
    potential = potential_profile(x, kind, width, barrier, curvature, separation, amplitude)

    kinetic = KINETIC_COEFF / mass_ratio
    diagonal = 2 * kinetic / dx ** 2 + potential
    side = np.full(len(x) - 1, -kinetic / dx ** 2)
    hamiltonian = np.diag(diagonal) + np.diag(side, 1) + np.diag(side, -1)
    energies, vectors = np.linalg.eigh(hamiltonian)
    energies = energies[:levels]
    wavefunctions = vectors[:, :levels].T / np.sqrt(dx)

    for wave in wavefunctions:
        peak = np.argmax(np.abs(wave))
        if wave[peak] < 0:
            wave *= -1

    normalizations = [integrate(wave ** 2, x) for wave in wavefunctions]
    return {
        "kind": kind,
        "x_nm": x.tolist(),
        "potential_eV": potential.tolist(),
        "energies_eV": energies.tolist(),
        "wavefunctions": wavefunctions.tolist(),
        "normalizations": normalizations,
        "grid_points": grid_points,
    }
