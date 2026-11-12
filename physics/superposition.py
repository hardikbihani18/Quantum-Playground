import math
import numpy as np

from errors import InvalidPhysicsInputError
from physics.box import ParticleInBox
from physics.constants import E_CHARGE, HBAR, ELECTRON_MASS
from physics.numerics import integrate


def evolve_superposition(length_nm=1.0, coefficients=None, time_fs=5.0,
                         mass_kg=ELECTRON_MASS, points=250):
    if coefficients is None:
        coefficients = [1.0, 0.6, 0.2]
    if not math.isfinite(float(time_fs)) or time_fs < 0:
        raise InvalidPhysicsInputError("Time cannot be negative.")

    values = np.array(coefficients, dtype=float)
    if len(values) == 0 or len(values) > 8 or not np.all(np.isfinite(values)):
        raise InvalidPhysicsInputError("Enter valid coefficients.")
    total = np.linalg.norm(values)
    if total == 0:
        raise InvalidPhysicsInputError("At least one coefficient must be non-zero.")
    values = values / total

    system = ParticleInBox(length_nm, mass_kg)
    x = np.linspace(0, length_nm, points)
    states = np.arange(1, len(values) + 1)
    wavefunctions = np.array([
        np.sqrt(2 / length_nm) * np.sin(n * np.pi * x / length_nm)
        for n in states
    ])
    energies = np.array([system.energy_eV(int(n)) for n in states])
    initial_wave = np.sum(values[:, None] * wavefunctions, axis=0)
    time_seconds = time_fs * 1e-15
    phases = np.exp(-1j * energies * E_CHARGE * time_seconds / HBAR)
    evolved_wave = np.sum((values * phases)[:, None] * wavefunctions, axis=0)

    initial_density = np.abs(initial_wave) ** 2
    final_density = np.abs(evolved_wave) ** 2
    level_list = []
    for n in states:
        level_list.append({"n": int(n), "energy_eV": system.energy_eV(int(n))})

    return {
        "x_nm": x.tolist(),
        "initial_density": initial_density.tolist(),
        "density": final_density.tolist(),
        "coefficients": values.tolist(),
        "levels": level_list,
        "time_fs": time_fs,
        "normalization": integrate(final_density, x),
    }
