import math
import numpy as np

from errors import InvalidPhysicsInputError
from physics.constants import PLANCK, LIGHT_SPEED, E_CHARGE

METALS = {
    "Cesium": 2.14,
    "Potassium": 2.30,
    "Sodium": 2.28,
    "Zinc": 4.31,
    "Copper": 4.70,
    "Platinum": 6.35,
}


def calculate_photoelectric(metal="Cesium", wavelength_nm=400.0):
    if metal not in METALS:
        raise InvalidPhysicsInputError("Choose a metal from the list.")
    if not math.isfinite(float(wavelength_nm)) or wavelength_nm <= 0:
        raise InvalidPhysicsInputError("Wavelength must be positive.")

    work_function = METALS[metal]
    frequency = LIGHT_SPEED / (wavelength_nm * 1e-9)
    photon_energy = PLANCK * frequency / E_CHARGE
    kinetic_energy = max(0, photon_energy - work_function)
    threshold_frequency = work_function * E_CHARGE / PLANCK
    threshold_wavelength = PLANCK * LIGHT_SPEED / (work_function * E_CHARGE) * 1e9

    graph_frequency = np.linspace(0, threshold_frequency * 2.3, 180) / 1e12
    graph_energy = np.maximum(0, graph_frequency * 1e12 * PLANCK / E_CHARGE - work_function)

    return {
        "metal": metal,
        "work_function_eV": work_function,
        "wavelength_nm": wavelength_nm,
        "frequency_THz": frequency / 1e12,
        "photon_energy_eV": photon_energy,
        "emission": photon_energy > work_function,
        "kinetic_energy_eV": kinetic_energy,
        "stopping_potential_V": kinetic_energy,
        "threshold_frequency_THz": threshold_frequency / 1e12,
        "threshold_wavelength_nm": threshold_wavelength,
        "graph_frequency_THz": graph_frequency.tolist(),
        "graph_kinetic_energy_eV": graph_energy.tolist(),
    }
