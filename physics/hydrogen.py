import math
from errors import InvalidPhysicsInputError
from physics.constants import RYDBERG, PLANCK, LIGHT_SPEED, E_CHARGE

SERIES = {"Lyman": 1, "Balmer": 2, "Paschen": 3}


def hydrogen_spectrum(series="Balmer", n_max=8):
    if series not in SERIES:
        raise InvalidPhysicsInputError("Choose Lyman, Balmer or Paschen.")
    lower = SERIES[series]
    if int(n_max) != n_max or n_max <= lower or n_max > 20:
        raise InvalidPhysicsInputError("Upper level must be greater than the final level.")

    lines = []
    for upper in range(lower + 1, n_max + 1):
        inverse_wavelength = RYDBERG * (1 / lower ** 2 - 1 / upper ** 2)
        wavelength = 1e9 / inverse_wavelength
        energy = PLANCK * LIGHT_SPEED / (wavelength * 1e-9) / E_CHARGE
        lines.append({
            "transition": f"{upper} to {lower}",
            "upper_n": upper,
            "lower_n": lower,
            "wavelength_nm": wavelength,
            "energy_eV": energy,
        })
    return {"series": series, "lower_n": lower, "lines": lines}
