import numpy as np


def integrate(values, positions):
    values = np.asarray(values, dtype=float)
    positions = np.asarray(positions, dtype=float)
    return float(np.sum((values[1:] + values[:-1]) * np.diff(positions) / 2))
