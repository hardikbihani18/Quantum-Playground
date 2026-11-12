import unittest

from physics.box import ParticleInBox, get_mass
from physics.photoelectric import calculate_photoelectric
from physics.barrier import tunneling_curve
from physics.superposition import evolve_superposition
from physics.solver import solve_schrodinger, HarmonicOscillator
from physics.hydrogen import hydrogen_spectrum


class QuantumPlaygroundTests(unittest.TestCase):
    def test_box_energy_increases_with_n_squared(self):
        system = ParticleInBox(1.0, get_mass(1.0))
        self.assertAlmostEqual(system.energy_eV(2) / system.energy_eV(1), 4.0, places=8)

    def test_box_probability_is_normalised(self):
        result = ParticleInBox(1.0).wavefunction(2)
        self.assertAlmostEqual(result["normalization"], 1.0, places=3)

    def test_photoelectric_threshold(self):
        above_threshold = calculate_photoelectric("Cesium", 300)
        below_threshold = calculate_photoelectric("Cesium", 800)
        self.assertTrue(above_threshold["emission"])
        self.assertFalse(below_threshold["emission"])

    def test_tunneling_decreases_with_barrier_width(self):
        result = tunneling_curve(width_nm=0.8)
        self.assertGreater(result["curve_transmission"][0], result["curve_transmission"][-1])
        self.assertGreaterEqual(result["transmission"], 0)
        self.assertLessEqual(result["transmission"], 1)

    def test_superposition_probability_is_normalised(self):
        result = evolve_superposition(time_fs=4.0)
        self.assertAlmostEqual(result["normalization"], 1.0, places=2)

    def test_solver_returns_increasing_energy_levels(self):
        result = solve_schrodinger(grid_points=81, levels=3)
        self.assertEqual(len(result["energies_eV"]), 3)
        self.assertLess(result["energies_eV"][0], result["energies_eV"][1])
        self.assertLess(result["energies_eV"][1], result["energies_eV"][2])

    def test_harmonic_oscillator_energy_levels_increase(self):
        levels = HarmonicOscillator().energy_levels(3)
        self.assertLess(levels[0], levels[1])
        self.assertLess(levels[1], levels[2])

    def test_hydrogen_balmer_first_line(self):
        result = hydrogen_spectrum("Balmer", 4)
        self.assertAlmostEqual(result["lines"][0]["wavelength_nm"], 656.3, delta=1.0)


if __name__ == "__main__":
    unittest.main()
