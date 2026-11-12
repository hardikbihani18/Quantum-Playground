import io
import json
import math
import os
from datetime import datetime
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from flask import Flask, Response, abort, jsonify, redirect, render_template, request, url_for

from errors import InvalidPhysicsInputError
from physics.box import ParticleInBox, get_mass, level_generator
from physics.photoelectric import calculate_photoelectric, METALS
from physics.barrier import tunneling_curve
from physics.superposition import evolve_superposition
from physics.solver import solve_schrodinger, POTENTIALS, HarmonicOscillator
from physics.hydrogen import hydrogen_spectrum, SERIES

app = Flask(__name__)

DATA_FOLDER = os.path.join(os.path.dirname(__file__), "data")
CHART_FOLDER = os.path.join(os.path.dirname(__file__), "static", "charts")
JSON_FILE = os.path.join(DATA_FOLDER, "latest_result.json")

MODULES = {
    "box": "Particle in a 1D box",
    "photoelectric": "Photoelectric effect",
    "tunneling": "Quantum tunneling",
    "superposition": "Superposition and time evolution",
    "solver": "Schrodinger solver",
    "hydrogen": "Hydrogen spectrum",
}

DEFAULTS = {
    "box": {"length_nm": 1.0, "mass_ratio": 1.0, "n": 1},
    "photoelectric": {"metal": "Cesium", "wavelength_nm": 400.0},
    "tunneling": {"energy_eV": 1.0, "barrier_eV": 3.0, "width_nm": 0.5, "mass_ratio": 1.0},
    "superposition": {"length_nm": 1.0, "mass_ratio": 1.0, "c1": 1.0, "c2": 0.6, "c3": 0.2, "time_fs": 5.0},
    "solver": {"kind": "finite_well", "mass_ratio": 1.0, "x_max_nm": 2.0, "grid_points": 161,
               "levels": 4, "width": 1.0, "barrier": 0.6, "curvature": 1.0,
               "separation": 0.6, "amplitude": 0.25},
    "hydrogen": {"series": "Balmer", "n_max": 8},
}


INTEGER_FIELDS = ["n", "grid_points", "levels", "n_max"]
TEXT_FIELDS = ["metal", "kind", "series"]


def read_values(module, source):
    values = DEFAULTS[module].copy()
    for name in values:
        if name not in source:
            continue
        raw = source.get(name)
        if name in TEXT_FIELDS:
            values[name] = str(raw)
        elif name in INTEGER_FIELDS:
            number = float(raw)
            if not number.is_integer():
                raise InvalidPhysicsInputError(name + " must be a whole number.")
            values[name] = int(number)
        else:
            values[name] = float(raw)
        if isinstance(values[name], float) and not math.isfinite(values[name]):
            raise InvalidPhysicsInputError(name + " must be a valid number.")
    return values


def calculate(module, values):
    if module == "box":
        system = ParticleInBox(values["length_nm"], get_mass(values["mass_ratio"]))
        result = system.wavefunction(values["n"])
        levels = list(filter(lambda row: row["n"] <= 6, level_generator(system, 8)))
        level_rows = list(map(lambda row: {
            "n": row["n"], "energy_eV": round(row["energy_eV"], 6)
        }, levels))
        table = pd.DataFrame(level_rows)
        result["levels"] = table.to_dict(orient="records")
        return result

    if module == "photoelectric":
        return calculate_photoelectric(values["metal"], values["wavelength_nm"])

    if module == "tunneling":
        return tunneling_curve(
            values["energy_eV"], values["barrier_eV"], values["width_nm"],
            get_mass(values["mass_ratio"]),
        )

    if module == "superposition":
        coefficients = [values["c1"], values["c2"], values["c3"]]
        return evolve_superposition(
            length_nm=values["length_nm"],
            coefficients=coefficients,
            time_fs=values["time_fs"],
            mass_kg=get_mass(values["mass_ratio"]),
        )

    if module == "solver":
        result = solve_schrodinger(
            kind=values["kind"], mass_ratio=values["mass_ratio"],
            x_max_nm=values["x_max_nm"], grid_points=values["grid_points"],
            levels=values["levels"], width=values["width"],
            barrier=values["barrier"], curvature=values["curvature"],
            separation=values["separation"], amplitude=values["amplitude"],
        )
        if values["kind"] == "harmonic":
            oscillator = HarmonicOscillator(values["mass_ratio"], values["curvature"])
            result["analytic_energies_eV"] = oscillator.energy_levels(values["levels"])
        return result

    if module == "hydrogen":
        return hydrogen_spectrum(values["series"], values["n_max"])

    raise InvalidPhysicsInputError("Unknown experiment.")


def make_chart(module, result):
    figure = None
    try:
        if module == "box":
            figure, axes = plt.subplots(2, 1, figsize=(7.8, 4.8))
            axes[0].plot(result["x_nm"], result["psi"], color="black", linewidth=1.8)
            axes[0].set_ylabel("Wavefunction")
            axes[1].plot(result["x_nm"], result["density"], color="black", linewidth=1.8)
            axes[1].set_ylabel("Probability density")
            axes[1].set_xlabel("Position x (nm)")
            for axis in axes:
                axis.grid(True, linestyle=":", color="#b8b8b8")

        elif module == "photoelectric":
            figure, axis = plt.subplots(figsize=(7.8, 4.4))
            axis.plot(result["graph_frequency_THz"], result["graph_kinetic_energy_eV"], color="black", linewidth=2)
            axis.axvline(result["threshold_frequency_THz"], color="black", linestyle="--", linewidth=1)
            axis.set_xlabel("Frequency (THz)")
            axis.set_ylabel("Maximum kinetic energy (eV)")
            axis.grid(True, linestyle=":", color="#b8b8b8")

        elif module == "tunneling":
            figure, axis = plt.subplots(figsize=(7.8, 4.4))
            axis.plot(result["curve_width_nm"], result["curve_transmission"], color="black", linewidth=2)
            axis.axvline(result["width_nm"], color="black", linestyle="--", linewidth=1)
            axis.set_xlabel("Barrier width (nm)")
            axis.set_ylabel("Transmission probability")
            axis.set_ylim(-0.03, 1.03)
            axis.grid(True, linestyle=":", color="#b8b8b8")

        elif module == "superposition":
            figure, axis = plt.subplots(figsize=(7.8, 4.4))
            axis.plot(result["x_nm"], result["initial_density"], color="black", linestyle="--", label="At t = 0")
            axis.plot(result["x_nm"], result["density"], color="black", linewidth=2, label="Selected time")
            axis.set_xlabel("Position x (nm)")
            axis.set_ylabel("Probability density")
            axis.legend(frameon=False)
            axis.grid(True, linestyle=":", color="#b8b8b8")

        elif module == "solver":
            figure, axis = plt.subplots(figsize=(7.8, 4.8))
            x = result["x_nm"]
            axis.plot(x, result["potential_eV"], color="black", linewidth=1.8, label="Potential")
            for index, wave in enumerate(result["wavefunctions"]):
                energy = result["energies_eV"][index]
                axis.plot(x, [energy + value * 0.12 for value in wave], linewidth=1.2,
                          label="State " + str(index + 1))
            axis.set_xlabel("Position x (nm)")
            axis.set_ylabel("Energy (eV)")
            axis.legend(frameon=False, ncol=2)
            axis.grid(True, linestyle=":", color="#b8b8b8")

        elif module == "hydrogen":
            figure, axis = plt.subplots(figsize=(7.8, 4.4))
            lines = result["lines"]
            labels = [line["transition"] for line in lines]
            wavelengths = [line["wavelength_nm"] for line in lines]
            axis.bar(labels, wavelengths, color="white", edgecolor="black", linewidth=1.5)
            axis.set_xlabel("Electron transition")
            axis.set_ylabel("Wavelength (nm)")
            axis.grid(axis="y", linestyle=":", color="#b8b8b8")

        figure.tight_layout()
        os.makedirs(CHART_FOLDER, exist_ok=True)
        chart_file = module + ".png"
        chart_path = os.path.join(CHART_FOLDER, chart_file)
        figure.savefig(chart_path, format="png", dpi=140, facecolor="white")
        return "charts/" + chart_file
    finally:
        if figure is not None:
            plt.close(figure)


def save_json(module, values, result):
    os.makedirs(DATA_FOLDER, exist_ok=True)
    saved_data = {
        "module": module,
        "saved_at": datetime.now().strftime("%d-%m-%Y %I:%M:%S %p"),
        "values": values,
        "result": result,
    }
    with open(JSON_FILE, "w", encoding="utf-8") as file:
        json.dump(saved_data, file, indent=4)


def show_module(module, values=None, message="", error=""):
    if module not in MODULES:
        abort(404)
    if values is None:
        values = DEFAULTS[module].copy()
    result = None
    chart = None
    if not error:
        try:
            result = calculate(module, values)
            chart = make_chart(module, result)
        except (InvalidPhysicsInputError, ValueError, OverflowError) as problem:
            error = str(problem)
    return render_template(
        "index.html", modules=MODULES, active=module, module_number=list(MODULES).index(module) + 1, values=values,
        result=result, chart=chart, message=message, error=error,
        metals=METALS, series=SERIES, potentials=POTENTIALS,
    )


@app.get("/")
def home():
    return redirect(url_for("module_page", module="box"))


@app.route("/<module>", methods=["GET", "POST"])
def module_page(module):
    if module not in MODULES:
        abort(404)
    message = ""
    try:
        values = read_values(module, request.form if request.method == "POST" else {})
        if request.method == "POST" and request.form.get("action") == "save":
            result = calculate(module, values)
            save_json(module, values, result)
            message = "Result saved to data/latest_result.json."
        return show_module(module, values, message=message)
    except (InvalidPhysicsInputError, ValueError, OverflowError) as problem:
        return show_module(module, values if "values" in locals() else DEFAULTS[module], error=str(problem))


@app.get("/saved")
def saved_result():
    try:
        with open(JSON_FILE, "r", encoding="utf-8") as file:
            saved = json.load(file)
        return render_template("saved.html", saved=saved, error="")
    except FileNotFoundError:
        return render_template("saved.html", saved=None, error="No saved result yet. Use the Save JSON button in an experiment.")


@app.get("/api/health")
def health():
    return jsonify({"status": "ok", "project": "Quantum Playground"})


@app.get("/api/box/levels")
def api_box_levels():
    try:
        values = read_values("box", request.args)
        count = int(request.args.get("count", 6))
        if count < 1 or count > 20:
            raise InvalidPhysicsInputError("Count must be between 1 and 20.")
        system = ParticleInBox(values["length_nm"], get_mass(values["mass_ratio"]))
        levels = list(level_generator(system, count))
        return jsonify({"levels": levels})
    except (InvalidPhysicsInputError, ValueError, OverflowError) as problem:
        return jsonify({"error": str(problem)}), 400


@app.route("/api/<module>", methods=["GET", "POST"])
@app.route("/api/box/wavefunction", methods=["POST"], defaults={"module": "box"})
@app.route("/api/photoelectric", methods=["POST"], defaults={"module": "photoelectric"})
@app.route("/api/tunneling", methods=["POST"], defaults={"module": "tunneling"})
@app.route("/api/superposition", methods=["POST"], defaults={"module": "superposition"})
@app.route("/api/solve", methods=["POST"], defaults={"module": "solver"})
@app.route("/api/hydrogen", methods=["GET"], defaults={"module": "hydrogen"})
def api_experiment(module):
    if module not in MODULES:
        abort(404)
    source = request.args if request.method == "GET" else request.get_json(silent=True)
    if not isinstance(source, dict):
        source = request.form
    try:
        values = read_values(module, source)
        result = calculate(module, values)
        return jsonify({"experiment": MODULES[module], "inputs": values, "results": result})
    except (InvalidPhysicsInputError, ValueError, OverflowError) as problem:
        return jsonify({"error": str(problem)}), 400


def csv_rows(module, values):
    result = calculate(module, values)
    if module == "box":
        return pd.DataFrame({
            "x_nm": result["x_nm"],
            "wavefunction": result["psi"],
            "probability_density": result["density"],
        })
    if module == "photoelectric":
        return pd.DataFrame({
            "frequency_THz": result["graph_frequency_THz"],
            "kinetic_energy_eV": result["graph_kinetic_energy_eV"],
        })
    if module == "tunneling":
        return pd.DataFrame({
            "width_nm": result["curve_width_nm"],
            "transmission_probability": result["curve_transmission"],
        })
    if module == "superposition":
        return pd.DataFrame({
            "x_nm": result["x_nm"],
            "initial_density": result["initial_density"],
            "selected_time_density": result["density"],
        })
    if module == "solver":
        data = {"x_nm": result["x_nm"], "potential_eV": result["potential_eV"]}
        for index, wave in enumerate(result["wavefunctions"]):
            data["wavefunction_" + str(index + 1)] = wave
        return pd.DataFrame(data)
    return pd.DataFrame(result["lines"])


@app.get("/export/<module>.csv")
@app.get("/api/export")
def export_csv(module=None):
    if module is None:
        module = request.args.get("module", "box")
    if module not in MODULES:
        abort(404)
    try:
        values = read_values(module, request.args)
        table = csv_rows(module, values)
        output = io.StringIO()
        table.to_csv(output, index=False)
        return Response(
            output.getvalue(), mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=quantum_" + module + ".csv"},
        )
    except (InvalidPhysicsInputError, ValueError, OverflowError) as problem:
        return str(problem), 400


if __name__ == "__main__":
    app.run(debug=True)
