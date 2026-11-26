"""Species parameters derived from the user's (unpublished) measurements.

Reads studies/private/sim_compare_data.csv (git-ignored, never pushed) and the colony
radius / cell number supplied by the user.  Nothing from this module is written to
tracked files.
"""
import csv, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PRIV = os.path.join(HERE, "private")
CSV = os.path.join(PRIV, "sim_compare_data.csv")

import json
# colony radius r [µm] and number of cells N (user's values) — private, git-ignored
COLONY = json.load(open(os.path.join(PRIV, "colony.json"), encoding="utf-8"))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def load_rows():
    with open(CSV, encoding="utf-8-sig", newline="") as fh:
        return [{k: (_f(v) if k != "species" else v) for k, v in r.items()} for r in csv.DictReader(fh)]


def species_params(species, wt=0.0, amp="tip"):
    """Dictionary of model inputs for one species and PEO concentration.
    amp = "tip": effective-stroke sweep 2θ0 = tip amplitude (consistent with the user's f);
    amp = "eff": 2θ0 = effective-stroke amplitude (stronger stroke, upper bracket)."""
    r = [x for x in load_rows() if x["species"] == species and abs(x["PEO_wt%"] - wt) < 1e-9][0]
    R, N = COLONY[species]["R"], COLONY[species]["N"]
    d = float(np.sqrt(4 * np.pi * R ** 2 / N))
    sweep = r["tip_amp_mean_deg"] if amp == "tip" else r["eff_stroke_amp_mean_deg"]
    t_eff = r["eff_stroke_time_mean_ms"] * 1e-3
    fb = r["beat_freq_mean_Hz"]
    return dict(species=species, wt=wt, eta=r["eta_mPas"], L=r["L_flag_mean_um"], d=d, R=R, N=N,
                theta0=float(np.radians(sweep) / 2), p_eff=float(t_eff * fb), f_beat=fb, t_eff=t_eff,
                omega_exp=r["omega_mean_rad/s"], f_exp=r["f_flag_mean_pN"], F_exp=r["F_colony_mean_pN"],
                U_exp=r["U_dif2400_mean_um/s"], eps_exp=r["epsilon_F/(N*f)"],
                F_exp_sd=r["F_colony_SD"], f_exp_sd=r["f_flag_SD"])


def f_formula(eta, omega, L, a=0.125):
    """The user's single-flagellum force estimate 2πηωL² / (ln(L/2a) + 0.84) [pN]."""
    return 2 * np.pi * eta * 1e-3 * omega * L ** 2 / (np.log(L / (2 * a)) + 0.84)
