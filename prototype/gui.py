"""Tkinter prototype: query a voxel (i, j, k), e.g. read from 3D Slicer, and report the candidate's
reliability, local uncertainty, portal-vein distance and, for HIGH candidates, the SCS.

Research prototype: model confidence is not HCC diagnostic probability.
Usage: python prototype/gui.py   (paths from config/parameters.yaml)
"""
import math
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from common import case_file, load_params, load_split, read           # noqa: E402
from scs import normalised_uncertainty, proximity, spatial_concern_score  # noqa: E402

PRM = load_params()
D_REF = PRM["scs"]["d_ref_mm"]
LINE = "-" * 30


def load_case(case):
    keys = ("stage1", "stage2", "var_map", "portal_distance")
    missing = [str(case_file(PRM, case, k)) for k in keys if not case_file(PRM, case, k).is_file()]
    if missing:
        raise FileNotFoundError("Missing file(s):\n" + "\n".join(missing))
    vols = [read(case_file(PRM, case, k), np.float64) for k in keys]   # NIfTI scaling applied
    if len({v.shape for v in vols}) != 1:
        raise ValueError(f"Volume shapes differ: {[v.shape for v in vols]}")
    return vols


def explanation(un, d_mm):
    dist = (f"The queried voxel is within {D_REF:.1f} mm of the portal vein, so portal-vein proximity "
            "is the dominant contributor to spatial concern." if d_mm <= D_REF else
            f"The queried voxel is {d_mm:.2f} mm from the portal vein, so the contribution from vessel "
            "proximity decreases with distance.")
    unc = ("relatively low" if un < 0.20 else "moderate" if un < 0.50 else "relatively high")
    return ("The candidate is HIGH model-confidence. This confidence represents model reliability rather "
            f"than the clinical probability of HCC. {dist} Local predictive uncertainty is {unc}.")


def report(case, ijk, vols):
    s1, s2, var, dist = vols
    head = f"CASE: {case}\n\nCoordinate: {ijk}\n\n"
    if not all(0 <= c < n for c, n in zip(ijk, s1.shape)):
        return head + f"ERROR\n{LINE}\nCoordinate is outside the image volume.\n\nVolume dimensions: {s1.shape}"
    blob, label = int(s1[ijk]), int(s2[ijk])
    u, d = float(var[ijk]), float(dist[ijk])
    if blob == 0:
        return (head + f"STAGE-1 CANDIDATE\n{LINE}\nNONE\n\nNo automatically detected candidate is present "
                "at this coordinate.\n\nSpatial concern score: NOT APPLICABLE.")
    if not (math.isfinite(u) and math.isfinite(d) and u >= 0 and d >= 0) or label not in (1, 2):
        return head + f"Stage-1 Blob ID: {blob}\n\nERROR\n{LINE}\nInvalid values at this coordinate."
    un = normalised_uncertainty(u)
    info = (f"CANDIDATE DETECTED\n{LINE}\nStage-1 Blob ID: {blob}\n"
            f"Model confidence: {'HIGH' if label == 1 else 'LOW'}\n\n"
            f"LOCAL MODEL INFORMATION\n{LINE}\nLocal MC-Dropout variance: {u:.6f}\n"
            f"Normalised uncertainty U_N: {un:.4f}\n")
    if label == 2:
        return (head + info + f"Portal-vein distance: {d:.2f} mm\n\nSPATIAL CONCERN SCORE\n{LINE}\nNOT CALCULATED\n\n"
                f"RATIONALE\n{LINE}\nThis candidate is LOW confidence. The framework deliberately abstains from "
                "assigning a quantitative spatial-concern score to LOW-confidence candidates rather than creating "
                "a false impression of certainty.\n\nLOW confidence represents uncertainty in the automated "
                "candidate characterisation. It does not mean non-tumour or low clinical importance.\n\n"
                "Clinical diagnostic and management decisions are outside the scope of this engineering prototype.")
    score = spatial_concern_score(u, d)
    return (head + info + f"\nANATOMICAL CONTEXT\n{LINE}\nPortal-vein distance: {d:.2f} mm\n"
            f"Portal proximity P: {proximity(d):.4f}\n\nSPATIAL CONCERN\n{LINE}\nScore: {score:.2f} / 10\n\n"
            f"EXPLANATION\n{LINE}\n{explanation(un, d)}")


class App:
    def __init__(self, root):
        train, val = load_split()
        self.cases, self.cache = sorted(train + val), {}
        root.title("Spatial Concern - HCC Candidate Analysis")
        root.geometry("900x720")
        main = ttk.Frame(root, padding=20)
        main.pack(fill="both", expand=True)
        ttk.Label(main, text="Spatial Concern Analysis", font=("Arial", 20, "bold")).pack(anchor="w")
        ttk.Label(main, text="Coordinate-based analysis of automatically detected HCC candidates",
                  font=("Arial", 11)).pack(anchor="w", pady=(2, 18))

        f1 = ttk.LabelFrame(main, text="1. Select case", padding=12)
        f1.pack(fill="x", pady=(0, 12))
        self.case = tk.StringVar(value=self.cases[0])
        ttk.Combobox(f1, textvariable=self.case, values=self.cases, state="readonly", width=25).pack(anchor="w")

        f2 = ttk.LabelFrame(main, text="2. Enter voxel index (I, J, K as shown in 3D Slicer)", padding=12)
        f2.pack(fill="x", pady=(0, 12))
        self.ijk = [tk.StringVar() for _ in range(3)]
        for col, (name, var) in enumerate(zip("IJK", self.ijk)):
            ttk.Label(f2, text=f"{name}:").grid(row=0, column=2 * col, padx=(0, 5))
            ttk.Entry(f2, textvariable=var, width=12).grid(row=0, column=2 * col + 1, padx=(0, 15))
        ttk.Button(f2, text="Check coordinate", command=self.check).grid(row=1, column=0, columnspan=6,
                                                                          pady=(15, 0), sticky="w")

        f3 = ttk.LabelFrame(main, text="3. Analysis result", padding=12)
        f3.pack(fill="both", expand=True, pady=(0, 12))
        self.out = tk.Text(f3, wrap="word", font=("Arial", 11), state="disabled")
        self.out.pack(fill="both", expand=True)
        ttk.Label(main, text="Research prototype - model confidence is not HCC diagnostic probability.",
                  font=("Arial", 9)).pack(anchor="w")

    def check(self):
        try:
            ijk = tuple(int(v.get()) for v in self.ijk)
        except ValueError:
            messagebox.showerror("Invalid coordinate", "I, J and K must be integer voxel indices.")
            return
        case = self.case.get()
        try:
            if case not in self.cache:
                self.cache[case] = load_case(case)
        except Exception as e:                       # missing or inconsistent files
            messagebox.showerror("Case loading error", str(e))
            return
        self.out.configure(state="normal")
        self.out.delete("1.0", tk.END)
        self.out.insert("1.0", report(case, ijk, self.cache[case]))
        self.out.configure(state="disabled")


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
