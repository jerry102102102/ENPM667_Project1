# kw_workpiece.py
# Workpiece equivalent stiffness kw(x,y) & distribution
# Produces: data/workpiece_center.csv, figs/kw_distribution.png

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR_DATA = "data"
OUT_DIR_FIGS = "figs"
os.makedirs(OUT_DIR_DATA, exist_ok=True)
os.makedirs(OUT_DIR_FIGS, exist_ok=True)

# ----- Parameters (from paper 4.1) -----
D = 800.0            # bending stiffness [N·m]
a = 0.5              # half-length [m]  (total length 1 m)
b = 0.5              # half-width  [m]  (total width 1 m)
F0 = 100.0           # point load [N]
xw = 0.0             # load position x [m]
yw = 0.0             # load position y [m]

# Reference numbers described in the paper (for comparison only)
paper_center_disp_fea = 7.02e-4  # m (COMSOL)
paper_center_disp_analytic = 6.41e-4  # m (adjusted analytic)

def kw_xy(x: np.ndarray, y: np.ndarray, D: float, a: float, b: float, eps: float = 1e-12) -> np.ndarray:
    """
    Equivalent stiffness of the plate at (x,y) from Eq. (13) in the paper:
    k_w = D*pi^4*(3a^4 + 2a^2 b^2 + 3b^4) / (a^3 b^3 * (1+cos(pi x / a))^2 * (1+cos(pi y / b))^2)
    """
    num = D * np.pi**4 * (3*a**4 + 2*a**2*b**2 + 3*b**4)
    denom_x = (1.0 + np.cos(np.pi * x / a))**2
    denom_y = (1.0 + np.cos(np.pi * y / b))**2
    denom = a**3 * b**3 * denom_x * denom_y
    return num / (denom + eps)

def main():
    # Center stiffness & displacement
    k_center = kw_xy(np.array([xw]), np.array([yw]), D, a, b)[0]
    w_hat = F0 / k_center  # displacement under F0 at (xw,yw)

    # Save CSV comparing to paper values
    df = pd.DataFrame([{
        "D_Nm": D, "a_m": a, "b_m": b, "F0_N": F0,
        "xw_m": xw, "yw_m": yw,
        "kw_center_N_per_m": k_center,
        "w_hat_center_m": w_hat,
        "paper_fea_center_m": paper_center_disp_fea,
        "paper_adjusted_analytic_m": paper_center_disp_analytic,
        "accuracy_vs_fea_%": 100.0 * (1.0 - abs(w_hat - paper_center_disp_fea) / paper_center_disp_fea)
    }])
    df.to_csv(os.path.join(OUT_DIR_DATA, "workpiece_center.csv"), index=False)

    # Distribution across the plate
    n = 201
    xs = np.linspace(-a, a, n)
    ys = np.linspace(-b, b, n)
    X, Y = np.meshgrid(xs, ys)
    KW = kw_xy(X, Y, D, a, b)

    # Plot log10(k_w)
    plt.figure(figsize=(6.5, 5.4), dpi=160)
    lvls = 30
    cs = plt.contourf(X, Y, np.log10(KW), levels=lvls)
    cbar = plt.colorbar(cs)
    cbar.set_label(r'$\log_{10}(k_w)\,[\mathrm{N/m}]$')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Workpiece stiffness distribution (log10)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR_FIGS, "kw_distribution.png"))
    plt.close()

    print("[kw_workpiece] center disp (analytic) = %.3e m; FEA(ref)=%.3e m" %
          (w_hat, paper_center_disp_fea))
    print("[kw_workpiece] figs/kw_distribution.png & data/workpiece_center.csv written.")

if __name__ == "__main__":
    main()
