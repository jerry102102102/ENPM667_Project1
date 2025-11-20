# cfc_sim.py
# Grid study for CFC like Fig.14–15: e_sum and p_cost vs. kappa1, kappa2

from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from cfc import default_params, simulate_cfc


def nan_argmin(Z: np.ndarray):
    """nan-safe argmin: returns (i, j) or None if all-NaN."""
    if np.all(np.isnan(Z)):
        return None
    idx = np.nanargmin(Z)
    return np.unravel_index(idx, Z.shape)


def main():
    os.makedirs("figs", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    p = default_params()

    # Paper-like grids (Fig. 14/15):
    # kappa1 in [50..150], kappa2 in [0..6e6] with step 5e5 (includes 0)
    kappa1_vals = np.arange(50.0, 151.0, 10.0)          # 50,60,...,150
    kappa2_vals = np.arange(0.0, 6.0e6 + 1.0, 0.5e6)    # 0, 0.5e6, ..., 6e6

    E = np.full((len(kappa1_vals), len(kappa2_vals)), np.nan, dtype=float)  # e_sum
    Pcost = np.full_like(E, np.nan)                                         # p_cost

    for i, k1 in enumerate(tqdm(kappa1_vals, desc="CFC grid (kappa1)")):
        for j, k2 in enumerate(kappa2_vals):
            out = simulate_cfc(p, kappa1=k1, kappa2=k2, P_gain=100.0)
            E[i, j] = out["e_sum"]
            Pcost[i, j] = out["p_cost"]

    # Save raw arrays
    np.savez("data/cfc_grid.npz",
             kappa1_vals=kappa1_vals,
             kappa2_vals=kappa2_vals,
             e_sum=E, p_cost=Pcost)

    idx_E = nan_argmin(E)
    idx_P = nan_argmin(Pcost)

    if idx_E is not None:
        bestE = (E[idx_E], kappa1_vals[idx_E[0]], kappa2_vals[idx_E[1]])
        print(f"[CFC] min e_sum at kappa1={bestE[1]:.1f}, kappa2={bestE[2]:.2e} -> {bestE[0]:.3e}")
    else:
        print("[CFC] All e_sum values are NaN (check model/params).")

    if idx_P is not None:
        bestP = (Pcost[idx_P], kappa1_vals[idx_P[0]], kappa2_vals[idx_P[1]])
        print(f"[CFC] min p_cost at kappa1={bestP[1]:.1f}, kappa2={bestP[2]:.2e} -> {bestP[0]:.3e}")
    else:
        print("[CFC] All p_cost values are NaN (check model/params).")

    # Mesh for plotting (kappa1 along Y to resemble paper’s perspective)
    K2, K1 = np.meshgrid(kappa2_vals, kappa1_vals, indexing="xy")

    # --- Fig.14-style: e_sum surface ---
    fig1 = plt.figure(figsize=(10, 7.2), dpi=140)
    ax1 = fig1.add_subplot(111, projection="3d")
    # Mask NaNs to avoid warnings in plotting
    E_masked = np.ma.masked_invalid(E)
    surf1 = ax1.plot_surface(K2, K1, E_masked, cmap="coolwarm", linewidth=0, antialiased=True)
    ax1.set_title("Total error under different control parameters")
    ax1.set_xlabel(r"$\kappa_2$ ($\times 10^{6}$)")
    ax1.set_ylabel(r"$\kappa_1$")
    ax1.set_zlabel(r"$e_{\mathrm{sum}}$ (m·s)")
    cbar1 = fig1.colorbar(surf1, shrink=0.75, pad=0.08)
    cbar1.set_label(r"$e_{\mathrm{sum}}$ (m·s)")
    if idx_E is not None:
        ax1.scatter([kappa2_vals[idx_E[1]]], [kappa1_vals[idx_E[0]]], [E[idx_E]], color="k", s=40)
    fig1.tight_layout()
    fig1.savefig("figs/cfc_e_sum_surface.png")
    plt.close(fig1)

    # --- Fig.15-style: p_cost surface ---
    fig2 = plt.figure(figsize=(10, 7.2), dpi=140)
    ax2 = fig2.add_subplot(111, projection="3d")
    P_masked = np.ma.masked_invalid(Pcost)
    surf2 = ax2.plot_surface(K2, K1, P_masked, cmap="coolwarm", linewidth=0, antialiased=True)
    ax2.set_title("Total control cost under different control parameters")
    ax2.set_xlabel(r"$\kappa_2$ ($\times 10^{6}$)")
    ax2.set_ylabel(r"$\kappa_1$")
    ax2.set_zlabel(r"$p_{\mathrm{cost}}$ (Pa·s)")
    cbar2 = fig2.colorbar(surf2, shrink=0.75, pad=0.08)
    cbar2.set_label(r"$p_{\mathrm{cost}}$ (Pa·s)")
    if idx_P is not None:
        ax2.scatter([kappa2_vals[idx_P[1]]], [kappa1_vals[idx_P[0]]], [Pcost[idx_P]], color="k", s=40)
    fig2.tight_layout()
    fig2.savefig("figs/cfc_p_cost_surface.png")
    plt.close(fig2)


if __name__ == "__main__":
    main()
