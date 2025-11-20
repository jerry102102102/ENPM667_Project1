# kr_robot.py
# End-effector normal stiffness via joint stiffness & Jacobian
# Produces: data/robot_single.csv, figs/kr_workspace.png

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

OUT_DIR_DATA = "data"
OUT_DIR_FIGS = "figs"
os.makedirs(OUT_DIR_DATA, exist_ok=True)
os.makedirs(OUT_DIR_FIGS, exist_ok=True)

# Try to use Robotics Toolbox UR5 model
try:
    from roboticstoolbox.models.DH import UR5
except Exception as e:
    raise RuntimeError(
        "robosticstoolbox-python uninstalled run:\n"
        "  uv add roboticstoolbox-python spatialmath-python\n"
        f"error:{e}"
    )

# ----- Robot & parameters -----
robot = UR5()
# Joint stiffness (paper gives KUKA KR70; we demo with UR5 but keep same magnitudes)
K_theta_diag = np.array([200, 90, 100, 6, 2.5, 2]) * 1e4   # [N·m/rad]
K_theta = np.diag(K_theta_diag)

# External wrench along end-effector +Z (normal)
F_scalar = 100.0  # [N] magnitude along normal z
F_vec_e = np.array([0.0, 0.0, F_scalar])  # in EE frame

def op_stiffness_from_Jv(Jv: np.ndarray, Ktheta: np.ndarray, damping: float = 1e-9) -> np.ndarray:
    """
    Operational stiffness using general 3x3 form:
      Kx = ( Jv Ktheta^{-1} Jv^T )^{-1}
    with damping on the inverse for numerical robustness.
    """
    X = Jv @ np.linalg.pinv(Ktheta) @ Jv.T  # 3x3
    X = 0.5 * (X + X.T)                     # symmetrize
    return np.linalg.inv(X + damping * np.eye(3))

def single_pose_compute(q: np.ndarray):
    # World-frame Jacobian 6x6
    J = robot.jacob0(q)     # [vx, vy, vz, wx, wy, wz] columns per joint
    Jv = J[0:3, :]          # translational 3x6

    # EE pose to get rotation
    T = robot.fkine(q)
    R_we = np.array(T.R)    # world->EE rotation, 3x3
    R_ew = R_we.T           # EE->world

    # Operational stiffness at world frame
    Kx_world = op_stiffness_from_Jv(Jv, K_theta)

    # Transform to EE frame (only translational block here)
    Kx_ee = R_ew @ Kx_world @ R_we

    # Compliance in EE frame
    C_ee = np.linalg.inv(Kx_ee)
    delta_e = C_ee @ F_vec_e  # displacement in EE frame due to +Z force

    # Effective normal stiffness (z)
    kr = F_scalar / max(abs(delta_e[2]), 1e-18)
    return dict(
        Kx_world=Kx_world, Kx_ee=Kx_ee,
        delta_e=delta_e, kr=kr,
        T=T
    )

def main():
    # ----- Single pose at q = 0 -----
    q0 = np.zeros(robot.n)
    res = single_pose_compute(q0)
    delta = res["delta_e"]
    kr = res["kr"]
    total_disp = np.linalg.norm(delta)

    # Save CSV (single verification)
    pd.DataFrame([{
        "kr_N_per_m": kr,
        "delta_ex_m": float(delta[0]),
        "delta_ey_m": float(delta[1]),
        "delta_ez_m": float(delta[2]),
        "delta_norm_m": total_disp
    }]).to_csv(os.path.join(OUT_DIR_DATA, "robot_single.csv"), index=False)

    print("[kr_robot] single-pose | kr=%.3e N/m, |delta|=%.3e m" % (kr, total_disp))

    # ----- Workspace map (coarse) -----
    # Sweep two joints while keeping others fixed to generate a (x,z) map like Fig.10
    q1_list = np.linspace(-np.pi/2, np.pi/2, 65)
    q2_list = np.linspace(-np.pi/3,  np.pi/2, 65)

    Xs, Zs, Klogs = [], [], []
    for q1 in q1_list:
        for q2 in q2_list:
            q = np.zeros(robot.n)
            q[0] = q1
            q[1] = q2
            try:
                T = robot.fkine(q)
                x, y, z = T.t
                res = single_pose_compute(q)
                kr_val = res["kr"]
                if np.isfinite(kr_val) and kr_val > 0:
                    Xs.append(x)
                    Zs.append(z)
                    Klogs.append(np.log10(kr_val))
            except Exception:
                # Skip singular/unreachable
                continue

    Xs = np.array(Xs)
    Zs = np.array(Zs)
    Klogs = np.array(Klogs)

    # Plot scatter map of log10(kr) over (x,z)
    plt.figure(figsize=(6.4, 5.4), dpi=160)
    sc = plt.scatter(Xs, Zs, c=Klogs, s=8, cmap="viridis")
    cbar = plt.colorbar(sc)
    cbar.set_label(r'$\log_{10}(k_r)\,[\mathrm{N/m}]$')
    plt.xlabel('x (m)')
    plt.ylabel('z (m)')
    plt.title('EE normal stiffness distribution in workspace (UR5 demo)')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR_FIGS, "kr_workspace.png"))
    plt.close()

    print("[kr_robot] figs/kr_workspace.png & data/robot_single.csv written.")

if __name__ == "__main__":
    main()
