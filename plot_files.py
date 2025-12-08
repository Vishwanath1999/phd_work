# %%
import os, glob, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import gaussian_filter
from numpy.lib.stride_tricks import sliding_window_view
from mpl_toolkits.mplot3d import Axes3D  # noqa
from matplotlib import cm

# ====== USER OPTIONS ======
SAVE_PLOTS = False  # <--- set False if you do NOT want to save figures
data_dir = './results/mrr_sac_cluster_delayed_toptica_pow_ton_un_norm_high_only_detuning_v6/high/random_pow'  # <--- set to your folder with *_p_cav.npy, *_detuning_theta_sum.npy
pump_min_W = 0.12
pump_max_W = 0.18
# ==========================


# ---------- classification helpers ----------

def classify_state_instant(P_mW,
                           rel_jitter,
                           cw_thresh_mW=3.0,
                           dks_low_mW=4.0,
                           dks_high_mW=5.5,
                           jitter_rel_thresh=0.02):
    """Return 'cw', 'MI/chaos', or 'DKS' for one time sample."""
    if P_mW < cw_thresh_mW:
        return "cw"
    if dks_low_mW <= P_mW <= dks_high_mW and rel_jitter < jitter_rel_thresh:
        return "DKS"
    return "MI/chaos"


# ---------- main accumulation over runs ----------

def build_phase_counts_time_resolved(
        data_dir,
        pump_min_W=0.12,
        pump_max_W=0.18,
        n_pump_bins=25,
        n_det_bins=60,
        cw_thresh_mW=3.0,
        dks_low_mW=4.0,
        dks_high_mW=5.5,
        jitter_window=10,
        jitter_rel_thresh=0.02,
):
    pcav_files = sorted(glob.glob(os.path.join(data_dir, "*_p_cav.npy")))
    if not pcav_files:
        print("No *_p_cav.npy found in", data_dir)
        return None, None, None, (None, None)

    # global detuning range
    all_det = []
    for f in pcav_files:
        dpath = f.replace("_p_cav.npy", "_detuning_theta_sum.npy")
        if os.path.exists(dpath):
            all_det.append(np.load(dpath).squeeze())
    if not all_det:
        print("No detuning files found.")
        return None, None, None, (None, None)

    all_det = np.concatenate(all_det)
    det_min = float(all_det.min())
    det_max = float(all_det.max())

    pump_edges = np.linspace(1e3 * pump_min_W, 1e3 * pump_max_W, n_pump_bins + 1)
    det_edges  = np.linspace(det_min, det_max, n_det_bins + 1)

    cw_counts  = np.zeros((n_pump_bins, n_det_bins))
    mi_counts  = np.zeros((n_pump_bins, n_det_bins))
    dks_counts = np.zeros((n_pump_bins, n_det_bins))

    for pcav_path in pcav_files:
        base = os.path.basename(pcav_path)
        m = re.match(r".*?_(\d{3})_.*_p_cav\.npy", base)
        if m is None:
            continue
        p_W  = float("0." + m.group(1))
        p_mW = 1e3 * p_W

        dpath = pcav_path.replace("_p_cav.npy", "_detuning_theta_sum.npy")
        if not os.path.exists(dpath):
            continue

        pcav = np.load(pcav_path).squeeze()  # W
        det  = np.load(dpath).squeeze()      # GHz
        N = min(pcav.size, det.size)
        if N < 5:
            continue
        pcav = pcav[:N]
        det  = det[:N]
        pcav_mW = 1e3 * pcav

        # local relative jitter
        if N < jitter_window:
            rel_jitter = np.zeros_like(pcav_mW)
        else:
            win = sliding_window_view(pcav_mW, jitter_window)
            std = win.std(axis=-1)
            pad_front = jitter_window // 2
            pad_back = N - (pad_front + std.size)
            rel_jitter = np.concatenate([
                std[:1].repeat(pad_front),
                std,
                std[-1:].repeat(pad_back),
            ]) / (pcav_mW + 1e-12)

        i_p = np.searchsorted(pump_edges, p_mW, side="right") - 1
        if not (0 <= i_p < n_pump_bins):
            continue

        for t in range(N):
            d = det[t]
            j_d = np.searchsorted(det_edges, d, side="right") - 1
            if not (0 <= j_d < n_det_bins):
                continue

            state = classify_state_instant(
                pcav_mW[t],
                rel_jitter[t],
                cw_thresh_mW=cw_thresh_mW,
                dks_low_mW=dks_low_mW,
                dks_high_mW=dks_high_mW,
                jitter_rel_thresh=jitter_rel_thresh,
            )

            if state == "cw":
                cw_counts[i_p, j_d] += 1
            elif state == "MI/chaos":
                mi_counts[i_p, j_d] += 1
            else:  # DKS
                dks_counts[i_p, j_d] += 1

    return cw_counts, mi_counts, dks_counts, (pump_edges, det_edges)


def phase_index_from_counts(cw_counts, mi_counts, dks_counts, smooth_sigma=0.6):
    if cw_counts is None:
        return None
    if smooth_sigma > 0:
        cw  = gaussian_filter(cw_counts,  sigma=smooth_sigma)
        mi  = gaussian_filter(mi_counts,  sigma=smooth_sigma)
        dks = gaussian_filter(dks_counts, sigma=smooth_sigma)
    else:
        cw, mi, dks = cw_counts, mi_counts, dks_counts

    phase_idx = np.zeros_like(cw, dtype=int)
    phase_idx[cw > 0] = 1
    phase_idx[mi > cw] = 2
    mask_dks = dks > np.maximum(cw, mi)
    phase_idx[mask_dks] = 3
    return phase_idx


# ---------- plot 1: phase boundaries ----------

def plot_phase_boundaries(cw_counts, mi_counts, dks_counts,
                          phase_idx, pump_edges, det_edges,
                          flip_detuning=True, save_path=None):
    if phase_idx is None:
        return None

    if flip_detuning:
        det_edges = det_edges[::-1]
        phase_idx = phase_idx[:, ::-1]
        cw_counts = cw_counts[:, ::-1]
        mi_counts = mi_counts[:, ::-1]
        dks_counts = dks_counts[:, ::-1]

    activity = mi_counts + dks_counts
    extent = [det_edges[0], det_edges[-1], pump_edges[0], pump_edges[-1]]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(activity, origin="lower", aspect="auto",
                   extent=extent, cmap="Greys")

    levels = [0.5, 1.5, 2.5]
    x = np.linspace(det_edges[0], det_edges[-1], phase_idx.shape[1])
    y = np.linspace(pump_edges[0], pump_edges[-1], phase_idx.shape[0])
    cs = ax.contour(x, y, phase_idx, levels=levels,
                    colors=["tab:blue", "tab:orange", "tab:green"],
                    linewidths=1.5)
    labels = ["empty↔cw", "cw↔MI", "MI↔DKS"]
    for c, lab in zip(cs.collections, labels):
        c.set_label(lab)

    ax.set_xlabel("Effective detuning (GHz)")
    ax.set_ylabel("Pump power P_pmp (mW)")
    ax.set_title("Phase boundaries: cw / MI / DKS")
    ax.legend(loc="best")
    plt.colorbar(im, ax=ax, label="non‑cw activity (counts)")
    plt.tight_layout()
    if SAVE_PLOTS and save_path is not None:
        fig.savefig(save_path, dpi=200)
    return fig


# ---------- plot 2: fractional DKS 2D ----------

def plot_fractional_DKS(cw_counts, mi_counts, dks_counts,
                        pump_edges, det_edges,
                        flip_detuning=True, save_path=None):
    total = cw_counts + mi_counts + dks_counts
    with np.errstate(divide='ignore', invalid='ignore'):
        f_dks = np.where(total > 0, dks_counts / total, 0.0)

    if flip_detuning:
        det_edges = det_edges[::-1]
        f_dks = f_dks[:, ::-1]

    extent = [det_edges[0], det_edges[-1], pump_edges[0], pump_edges[-1]]
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(f_dks, origin="lower", aspect="auto",
                   extent=extent, vmin=0.0, vmax=1.0, cmap="viridis")

    ax.set_xlabel("Effective detuning (GHz)")
    ax.set_ylabel("Pump power P_pmp (mW)")
    ax.set_title("Fraction of time in DKS state")

    ny, nx = f_dks.shape
    x = np.linspace(det_edges[0], det_edges[-1], nx)
    y = np.linspace(pump_edges[0], pump_edges[-1], ny)
    cs = ax.contour(x, y, f_dks, levels=[0.25, 0.5, 0.75],
                    colors="white", linewidths=1.2)
    ax.clabel(cs, fmt="%.2f", colors="white", fontsize=9)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("f_DKS")
    plt.tight_layout()
    if SAVE_PLOTS and save_path is not None:
        fig.savefig(save_path, dpi=200)
    return fig


# ---------- plot 3: fractional DKS 3D ----------

def plot_fractional_DKS_3d(cw_counts, mi_counts, dks_counts,
                           pump_edges, det_edges, save_path=None):
    total = cw_counts + mi_counts + dks_counts
    with np.errstate(divide='ignore', invalid='ignore'):
        f_dks = np.where(total > 0, dks_counts / total, 0.0)

    pump_centers = 0.5 * (pump_edges[:-1] + pump_edges[1:])
    det_centers  = 0.5 * (det_edges[:-1] + det_edges[1:])
    D, P = np.meshgrid(det_centers, pump_centers)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(D, P, f_dks, cmap=cm.viridis,
                           linewidth=0, antialiased=True)
    ax.set_xlabel("Effective detuning (GHz)")
    ax.set_ylabel("Pump power P_pmp (mW)")
    ax.set_zlabel("f_DKS")
    ax.set_title("3D DKS probability landscape")
    fig.colorbar(surf, shrink=0.6, aspect=12, label="f_DKS")
    if SAVE_PLOTS and save_path is not None:
        fig.savefig(save_path, dpi=200)
    return fig


# ---------- plot 4: time-sample scatter ----------

def build_time_sample_cloud(data_dir,
                            cw_thresh_mW=3.0,
                            dks_low_mW=4.0,
                            dks_high_mW=5.5,
                            jitter_window=10,
                            jitter_rel_thresh=0.02):
    pcav_files = sorted(glob.glob(os.path.join(data_dir, "*_p_cav.npy")))
    det_list, pump_list, state_list = [], [], []

    for pcav_path in pcav_files:
        base = os.path.basename(pcav_path)
        m = re.match(r".*?_(\d{3})_.*_p_cav\.npy", base)
        if m is None:
            continue
        p_W  = float("0." + m.group(1))
        p_mW = 1e3 * p_W

        dpath = pcav_path.replace("_p_cav.npy", "_detuning_theta_sum.npy")
        if not os.path.exists(dpath):
            continue

        pcav = np.load(pcav_path).squeeze()
        det  = np.load(dpath).squeeze()
        N = min(pcav.size, det.size)
        if N < 5:
            continue
        pcav = pcav[:N]
        det  = det[:N]
        pcav_mW = 1e3 * pcav

        if N < jitter_window:
            rel_jitter = np.zeros_like(pcav_mW)
        else:
            win = sliding_window_view(pcav_mW, jitter_window)
            std = win.std(axis=-1)
            pad_front = jitter_window // 2
            pad_back = N - (pad_front + std.size)
            rel_jitter = np.concatenate([
                std[:1].repeat(pad_front),
                std,
                std[-1:].repeat(pad_back),
            ]) / (pcav_mW + 1e-12)

        for t in range(N):
            state = classify_state_instant(
                pcav_mW[t],
                rel_jitter[t],
                cw_thresh_mW=cw_thresh_mW,
                dks_low_mW=dks_low_mW,
                dks_high_mW=dks_high_mW,
                jitter_rel_thresh=jitter_rel_thresh,
            )
            det_list.append(det[t])
            pump_list.append(p_mW)
            state_list.append(state)

    return np.array(det_list), np.array(pump_list), np.array(state_list)


def plot_time_sample_scatter(det_all, pump_all, state_all,
                             flip_detuning=True, save_path=None):
    if flip_detuning:
        # just reverse x-axis ordering visually is enough
        pass  # det_all values themselves carry sign; no transform needed

    fig, ax = plt.subplots(figsize=(7, 6))
    for state, color in [("cw", "tab:blue"),
                         ("MI/chaos", "tab:orange"),
                         ("DKS", "tab:green")]:
        mask = (state_all == state)
        if np.any(mask):
            ax.scatter(det_all[mask], pump_all[mask],
                       s=4, alpha=0.2, color=color, label=state)

    ax.set_xlabel("Effective detuning (GHz)")
    ax.set_ylabel("Pump power P_pmp (mW)")
    ax.set_title("Time-sample cloud: cw / MI / DKS")
    ax.legend(markerscale=3)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    if SAVE_PLOTS and save_path is not None:
        fig.savefig(save_path, dpi=200)
    return fig



def plot_DKS_fraction_with_contours(cw_counts, mi_counts, dks_counts,
                                    phase_idx, pump_edges, det_edges,
                                    flip_detuning=True, save_path=None):
    """
    2D color map of f_DKS with white iso-contours,
    plus thin colored contours for cw / MI / DKS “phase” boundaries.
    """
    total = cw_counts + mi_counts + dks_counts
    with np.errstate(divide='ignore', invalid='ignore'):
        f_dks = np.where(total > 0, dks_counts / total, 0.0)

    if flip_detuning:
        det_edges = det_edges[::-1]
        f_dks = f_dks[:, ::-1]
        phase_idx = phase_idx[:, ::-1]

    extent = [det_edges[0], det_edges[-1], pump_edges[0], pump_edges[-1]]
    ny, nx = f_dks.shape
    x = np.linspace(det_edges[0], det_edges[-1], nx)
    y = np.linspace(pump_edges[0], pump_edges[-1], ny)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(f_dks, origin="lower", aspect="auto",
                   extent=extent, vmin=0.0, vmax=1.0, cmap="viridis")

    ax.set_xlabel("Effective detuning (GHz)")
    ax.set_ylabel("Pump power P_pmp (mW)")
    ax.set_title("Fraction of time in DKS state")

    # DKS fraction iso-contours (0.25, 0.5, 0.75)
    cs_dks = ax.contour(x, y, f_dks,
                        levels=[0.25, 0.5, 0.75],
                        colors="white", linewidths=1.2)
    ax.clabel(cs_dks, fmt="%.2f", colors="white", fontsize=9)

    # Regime boundaries from phase index:
    # levels between 0/1, 1/2, 2/3 → 0.5, 1.5, 2.5
    levels_phase = [0.5, 1.5, 2.5]
    cs_phase = ax.contour(x, y, phase_idx,
                          levels=levels_phase,
                          colors=["tab:blue", "tab:orange", "tab:green"],
                          linewidths=1.0)
    # Optional labels:
    #  blue: empty↔cw, orange: cw↔MI, green: MI↔DKS

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("f_DKS")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200)
    return fig


def plot_regime_contour_only(phase_idx, pump_edges, det_edges,
                             flip_detuning=True, save_path=None):
    """
    Pure contour map that says: at this (pump, detuning) the
    dominant regime is cw / MI / DKS.
    """
    if flip_detuning:
        det_edges = det_edges[::-1]
        phase_idx = phase_idx[:, ::-1]

    ny, nx = phase_idx.shape
    x = np.linspace(det_edges[0], det_edges[-1], nx)
    y = np.linspace(pump_edges[0], pump_edges[-1], ny)

    fig, ax = plt.subplots(figsize=(7, 6))

    # filled background by regime index (optional, light colors)
    im = ax.imshow(phase_idx, origin="lower", aspect="auto",
                   extent=[det_edges[0], det_edges[-1],
                           pump_edges[0], pump_edges[-1]],
                   cmap=plt.get_cmap("tab10", 4), vmin=0, vmax=3, alpha=0.4)

    # sharp boundaries
    levels = [0.5, 1.5, 2.5]
    cs = ax.contour(x, y, phase_idx, levels=levels,
                    colors=["tab:blue", "tab:orange", "tab:green"],
                    linewidths=1.5)

    ax.set_xlabel("Effective detuning (GHz)")
    ax.set_ylabel("Pump power P_pmp (mW)")
    ax.set_title("Regime map: cw / MI / DKS")

    # legend for contours
    labels = ["empty↔cw", "cw↔MI", "MI↔DKS"]
    for c, lab in zip(cs.collections, labels):
        c.set_label(lab)
    ax.legend(loc="best")

    plt.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=200)
    return fig

# ---------- main driver ----------
# %%
if __name__ == "__main__":
    os.makedirs(data_dir, exist_ok=True)

    cw_c, mi_c, dks_c, (pump_edges, det_edges) = build_phase_counts_time_resolved(
        data_dir,
        pump_min_W=pump_min_W,
        pump_max_W=pump_max_W,
        n_pump_bins=25,
        n_det_bins=60,
        cw_thresh_mW=3.0,
        dks_low_mW=4.0,
        dks_high_mW=5.5,
        jitter_window=10,
        jitter_rel_thresh=0.02,
    )

    phase_idx = phase_index_from_counts(cw_c, mi_c, dks_c, smooth_sigma=0.6)

    plot_phase_boundaries(
        cw_c, mi_c, dks_c, phase_idx, pump_edges, det_edges,
        flip_detuning=True,
        save_path=os.path.join(data_dir, "phase_boundaries.png"),
    )

    plot_fractional_DKS(
        cw_c, mi_c, dks_c, pump_edges, det_edges,
        flip_detuning=True,
        save_path=os.path.join(data_dir, "fractional_DKS.png"),
    )

    plot_fractional_DKS_3d(
        cw_c, mi_c, dks_c, pump_edges, det_edges,
        save_path=os.path.join(data_dir, "fractional_DKS_3d.png"),
    )

    det_all, pump_all, state_all = build_time_sample_cloud(
        data_dir,
        cw_thresh_mW=3.0,
        dks_low_mW=4.0,
        dks_high_mW=5.5,
        jitter_window=10,
        jitter_rel_thresh=0.02,
    )

    plot_time_sample_scatter(
        det_all, pump_all, state_all,
        flip_detuning=True,
        save_path=os.path.join(data_dir, "time_sample_scatter.png"),
    )

    fig1 = plot_DKS_fraction_with_contours(
    cw_c, mi_c, dks_c,
    phase_idx,
    pump_edges,
    det_edges,
    flip_detuning=True,
    save_path=os.path.join(data_dir, "DKS_fraction_plus_regimes.png"),
    )

    fig2 = plot_regime_contour_only(
        phase_idx,
        pump_edges,
        det_edges,
        flip_detuning=True,
        save_path=os.path.join(data_dir, "regime_contours.png"),
    )

    plt.show()

# %%
