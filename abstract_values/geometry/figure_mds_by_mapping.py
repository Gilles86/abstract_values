"""MDS of the crossnobis geometry, separately per mapping, for V1 and parietal cortex.

Group distance matrices per mapping (each participant's RDM normalised by its
mean, then averaged); 2-D metric MDS on sqrt(distance); every solution
Procrustes-aligned (rotation/reflection/scale) to the V1 or parietal solution
pooled over both mappings, so the four maps share a frame. Gabors mark the
orientations; the number next to each is its value (CHF) under that mapping.

    python -m abstract_values.geometry.figure_mds_by_mapping      # pycortex2 env
    -> notes/figures/fig_mds_by_mapping.pdf
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from scipy.linalg import orthogonal_procrustes
from sklearn.manifold import MDS

from abstract_values.geometry.analyze_crossnobis import load
from abstract_values.geometry.figure_circularity import subject_rdms
from abstract_values.visualize.figure_summary import MAP_COL, PAR_C, V1_C, gabor_image

REPO = Path(__file__).resolve().parents[2]
REGIONS = {"V1": (["V1"], V1_C), "Parietal (IPS0–5)": (["IPS0", "IPS1–2", "IPS3–5"], PAR_C)}


def embed(R):
    Y = MDS(n_components=2, dissimilarity="precomputed", random_state=0, n_init=8,
            normalized_stress="auto").fit_transform(np.sqrt(np.clip(R, 0, None)))
    return Y - Y.mean(0)


def align(Y, ref):
    Rm, sc = orthogonal_procrustes(Y, ref)
    return Y @ Rm * (np.linalg.norm(ref) / np.linalg.norm(Y))


def main():
    d = load()
    plt.rcParams.update({"font.family": "Helvetica", "font.size": 7, "pdf.fonttype": 42})
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 6.2), constrained_layout=True)
    for i, (name, (rois, col)) in enumerate(REGIONS.items()):
        pooled, oris = subject_rdms(d, rois)
        ref = embed(np.mean(list(pooled.values()), 0))
        for j, m in enumerate(("cdf", "inverse_cdf")):
            dm_ = d[d.mapping == m]
            rd, _ = subject_rdms(dm_, rois)
            Y = align(embed(np.mean(list(rd.values()), 0)), ref)
            val = (dm_.groupby("ori_i").val_i.first()
                   .reindex(oris).fillna(dm_.groupby("ori_j").val_j.first()).to_numpy())
            ax = axes[i, j]
            ax.plot(Y[:, 0], Y[:, 1], "-", color="0.75", lw=0.8, zorder=1)
            ax.plot([Y[-1, 0], Y[0, 0]], [Y[-1, 1], Y[0, 1]], color="0.3", lw=1.0,
                    ls=(0, (2, 1.5)), zorder=1)
            span = np.ptp(Y, 0).max()
            for (x, y), o, v in zip(Y, oris, val):
                ax.add_artist(AnnotationBbox(OffsetImage(gabor_image(o), zoom=0.26), (x, y),
                                             frameon=False, zorder=3))
                ax.text(x + 0.045 * span, y + 0.045 * span, f"{v:.0f}", fontsize=5.6,
                        color=MAP_COL[m], zorder=4)
            ax.set_aspect("equal")
            ax.margins(0.12)
            ax.axis("off")
            ax.set_title(f"{name}, {'CDF' if m == 'cdf' else 'inverse-CDF'} sessions",
                         loc="left", fontsize=7.5, color=col, fontweight="bold")
    fig.text(0.01, 0.005, "Gabors: the 23 orientations; numbers: their value (CHF) under "
             "that session's mapping; dashed: the step from 172.5° back to 7.5°. Each map is "
             "aligned to the region's pooled solution.", fontsize=6, color="0.4")
    out = REPO / "notes/figures/fig_mds_by_mapping.pdf"
    fig.savefig(out, dpi=300)
    fig.savefig(out.with_suffix(".png"), dpi=200)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
