import numpy as np
from sklearn.manifold import MDS
from scipy.spatial import procrustes
from abstract_values.geometry.analyze_crossnobis import load
from abstract_values.geometry.figure_circularity import subject_rdms
d = load()
regions = {"V1": ["V1"], "Parietal": ["IPS0", "IPS1–2", "IPS3–5"]}
rng = np.random.default_rng(0)
def embed(R):
    m = MDS(n_components=2, dissimilarity="precomputed", random_state=0, normalized_stress="auto", n_init=4)
    Y = m.fit_transform(np.sqrt(np.clip(R, 0, None)))
    return Y, m.stress_
def seam_ratio(Y):
    steps = np.linalg.norm(np.diff(Y, axis=0), axis=1)
    return np.linalg.norm(Y[-1] - Y[0]) / np.median(steps)
for name, rois in regions.items():
    rd, oris = subject_rdms(d, rois)
    S = np.array(list(rd.values()))
    Y0, st0 = embed(S.mean(0))
    r0 = seam_ratio(Y0)
    ratios, disp, sts = [], [], []
    for b in range(300):
        idx = rng.integers(0, len(S), len(S))
        Y, st = embed(S[idx].mean(0))
        _, Ya, dis = procrustes(Y0, Y)
        ratios.append(seam_ratio(Y)); disp.append(dis); sts.append(st)
    ratios = np.array(ratios)
    print(f"{name}: seam/median-step = {r0:.2f}, bootstrap 95% [{np.quantile(ratios,.025):.2f}, {np.quantile(ratios,.975):.2f}]; "
          f"Procrustes disparity median {np.median(disp):.3f} (95% {np.quantile(disp,.975):.3f}); stress {st0:.3f}")
