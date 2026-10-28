"""A3 (cluster step): crossnobis + model-implied RDMs for the 90 deg categorical index.

Per subject, ROI and session (= mapping) this writes the 23 x 23 distance matrix
between the 23 orientations (7.5 .. 172.5 deg) in four versions:

  kind = crossnobis, split = all   8 runs, leave-one-run-out (as geometry/crossnobis.py)
  kind = crossnobis, split = odd   runs 1,3,5,7 only (crossnobis within the half)
  kind = crossnobis, split = even  runs 2,4,6,8 only  -> split-half reliability
  kind = vonmises  (model-implied, orientation model 'vonmises-prf')
  kind = aprf      (model-implied, value model 'aprf' = log-Gaussian on CHF)

Crossnobis reuses ``abstract_values.geometry.crossnobis.crossnobis`` unchanged
(Ledoit-Wolf whitening from the training runs, unbiased, per voxel).

Model-implied distance between stimuli i, j:
    D_ij = mean_v ((f_v(s_i) - f_v(s_j)) / sigma_v)^2
f_v = the voxel's fitted tuning curve (joint fit over both sessions), evaluated at
the orientation (vonmises-prf) or at the session's CHF value (aprf), sigma_v = SD of
that voxel's residuals (beta - prediction) within the session. This is the linear
Fisher information of an independent-noise population, summed over the stimulus
step, i.e. the model's version of the crossnobis distance. Voxels: those in the ROI
whose mean leave-one-run-out cvR2 beats the null model (aprf-null.cv), the
project's signal criterion (utils/data.py, cvr2_signal); <10 such voxels -> NaN.

    python compute_rdms.py 03 [--bids-folder ...]
    -> derivatives/followup_2026-10/a3_categorical_90/sub-<s>/sub-<s>_desc-a3_rdm.tsv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from nilearn import image, masking

from abstract_values.encoding_models.model_specs import get_spec
from abstract_values.geometry.crossnobis import crossnobis, trial_labels
from abstract_values.utils.data import BIDS_FOLDER, Subject

# label -> list of (mask desc, hemi); unions are taken over the list
ROIS = {
    "V1": [("BensonV1", "LR")],
    "V2V3": [("BensonV2", "LR"), ("BensonV3", "LR")],
    "IPS0": [("IPS0", None)],
    "IPS12": [("IPS12", None)],
    "IPS345": [("IPS345", None)],
    "IPS": [("IPS0", None), ("IPS12", None), ("IPS345", None)],
    "NPCr": [("NPCr", None)],
}
MODELS = {"vonmises": ("vonmises-prf", "vonmises-prf.cv", "orientation"),
          "aprf": ("standard", "aprf.cv", "value")}
NULL_CV = "aprf-null.cv"
MIN_VOX = 10


def roi_mask(sub, parts, ref):
    data = None
    for desc, hemi in parts:
        m = image.resample_to_img(sub.get_roi_mask(desc, hemi=hemi), ref,
                                  interpolation="nearest")
        d = m.get_fdata() > 0.5
        data = d if data is None else (data | d)
    return nib.Nifti1Image(data.astype(np.uint8), ref.affine)


def mean_cvr2(sub, model_dir, bids, ref):
    d = Path(bids) / "derivatives" / "encoding_models" / model_dir / f"sub-{sub.subject_id}" / "func"
    fns = sorted(f for f in d.glob(f"sub-{sub.subject_id}_ses-*_run-*_desc-cvr2_pe.nii.gz"))
    if not fns:
        return None
    imgs = [image.resample_to_img(str(f), ref, interpolation="nearest") for f in fns]
    return image.mean_img(imgs)


def load_pars(sub, spec, bids, ref):
    d = Path(bids) / "derivatives" / "encoding_models" / spec.out_subdir / f"sub-{sub.subject_id}" / "func"
    out = {}
    for p in spec.save_params:
        fn = d / f"sub-{sub.subject_id}_task-abstractvalue_space-T1w_desc-{p}_pe.nii.gz"
        if not fn.exists():
            return None
        out[p] = image.resample_to_img(str(fn), ref, interpolation="nearest")
    return out


def model_rdm(model, pars_df, stim_levels, stim_trials, B):
    """stim_levels: (23,) stimulus coordinates; stim_trials: (n_trials,) per trial."""
    def pred(x):
        p = model.predict(paradigm=pd.DataFrame({"x": np.asarray(x, np.float32)}),
                          parameters=pars_df.astype(np.float32))
        return np.asarray(p)
    F = pred(stim_levels)                     # (23, vox)
    resid = B - pred(stim_trials)
    sig = resid.std(0)
    ok = np.isfinite(sig) & (sig > 0) & np.all(np.isfinite(F), 0)
    Z = F[:, ok] / sig[ok]
    diff = Z[:, None, :] - Z[None, :, :]
    return (diff ** 2).mean(-1), int(ok.sum())


def main(subject, bids_folder=BIDS_FOLDER):
    sub = Subject(subject, bids_folder=bids_folder)
    lab = trial_labels(sub)
    betas = sub.get_single_trial_estimates(sorted(sub.get_sessions()))
    if betas.shape[-1] != len(lab):
        raise ValueError(f"{betas.shape[-1]} betas vs {len(lab)} labelled trials")
    ref = image.index_img(betas, 0)
    null_cv = mean_cvr2(sub, NULL_CV, bids_folder, ref)
    models = {}
    for k, (spec_name, cv_dir, space) in MODELS.items():
        spec = get_spec(spec_name)
        pars = load_pars(sub, spec, bids_folder, ref)
        cv = mean_cvr2(sub, cv_dir, bids_folder, ref)
        if pars is None or cv is None or null_cv is None:
            print(f"sub-{subject}: no {k} fit / cvR2 -> skip model-implied {k}")
            continue
        models[k] = (spec, pars, cv, space)

    rows = []
    for roi, parts in ROIS.items():
        try:
            mask = roi_mask(sub, parts, ref)
        except FileNotFoundError as e:
            print(f"skip {roi}: {e}")
            continue
        X = masking.apply_mask(betas, mask)
        good = np.all(np.isfinite(X), 0) & (X.std(0) > 0)
        X = X[:, good]
        for ses, g in lab.groupby("session"):
            idx = g.index.to_numpy()
            oris = np.sort(g.orientation.unique())
            val = g.groupby("orientation").value.first().reindex(oris).to_numpy()
            i, j = np.triu_indices(len(oris), 1)

            def emit(D, kind, split, nvox):
                rows.append(pd.DataFrame(dict(
                    subject=subject, roi=roi, session=ses, mapping=g.mapping.iloc[0],
                    kind=kind, split=split, n_voxels=nvox, ori_i=oris[i], ori_j=oris[j],
                    val_i=val[i], val_j=val[j], d=D[i, j])))

            runs = g.run.to_numpy()
            for split, keep in [("all", np.ones(len(g), bool)),
                                ("odd", runs % 2 == 1), ("even", runs % 2 == 0)]:
                D, uc = crossnobis(X[idx][keep], runs[keep], g.orientation.to_numpy()[keep])
                assert np.allclose(uc, oris)
                emit(D, "crossnobis", split, X.shape[1])

            for k, (spec, pars, cv, space) in models.items():
                inroi = masking.apply_mask(cv, mask)[good] - masking.apply_mask(null_cv, mask)[good]
                pdf = pd.DataFrame({p: masking.apply_mask(im, mask)[good] for p, im in pars.items()})
                sel = (inroi > 0) & np.all(np.isfinite(pdf.to_numpy()), 1)
                if sel.sum() < MIN_VOX:
                    emit(np.full((23, 23), np.nan), k, "all", int(sel.sum()))
                    continue
                model = spec.cls(**spec.cls_kwargs)
                pdf = pdf.loc[sel, spec.save_params].reset_index(drop=True)
                if space == "orientation":
                    lev, tr = np.deg2rad(oris) % np.pi, np.deg2rad(g.orientation.to_numpy()) % np.pi
                else:
                    lev, tr = val, g.value.to_numpy()
                D, nv = model_rdm(model, pdf, lev, tr, X[idx][:, sel])
                emit(D, k, "all", nv)
            print(f"sub-{subject} {roi} ses-{ses}: {X.shape[1]} voxels", flush=True)

    res = pd.concat(rows)
    odir = Path(bids_folder) / "derivatives" / "followup_2026-10" / "a3_categorical_90" / f"sub-{subject}"
    odir.mkdir(parents=True, exist_ok=True)
    fn = odir / f"sub-{subject}_desc-a3_rdm.tsv"
    res.to_csv(fn, sep="\t", index=False)
    print(f"Wrote {fn}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("subject")
    p.add_argument("--bids-folder", default=BIDS_FOLDER)
    a = p.parse_args()
    main(a.subject, a.bids_folder)
