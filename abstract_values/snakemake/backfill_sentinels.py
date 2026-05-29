#!/usr/bin/env python3
"""Backfill Snakemake `.done` / `.touch` sentinels for legacy pipeline outputs.

When the Snakemake driver replaces the legacy `ingest_new_session.sh` chain,
all the actual derivative files already on disk from prior runs lack their
matching Snakemake sentinels (since the legacy scripts didn't know about them).
Without backfill, Snakemake's next run would re-execute every rule from
scratch — hours of cluster time on top of work that's already done.

This script walks `derivatives/` and, per rule, touches the sentinel iff
the actual output files for that rule are present in the expected count.
Conservative by construction: a missing real output never gets a sentinel.

Usage:
  python backfill_sentinels.py --bids-folder /shares/zne.uzh/gdehol/ds-abstractvalue
  python backfill_sentinels.py --dry-run   # report-only

Run from anywhere; --bids-folder defaults to the cluster path.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path


# Per-fit sentinel directories on the Snakemake side (underscored variant
# names — see aprf_r2() in Snakefile) → corresponding ENCODER dir on disk
# (hyphenated / dotted, per the encoder scripts' on-disk convention) plus a
# representative output filename pattern that confirms the fit completed.
#
# r2 desc: the encoder always writes one `desc-r2*_pe.nii.gz` regardless of
# parameterisation, so we use it as the single-file completeness witness for
# non-CV fits.
# cv desc: fit_*_cv.py writes the *mean* `desc-cvr2*_pe.nii.gz` at the end;
# its presence confirms all folds finished and were aggregated.
APRF_VARIANTS = {
    # snakemake_dir          encoder_dir              desc_pattern
    "aprf":                   ("aprf",                   "r2"),
    "aprf_cv":                ("aprf.cv",                "cvr2"),
    "aprf_session_shift":     ("aprf-session-shift",     "r2"),
    "aprf_session_shift_cv":  ("aprf-shift.cv",          "cvr2"),
    "aprf_weighted":          ("aprf-weighted",          "r2"),
    "aprf_weighted_cv":       ("aprf-weighted.cv",       "cvr2"),
    "aprf_gauss":             ("aprf-gauss",             "r2"),
    "aprf_gauss_cv":          ("aprf-gauss.cv",          "cvr2"),
    "aprf_gauss_session_shift":    ("aprf-gauss-session-shift", "r2"),
    "aprf_gauss_session_shift_cv": ("aprf-gauss-shift.cv",      "cvr2"),
    "vonmises":               ("vonmises",               "r2"),
    "vonmises_cv":            ("vonmises.cv",            "cvr2"),
}

# Surface-sampling sentinel directories (these match the ENCODER dir names,
# not the underscored Snakemake variant names — see sample_r2_to_surface
# rule's wildcard_constraints).
SURFACE_DIRS = list({d for d, _ in APRF_VARIANTS.values()})

# Decoding ROIs from config (decode_rois). We expect the .done sentinels to
# fan out over (roi, nv, lam).
DECODE_ROIS = ["BensonV1", "NPCr"]
DECODE_NV   = ["0", "50", "100", "250", "500"]
DECODE_LAM  = ["0.0", "0.1"]

# Expected-decoded value fanout.
EU_ROIS  = ["NPCr"]
EU_NV    = ["100", "fdr05"]
EU_NOISE = ["spherical", "residual"]

# Fisher information fanout.
FI_ROIS_VONMISES = ["BensonV1"]
FI_ROIS_APRF     = ["NPCr"]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
class Stats:
    def __init__(self):
        self.touched = 0
        self.skipped = 0
        self.already = 0

    def report(self, label):
        print(f"  {label}: touched={self.touched}, "
              f"already_present={self.already}, "
              f"skipped_no_outputs={self.skipped}")


def discover_subjects(bids_root: Path) -> list[str]:
    """Return subject labels (e.g. ['03', '07', 'pil01']) for any sub-*
    directory that has at least one ses-*/func subtree (real BIDS)."""
    out = []
    sub_re = re.compile(r"^sub-(.+)$")
    for p in sorted(bids_root.iterdir()):
        m = sub_re.match(p.name)
        if not (m and p.is_dir()):
            continue
        has_ses = any((p / d.name / "func").is_dir() for d in p.iterdir()
                      if re.match(r"^ses-\d+$", d.name))
        if has_ses:
            out.append(m.group(1))
    return out


def touch_sentinel(path: Path, dry_run: bool, stats: Stats) -> bool:
    """If `path` doesn't exist, touch it (and create its parent dir).
    Returns True if we touched a NEW sentinel, False if already present.
    Conservative: caller must have already verified that real outputs exist."""
    if path.exists():
        stats.already += 1
        return False
    if not dry_run:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    stats.touched += 1
    return True


def smooth_suffix_in_filename(smooth_in_dir: str) -> str:
    """The encoder embeds '_smoothed' inside the filename's desc field, not
    the dir; this helper translates 'smooth_in_dir' (the Snakemake suffix
    on the sentinel dir) to the matching filename suffix."""
    return "_smoothed" if smooth_in_dir == "_smoothed" else ""


# ─────────────────────────────────────────────────────────────────────────────
# Per-rule backfillers
# ─────────────────────────────────────────────────────────────────────────────
def backfill_fmriprep(deriv: Path, subject: str, dry_run: bool, stats: Stats):
    """fmriprep's HTML report IS the sentinel; nothing to backfill, but we
    still check existence to gate downstream rules' backfilling."""
    html = deriv / "fmriprep" / f"sub-{subject}.html"
    return html.exists()


def backfill_masks(deriv: Path, subject: str, dry_run: bool, stats: Stats):
    """ROI masks (create_roi_masks, create_npc_masks): the mask NIfTIs ARE
    the sentinels. Nothing to backfill, just report existence."""
    npc = deriv / "masks" / f"sub-{subject}" / "anat" / \
          f"sub-{subject}_space-T1w_desc-NPCr_mask.nii.gz"
    benson = deriv / "masks" / f"sub-{subject}" / "anat" / \
             f"sub-{subject}_space-T1w_hemi-LR_desc-BensonV1_mask.nii.gz"
    if npc.exists():    stats.already += 1
    if benson.exists(): stats.already += 1


def backfill_glmsingle(deriv: Path, subject: str, dry_run: bool, stats: Stats):
    """GLMsingle: sentinel `glmsingle{smooth}/sub-{S}/_done.touch`.
    Verify by checking the encoder's BOLD beta NIfTIs in func/ exist."""
    for smooth in ("", "_smoothed"):
        sub_dir = deriv / f"glmsingle{smooth}" / f"sub-{subject}"
        sentinel = sub_dir / "_done.touch"
        func = sub_dir / "func"
        # GLMsingle's output: sub-XX/func/ subtree with per-(ses,run) NIfTIs.
        # Conservative: require func/ exists and contains at least one .nii.gz.
        if func.is_dir() and any(func.rglob("*.nii.gz")):
            touch_sentinel(sentinel, dry_run, stats)
        elif not sentinel.exists():
            stats.skipped += 1


def backfill_aprf_fits(deriv: Path, subject: str, dry_run: bool, stats: Stats):
    """All aPRF-family fit sentinels (including CV and vonmises variants)."""
    for snake_dir, (enc_dir, desc) in APRF_VARIANTS.items():
        for smooth in ("", "_smoothed"):
            sentinel = (deriv / "encoding_models" / f"{snake_dir}{smooth}"
                        / f"sub-{subject}" / ".done")
            func = (deriv / "encoding_models" / f"{enc_dir}{smooth}"
                    / f"sub-{subject}" / "func")
            smooth_in_fn = smooth_suffix_in_filename(smooth)
            witness = func / (f"sub-{subject}_task-abstractvalue_space-T1w"
                              f"_desc-{desc}{smooth_in_fn}_pe.nii.gz")
            if witness.exists():
                touch_sentinel(sentinel, dry_run, stats)
            elif not sentinel.exists():
                stats.skipped += 1


def backfill_aggregate_cvr2(deriv: Path, subject: str, dry_run: bool,
                              stats: Stats):
    """CV aggregation sentinel `.cvr2_aggregated` lives under the *encoder*
    `.cv` dir (per the surface sampling rule's model_dir wildcard). Check the
    aggregated mean cvR² NIfTI is present."""
    for snake_dir, (enc_dir, desc) in APRF_VARIANTS.items():
        if desc != "cvr2":
            continue
        for smooth in ("", "_smoothed"):
            sentinel = (deriv / "encoding_models" / f"{enc_dir}{smooth}"
                        / f"sub-{subject}" / ".cvr2_aggregated")
            func = (deriv / "encoding_models" / f"{enc_dir}{smooth}"
                    / f"sub-{subject}" / "func")
            smooth_in_fn = smooth_suffix_in_filename(smooth)
            witness = func / (f"sub-{subject}_task-abstractvalue_space-T1w"
                              f"_desc-cvr2{smooth_in_fn}_pe.nii.gz")
            if witness.exists():
                touch_sentinel(sentinel, dry_run, stats)
            elif not sentinel.exists():
                stats.skipped += 1


def backfill_surface_sampling(deriv: Path, subject: str, dry_run: bool,
                                stats: Stats):
    """Surface sampling sentinel `.surface_sampled` under each encoder dir.
    Witness: a fsaverage GIfTI of the right desc."""
    for snake_dir, (enc_dir, desc) in APRF_VARIANTS.items():
        for smooth in ("", "_smoothed"):
            sentinel = (deriv / "encoding_models" / f"{enc_dir}{smooth}"
                        / f"sub-{subject}" / ".surface_sampled")
            func = (deriv / "encoding_models" / f"{enc_dir}{smooth}"
                    / f"sub-{subject}" / "func")
            smooth_in_fn = smooth_suffix_in_filename(smooth)
            # The surface sample writes fsaverage L+R GIfTIs.
            witness = func / (f"sub-{subject}_task-abstractvalue_hemi-L"
                              f"_space-fsaverage_desc-{desc}{smooth_in_fn}"
                              f"_pe.func.gii")
            if witness.exists():
                touch_sentinel(sentinel, dry_run, stats)
            elif not sentinel.exists():
                stats.skipped += 1


def backfill_decoding(deriv: Path, subject: str, dry_run: bool, stats: Stats):
    """decode_gabor / decode_value sentinels per (roi, nv, lam, smooth).
    Witness: at least one decoded TSV at derivatives/decoding/{kind}/."""
    for kind in ("gabor", "value"):
        for smooth in ("", "_smoothed"):
            for roi in DECODE_ROIS:
                for nv in DECODE_NV:
                    for lam in DECODE_LAM:
                        sentinel = (deriv / f"decoded_{kind}{smooth}"
                                    / f"sub-{subject}"
                                    / (f"sub-{subject}_roi-{roi}_nv-{nv}"
                                       f"_lam-{lam}.done"))
                        # Real TSV layout per decode_{gabor,value}.py:
                        # derivatives/decoding/{kind}/sub-XX/func/
                        # sub-XX_task-...desc-{kind}{smooth}_pe.tsv with
                        # filename markers mask-{roi}_nvoxels-{nv}_lambda-{lam}.
                        func = (deriv / "decoding" / kind / f"sub-{subject}"
                                / "func")
                        smooth_in_fn = smooth_suffix_in_filename(smooth)
                        pattern = (f"*mask-{roi}*nvoxels-{nv}*lambda-{lam}*"
                                   f"{smooth_in_fn}*decoded*.tsv")
                        if func.is_dir() and any(func.glob(pattern)):
                            touch_sentinel(sentinel, dry_run, stats)
                        elif not sentinel.exists():
                            stats.skipped += 1


def backfill_fisher_information(deriv: Path, subject: str,
                                  dry_run: bool, stats: Stats,
                                  sessions: list[str]):
    """compute_fisher_information (vonmises + aprf) sentinels per
    (roi, session, smooth)."""
    for kind, rois, src_dir in (
        ("vonmises", FI_ROIS_VONMISES, "vonmises"),
        ("aprf",     FI_ROIS_APRF,     "aprf"),
    ):
        for smooth in ("", "_smoothed"):
            for roi in rois:
                for ses in ["all"] + sessions:
                    sentinel = (deriv / f"fisher_information_{kind}{smooth}"
                                / f"sub-{subject}"
                                / (f"sub-{subject}_roi-{roi}_ses-{ses}.done"))
                    # Real TSV under encoder dir, ses-{ses} subdir for per-ses,
                    # func/ directly for ses-all.
                    if ses == "all":
                        base = (deriv / "encoding_models" / src_dir
                                / f"sub-{subject}" / "func")
                    else:
                        base = (deriv / "encoding_models" / src_dir
                                / f"sub-{subject}" / f"ses-{ses}" / "func")
                    smooth_in_fn = smooth_suffix_in_filename(smooth)
                    pattern = (f"*mask-{roi}*fisher*{smooth_in_fn}*.tsv")
                    if base.is_dir() and any(base.glob(pattern)):
                        touch_sentinel(sentinel, dry_run, stats)
                    elif not sentinel.exists():
                        stats.skipped += 1


def backfill_expected_decoded(deriv: Path, subject: str,
                                dry_run: bool, stats: Stats,
                                sessions: list[str]):
    """compute_expected_decoded_value_aprf sentinels per (roi, nv, noise,
    smooth). Witness: per-session TSV under aprf-session-shift/."""
    for smooth in ("", "_smoothed"):
        for roi in EU_ROIS:
            for nv in EU_NV:
                for noise in EU_NOISE:
                    sentinel = (deriv
                                / f"expected_decoded_value_aprf{smooth}"
                                / f"sub-{subject}"
                                / (f"sub-{subject}_roi-{roi}_nv-{nv}"
                                   f"_noise-{noise}.done"))
                    # The script writes per-session TSVs at:
                    # encoding_models/aprf-session-shift/sub-XX/ses-X/func/
                    # sub-XX_..._mask-{roi}_nvoxels-{nv}_*noise-{noise}*
                    # [_smoothed]_desc-expected_decoded_pe.tsv
                    found = False
                    for ses in sessions:
                        base = (deriv / "encoding_models"
                                / "aprf-session-shift" / f"sub-{subject}"
                                / f"ses-{ses}" / "func")
                        smooth_in_fn = smooth_suffix_in_filename(smooth)
                        # Noise-token: "noise-{noise}" appears for both
                        # spherical and residual in the filename schema.
                        pattern = (f"*mask-{roi}*nvoxels-{nv}*noise-{noise}*"
                                   f"{smooth_in_fn}*expected_decoded*.tsv")
                        if base.is_dir() and any(base.glob(pattern)):
                            found = True
                            break
                    if found:
                        touch_sentinel(sentinel, dry_run, stats)
                    elif not sentinel.exists():
                        stats.skipped += 1


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────
def session_indices(bids_root: Path, subject: str) -> list[str]:
    """1-based session numbers present on disk for the subject."""
    sub_dir = bids_root / f"sub-{subject}"
    if not sub_dir.is_dir():
        return []
    return sorted(
        m.group(1)
        for d in sub_dir.iterdir()
        if (m := re.match(r"^ses-(\d+)$", d.name))
        and (d / "func").is_dir()
    )


def main():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bids-folder",
                   default="/shares/zne.uzh/gdehol/ds-abstractvalue",
                   help="Cluster BIDS root.")
    p.add_argument("--dry-run", action="store_true",
                   help="Report-only; no sentinels touched.")
    p.add_argument("--subjects", nargs="+", default=None,
                   help="Restrict to these subjects (default: all on disk).")
    args = p.parse_args()

    bids = Path(args.bids_folder)
    deriv = bids / "derivatives"

    subjects = args.subjects or discover_subjects(bids)
    print(f"BIDS root: {bids}")
    print(f"Subjects:  {subjects}")
    print(f"Mode:      {'DRY-RUN' if args.dry_run else 'COMMIT'}")
    print()

    grand = Stats()
    for s in subjects:
        sessions = session_indices(bids, s)
        if not sessions:
            print(f"[sub-{s}] no sessions on disk — skipping")
            continue
        print(f"[sub-{s}]  sessions={sessions}")
        sub_stats = Stats()
        if not backfill_fmriprep(deriv, s, args.dry_run, sub_stats):
            print(f"  (no fmriprep output — skipping mask/GLM/encoding "
                  "sentinels)")
            continue
        backfill_masks(deriv, s, args.dry_run, sub_stats)
        backfill_glmsingle(deriv, s, args.dry_run, sub_stats)
        backfill_aprf_fits(deriv, s, args.dry_run, sub_stats)
        backfill_aggregate_cvr2(deriv, s, args.dry_run, sub_stats)
        backfill_surface_sampling(deriv, s, args.dry_run, sub_stats)
        backfill_decoding(deriv, s, args.dry_run, sub_stats)
        backfill_fisher_information(deriv, s, args.dry_run, sub_stats, sessions)
        if len(sessions) >= 2:
            backfill_expected_decoded(deriv, s, args.dry_run, sub_stats, sessions)
        sub_stats.report(f"sub-{s}")
        grand.touched += sub_stats.touched
        grand.already += sub_stats.already
        grand.skipped += sub_stats.skipped
        print()

    print("─" * 60)
    grand.report("TOTAL")
    if args.dry_run and grand.touched:
        print(f"\n(dry-run) would have touched {grand.touched} sentinels.")


if __name__ == "__main__":
    main()
