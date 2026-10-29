#!/bin/bash
# Backfill one late participant into every MRI analysis that the Snakemake
# pipeline does not cover (ROI masks beyond Benson/NPC, bid / mix / quantile /
# permnull / run-split encoding models, per-ROI decoding and expected decoded
# values, crossnobis, lambda profiles, connective fields, follow-up analyses).
#
# Prerequisite: the Snakemake pipeline is done for the subject
# (derivatives/_snake/sub-<S>.done): fmriprep, GLMsingle, Benson/NPC masks and
# the core aprf / vonmises fits.
#
# Run ON THE LOGIN NODE (it only calls sbatch), from the repo root:
#   bash abstract_values/prepare/slurm_jobs/backfill_late_subject.sh 36
#   DRY_RUN=1 bash abstract_values/prepare/slurm_jobs/backfill_late_subject.sh 36
#
# Everything is chained with --dependency=afterok, so it runs unattended.
# Every job writes only sub-<S> outputs. A manifest of (job id, name) is written
# to ~/logs/backfill_sub-<S>_<timestamp>.tsv.
#
# Written for sub-36 (2026-10-29) from the submit lines that backfilled sub-35
# (sacct SubmitLine). Array-indexed job scripts (mix lambda, lambda sims, run
# split) are submitted as single tasks whose index points at <S> in the current
# aprf-session-shift subject listing.
set -eo pipefail

S=${1:?usage: backfill_late_subject.sh <subject label, e.g. 36>}
BIDS=/shares/zne.uzh/gdehol/ds-abstractvalue
D=$BIDS/derivatives
REPO=$HOME/git/abstract_values
CF_REPO=/scratch/gdehol/abstract_values_cf   # clean clone for connective fields
PY=$HOME/data/conda/envs/abstract_values/bin/python
EM=abstract_values/encoding_models/slurm_jobs
MASKS=$D/masks/sub-$S/anat
UNION=$MASKS/sub-${S}_space-T1w_desc-permnullunion_mask.nii.gz
STAMP=$(date +%Y%m%d_%H%M%S)
MANIFEST=$HOME/logs/backfill_sub-${S}_${STAMP}.tsv

cd "$REPO"
export TMPDIR=/scratch/$USER/tmp TMP=/scratch/$USER/tmp TEMP=/scratch/$USER/tmp
mkdir -p "$TMPDIR"
export KERAS_BACKEND=tensorflow PYTHONUNBUFFERED=1

[ -f "$D/_snake/sub-$S.done" ] || { echo "sub-$S: Snakemake pipeline not done"; exit 1; }

# index of S in the session-shift subject listing (what the array scripts use)
mapfile -t SUBS < <(ls "$D/encoding_models/aprf-session-shift" | grep '^sub-' | sed 's/^sub-//' | sort)
N=${#SUBS[@]}
IDX=-1
for i in "${!SUBS[@]}"; do [ "${SUBS[$i]}" = "$S" ] && IDX=$i; done
[ "$IDX" -ge 0 ] || { echo "sub-$S has no aprf-session-shift fit"; exit 1; }
echo "sub-$S: index $IDX of $N in aprf-session-shift"

# sbatch wrapper: sb <name> [sbatch args...] <script> ; prints the job id
sb () {
    local name=$1; shift
    if [ "${DRY_RUN:-0}" = "1" ]; then
        echo "DRY $name: sbatch $*" >&2; echo "0"; return
    fi
    local id
    id=$(sbatch --parsable -A zne.uzh --partition=standard --exclude=u24-chaihm0-651 \
                --job-name="bf${S}_${name}" "$@")
    id=${id%%;*}
    printf '%s\t%s\n' "$id" "$name" >> "$MANIFEST"
    echo "$id"
}
dep () { [ "${DRY_RUN:-0}" = "1" ] && { echo "--dependency=afterok:0"; return; }; echo "--dependency=afterok:$(IFS=:; echo "$*")"; }

# ── 1. masks ────────────────────────────────────────────────────────────────
# Wang/Benson atlas ROIs, M1, value clusters (+LOSO), vmPFC aparc, permnull
# union. The group value-cluster labels (2026-10-02) were drawn without sub-31,
# -32, -35, -36, so their LOSO labels are plain copies (see loso/README).
MASK_ID=$(sb masks -c2 --mem=24G --time=01:00:00 -o "$HOME/logs/bf${S}_masks_%j.txt" --wrap "
set -e; cd $REPO; export PYTHONPATH=$REPO
SM=$D/surface_masks; mkdir -p \$SM/loso/sub-$S
for c in valueCS valueV3aIPS0; do
  cp -n \$SM/desc-\${c}_L_space-fsaverage_hemi-lh.label.gii \$SM/loso/sub-$S/desc-\${c}loso_L_space-fsaverage_hemi-lh.label.gii
done
grep -q 'sub-$S' \$SM/loso/README_new_subjects.txt || echo 'sub-$S: plain copies as well (group clusters drawn without sub-$S). Written $(date +%F).' >> \$SM/loso/README_new_subjects.txt
$PY -m abstract_values.surface.project_fsaverage_rois $S --bids-folder $BIDS
$PY -m abstract_values.prepare.make_vmpfc_aparc_mask $S --bids-folder $BIDS
$PY abstract_values/encoding_models/permnull_select.py $S --bids-folder $BIDS --union-mask-only
ls -la $MASKS")
NPC3_ID=$(sb npc3 --time=00:40:00 --export=ALL,PARTICIPANT_LABEL=$S,ROIS=NPC3 \
             abstract_values/prepare/slurm_jobs/create_extra_roi_masks.sh)
DM=$(dep "$MASK_ID")
DMN=$(dep "$MASK_ID" "$NPC3_ID")

# ── 2. CV fit -> aggregate cvR2 -> fsaverage/fsnative surface ─────────────────
cv_chain () {   # cv_chain <name> <cv dir> <smoothed> <sbatch args for the fit...>
    local name=$1 dir=$2 sm=$3; shift 3
    local f a
    f=$(sb "$name" "$@")
    a=$(sb "agg_${name}" $(dep "$f") --export=ALL,PARTICIPANT_LABEL=$S,MODEL_DIR=$dir,SMOOTHED=$sm \
           $EM/aggregate_cvr2.sh)
    sb "surf_${name}" -c2 --mem=8G -t 30 $(dep "$a") \
       --export=ALL,PARTICIPANT_LABEL=$S,MODEL_DIR=$dir,DESC=cvr2,SMOOTHED=$sm \
       abstract_values/surface/slurm_jobs/sample_r2_to_surface.sh > /dev/null
}
# reported-value twins (smoothed betas only, as for the cohort)
for m in standard-bid:aprf-bid.cv standard-bidmatched:aprf-bidmatched.cv \
         linear-bid:aprf-linear-bid.cv linear-bidmatched:aprf-linear-bidmatched.cv \
         null-bidmatched:aprf-null-bidmatched.cv \
         session-shift-bid:aprf-session-shift-bid.cv \
         session-shift-bidmatched:aprf-session-shift-bidmatched.cv \
         vonmises-prf-bidmatched:vonmises-prf-bidmatched.cv; do
    cv_chain "cv_${m%%:*}" "${m#*:}" 1 --export=ALL,PARTICIPANT_LABEL=$S,MODEL=${m%%:*},SMOOTHED=1 $EM/fit_aprf_cv.sh
done
for sp in bid bidmatched; do
    cv_chain "cvw_$sp" "aprf-weighted-$sp.cv" 1 --export=ALL,PARTICIPANT_LABEL=$S,SPACE=$sp,SMOOTHED=1 \
        $EM/fit_aprf_weighted_cv.sh
done
for sm in 0 1; do
    cv_chain "cv_vmprf_$sm" vonmises-prf.cv $sm --export=ALL,PARTICIPANT_LABEL=$S,MODEL=vonmises-prf,SMOOTHED=$sm \
        $EM/fit_aprf_cv.sh
done
sb fit_vmprf --export=ALL,PARTICIPANT_LABEL=$S,MODEL=vonmises-prf $EM/fit_aprf.sh > /dev/null

# vonmises preferred orientation (cos/sin) + missing surface samples
sb prefori_surf -c2 --mem=16G --time=01:30:00 -o "$HOME/logs/bf${S}_prefori_surf_%j.txt" --wrap "
set -e; cd $REPO; export PYTHONPATH=$REPO
$PY -m abstract_values.encoding_models.compute_preferred_orientation $S --bids-folder $BIDS
$PY -m abstract_values.encoding_models.compute_preferred_orientation $S --smoothed --bids-folder $BIDS
for sm in 0 1; do
  for d in orient_cos orient_sin; do PARTICIPANT_LABEL=$S MODEL_DIR=vonmises DESC=\$d SMOOTHED=\$sm bash abstract_values/surface/slurm_jobs/sample_r2_to_surface.sh; done
  PARTICIPANT_LABEL=$S MODEL_DIR=aprf-null.cv DESC=cvr2 SMOOTHED=\$sm bash abstract_values/surface/slurm_jobs/sample_r2_to_surface.sh
done" > /dev/null

# tSNR / mean-EPI QA maps (derivatives/qa/tsnr)
sb tsnr --export=ALL,PARTICIPANT_LABEL=$S abstract_values/visualize/slurm_jobs/collect_vmpfc_dropout.sh > /dev/null

# ── 3. ROI-restricted encoding models (need the permnull union mask) ─────────
for sm in 0 1; do
    sb "pn_ss_$sm" $DM --time=03:00:00 --export=ALL,PARTICIPANT_LABEL=$S,MODEL=session-shift,SMOOTHED=$sm $EM/permnull_select.sh > /dev/null
    for m in gaussian gauss-session-shift quantile quantile-wrapped; do
        sb "pn_${m}_$sm" $DM --time=02:00:00 --export=ALL,PARTICIPANT_LABEL=$S,MODEL=$m,SMOOTHED=$sm $EM/permnull_select.sh > /dev/null
    done
    sb "pn_linear_$sm" $DM --time=00:30:00 --export=ALL,PARTICIPANT_LABEL=$S,MODEL=linear,SMOOTHED=$sm $EM/permnull_select.sh > /dev/null
    for m in quantile quantile-wrapped; do
        sb "cv_${m}_$sm" $DM --time=02:00:00 --mem=16G \
           --export=ALL,PARTICIPANT_LABEL=$S,MODEL=$m,SMOOTHED=$sm,MASK=$UNION $EM/fit_aprf_cv.sh > /dev/null
    done
    for t in m050 m025 p000 p025 p050 p075 p100 p125 p150; do
        sb "mxfit_${t}_$sm" $DM --time=00:30:00 \
           --export=ALL,PARTICIPANT_LABEL=$S,MODEL=mix-$t,SMOOTHED=$sm,MASK=$UNION $EM/fit_aprf.sh > /dev/null
        sb "mxcv_${t}_$sm" $DM --time=01:30:00 \
           --export=ALL,PARTICIPANT_LABEL=$S,MODEL=mix-$t,SMOOTHED=$sm,MASK=$UNION $EM/fit_aprf_cv.sh > /dev/null
    done
done
# session-shift on odd / even runs (array task 2i+1 = odd, 2i+2 = even)
sb runsplit $DM --time=00:30:00 --array=$((2 * IDX + 1)),$((2 * IDX + 2)) $EM/fit_aprf_runsplit.sh > /dev/null

# mix-lambda (observed + simulated), per-ROI width / grid variants
ARR=(); for c in 0 1 2 3 4 5 6; do ARR+=($((c * N + IDX + 1))); done
sb mixlam $DM --time=03:00:00 --array=$(IFS=,; echo "${ARR[*]}") $EM/fit_mix_lambda.sh > /dev/null
ARR=(); for c in $(seq 0 8); do for r in 0 1 2 3 4 5; do ARR+=($(((c * N + IDX) * 6 + r + 1))); done; done
sb mixlam_width $DM --time=03:00:00 --array=$(IFS=,; echo "${ARR[*]}") \
   --export=ALL,MODEL=width,PER_ROI=1 $EM/fit_mix_lambda.sh > /dev/null
ARR=(); for c in 0 1 2 3 4; do for r in 0 1 2 3 4 5; do ARR+=($(((c * N + IDX) * 6 + r + 1))); done; done
sb mixlam_grid $DM --time=10:00:00 --array=$(IFS=,; echo "${ARR[*]}") \
   --export=ALL,MODEL=grid,PER_ROI=1 $EM/fit_mix_lambda.sh > /dev/null
# end-to-end lambda recovery sims (8 conditions) and their cvR2 profiles (6)
ARR=(); for c in $(seq 0 7); do ARR+=($((c * N + IDX + 1))); done
sb sim_lambda $DM --time=00:30:00 --array=$(IFS=,; echo "${ARR[*]}") $EM/simulate_lambda_pipeline.sh > /dev/null
ARR=(); for c in $(seq 0 5); do ARR+=($((c * N + IDX + 1))); done
sb lamprof_sim $DM --time=00:30:00 --array=$(IFS=,; echo "${ARR[*]}") $EM/fit_lambda_profile_sim.sh > /dev/null
# lambda profiles (ROI, bid axes, power axis, whole brain) -> fsaverage
LB=()
for sm in 0 1; do
    sb "lamprof_$sm" $DM --time=00:30:00 --export=ALL,PARTICIPANT_LABEL=$S,SMOOTHED=$sm $EM/fit_lambda_profile.sh > /dev/null
    sb "lamprof_bid_$sm" $DM --time=00:30:00 --export=ALL,PARTICIPANT_LABEL=$S,SMOOTHED=$sm,BIDS=1 $EM/fit_lambda_profile.sh > /dev/null
    sb "gamprof_$sm" $DM --time=00:30:00 --export=ALL,PARTICIPANT_LABEL=$S,SMOOTHED=$sm,POWER=1 $EM/fit_lambda_profile.sh > /dev/null
    LB+=($(sb "lambrain_$sm" $DM --mem=32G --time=02:00:00 --export=ALL,PARTICIPANT_LABEL=$S,SMOOTHED=$sm,WHOLE_BRAIN=1 $EM/fit_lambda_profile.sh))
done
sb lamsurf $(dep "${LB[@]}") --export=ALL,PARTICIPANT_LABEL=$S abstract_values/surface/slurm_jobs/sample_lambda_to_surface.sh > /dev/null

# ── 4. decoding / expected decoded values per ROI ──────────────────────────────
roi_mask () {   # roi_mask <desc> -> path (Benson masks carry hemi-LR)
    case $1 in Benson*) echo "$MASKS/sub-${S}_space-T1w_hemi-LR_desc-$1_mask.nii.gz" ;;
               *)       echo "$MASKS/sub-${S}_space-T1w_desc-$1_mask.nii.gz" ;; esac
}
for r in BensonV2 BensonV3 IPS0 IPS12 IPS345 V3ab M1l; do
    for sp in 0 1; do
        sb "dec_${r}_$sp" $DM --time=02:00:00 \
           --export=ALL,PARTICIPANT_LABEL=$S,MASK=$(roi_mask $r),MASK_DESC=$r,SPHERICAL=$sp,N_VOXELS=100 \
           $EM/decode_value.sh > /dev/null
    done
done
sb dec_vmPFC $DM --time=02:00:00 \
   --export=ALL,PARTICIPANT_LABEL=$S,MASK=$(roi_mask vmPFCOFCaparc),MASK_DESC=vmPFCOFCaparc,SPHERICAL=1,N_VOXELS=100 \
   $EM/decode_value.sh > /dev/null
for r in BensonV1 IPS0 IPS12 IPS345 vmPFCOFCaparc; do
    sb "declin_$r" $DM --time=02:00:00 \
       --export=ALL,PARTICIPANT_LABEL=$S,MASK=$(roi_mask $r),MASK_DESC=$r,SPHERICAL=1,N_VOXELS=100,MODEL=linear \
       $EM/decode_value.sh > /dev/null
done
# expected decoded value (weighted basis, per-session weights) and orientation
for rh in BensonV1:LR BensonV2:LR BensonV3:LR hV4:None IPS0:None IPS12:None LO:None \
          NPCl:None NPCr:None TO:None V3ab:None; do
    for p in presented flat mapping; do
        sb "euw_${rh%%:*}_$p" $DM --time=00:30:00 \
           --export=ALL,PARTICIPANT_LABEL=$S,ROI=${rh%%:*},HEMI=${rh#*:},PER_SESSION=1,DECODE_PRIOR=$p \
           $EM/compute_expected_decoded_value_weighted.sh > /dev/null
    done
    sb "euvm_${rh%%:*}" $DM --time=00:30:00 --export=ALL,PARTICIPANT_LABEL=$S,ROI=${rh%%:*},HEMI=${rh#*:} \
       $EM/compute_expected_decoded_orientation_vonmises.sh > /dev/null
done
for r in IPS345 M1l; do
    sb "euw_${r}_presented" $DM --time=00:30:00 --export=ALL,PARTICIPANT_LABEL=$S,ROI=$r,HEMI=None,PER_SESSION=1 \
       $EM/compute_expected_decoded_value_weighted.sh > /dev/null
done

# ── 5. geometry, follow-ups ───────────────────────────────────────────────────
sb crossnobis $DM --time=00:30:00 --export=ALL,PARTICIPANT_LABEL=$S abstract_values/geometry/slurm_jobs/crossnobis.sh > /dev/null
sb cn_power $DM --time=00:30:00 --export=ALL,PARTICIPANT_LABEL=$S abstract_values/geometry/slurm_jobs/power_mapping_change.sh > /dev/null
A3=analyses/followup_2026-10/a3_categorical_90/slurm_jobs/compute_rdms.sh
sb a3_rdms $DM --export=ALL,PARTICIPANT_LABEL=$S $A3 > /dev/null
sb a3_controls $DM --export=ALL,PARTICIPANT_LABEL=$S,SCRIPT=compute_controls.py $A3 > /dev/null
sb a4 $DM --time=10:00:00 --export=ALL,PARTICIPANT_LABEL=$S analyses/followup_2026-10/a4_tuning_shape/fit_tuning_shape.sh > /dev/null
sb a5 $DM --time=00:30:00 --export=ALL,PARTICIPANT_LABEL=$S analyses/followup_2026-10/a5_coupling_90/a5_coupling_90.sh > /dev/null
for desc in gabor response; do
    sb "a1_M1_$desc" $DM --time=04:00:00 --export=ALL,SUBJECT=$S,DESC=$desc $EM/fit_motor_cv.sh > /dev/null
    sb "a1_preSMA_$desc" --time=04:00:00 --export=ALL,SUBJECT=$S,DESC=$desc,ROI_SET=preSMA $EM/fit_motor_cv.sh > /dev/null
done

# ── 6. connective fields (clean clone, see project memory) ────────────────────
CFJ=$CF_REPO/abstract_values/connective_fields/slurm_jobs
I24="PROJECTION=iem N_BASIS=24 KAPPA=16"
V=()
for l in 0 1 2; do V+=("LAGS=$l" "$I24 LAGS=$l"); done
V+=("LAGS=2 SHAM_OUTER=1" "$I24 LAGS=2 SHAM_OUTER=1")
for t in IPS0+IPS12+IPS345 IPS0 IPS12 IPS345; do
    for l in 0 1 2; do V+=("$I24 TARGET_ROI=$t LAGS=$l"); done
    for l in 0 1; do V+=("$I24 TARGET_ROI=$t TEMPLATE=mixture LAGS=$l"); done
done
V+=("$I24 TARGET_ROI=IPS0+IPS12+IPS345 LAGS=2 SHAM_OUTER=1")
for t in IPS0+IPS12+IPS345 IPS345; do
    for kk in "36 32" "48 64"; do set -- $kk
        for l in 0 2; do V+=("PROJECTION=iem N_BASIS=$1 KAPPA=$2 TARGET_ROI=$t LAGS=$l"); done
    done
done
for g in "GAIN=1" "DRIVE=low" "DRIVE=high"; do V+=("$I24 TARGET_ROI=IPS0+IPS12+IPS345 TEMPLATE=mixture $g"); done
VN=()   # NPC1-3 targets (need the NPC3 mask)
for t in IPS0 IPS0+IPS12+IPS345 IPS12 IPS345 NPC1 NPC1+NPC2+NPC3 NPC2 NPC3; do
    for x in "TEMPLATE=basis LAGS=0" "TEMPLATE=basis LAGS=1" "NPC_CLASS=value LAGS=1" \
             "NPC_CLASS=orientation LAGS=1" "NPC_CLASS=bell LAGS=1" "NPC_CLASS=basis LAGS=1"; do
        VN+=("$I24 TARGET_ROI=$t $x")
    done
done
for t in NPC1 NPC2 NPC3 NPC1+NPC2+NPC3; do for l in 0 1; do VN+=("$I24 TARGET_ROI=$t LAGS=$l"); done; done
export VARIANTS
VARIANTS=$(IFS=';'; echo "${V[*]}")
sb cf_coupling $DM --time=03:00:00 --export=ALL,PARTICIPANT_LABEL=$S,REPO=$CF_REPO $CFJ/test_coupling_variants.sh > /dev/null
VARIANTS=$(IFS=';'; echo "${VN[*]}")
sb cf_coupling_npc $DMN --time=03:00:00 --export=ALL,PARTICIPANT_LABEL=$S,REPO=$CF_REPO $CFJ/test_coupling_variants.sh > /dev/null
unset VARIANTS
P=("TARGET_ROI=IPS0" "TARGET_ROI=IPS12" "TARGET_ROI=IPS345")
for l in 0 1 2; do P+=("TARGET_ROI=IPS0+IPS12+IPS345 LAGS=$l"); done
P+=("TARGET_ROI=IPS0+IPS12+IPS345 N_BASIS=36 KAPPA=32" "TARGET_ROI=IPS0+IPS12+IPS345 N_BASIS=48 KAPPA=64")
for s in all top25 top50; do P+=("TARGET_ROI=IPS0+IPS12+IPS345 SELECTION=$s"); done
P+=("TARGET_ROI=NPC1" "TARGET_ROI=NPC2" "TARGET_ROI=NPC3"
    "TARGET_ROI=NPC1+NPC2+NPC3 LAGS=0" "TARGET_ROI=NPC1+NPC2+NPC3 LAGS=1")
export PEAKS
PEAKS=$(IFS=';'; echo "${P[*]}")
sb cf_peaks $DMN --time=04:00:00 -c4 --mem=16G -o "$HOME/logs/bf${S}_cf_peaks_%j.txt" \
   --export=ALL,PARTICIPANT_LABEL=$S,REPO=$CF_REPO --wrap "
fail=0; IFS=';' read -ra PS <<< \"\$PEAKS\"
for p in \"\${PS[@]}\"; do echo \"=== \$p\"; env \$p bash $CFJ/peaks.sh || fail=1; done; exit \$fail" > /dev/null
unset PEAKS
for sm in 0 1; do
    sb "tuncon_$sm" $DM --export=ALL,PARTICIPANT_LABEL=$S,SMOOTHED=$sm,REPO=$CF_REPO \
       abstract_values/connective_fields/slurm_jobs/fit_tuning_connectivity.sh > /dev/null
done

echo "Submitted $(wc -l < "$MANIFEST" 2>/dev/null || echo 0) jobs; manifest: $MANIFEST"
echo "masks job: $MASK_ID   NPC3 job: $NPC3_ID"
