#!/bin/bash
# Build the local pycortex surface bundle for one subject, end to end.
#
# Everything downstream of fmriprep that makes a subject *viewable* lives on
# the Mac, not the cluster: the pycortex filestore and the flat maps are
# local, so this cannot be a snakemake rule. Run it once per subject after
# the cluster pipeline has finished.
#
#   bash abstract_values/prepare/build_surface_bundle.sh 29
#   bash abstract_values/prepare/build_surface_bundle.sh 29 --flatten   # submit autoflatten
#
# Steps, each skipped when its output already exists:
#
#   1. Sample aPRF parameters + the null model to the surface (cluster).
#      The snakemake graph only runs sample_r2_to_surface, so mode / fwhm /
#      amplitude and aprf-null.cv are otherwise never on any surface.
#   2. Check the FreeSurfer subject has autoflatten patches. Without them
#      pycortex has no flat map and you get inflated-only views. Flattening
#      is ~35 min on the cluster, so it is opt-in via --flatten.
#   3. Rsync FreeSurfer recon + surfaces + patches to local.
#   4. Import the subject into the pycortex filestore, and import_flat.
#   5. Build the webgl bundle into derivatives/qa/webgl/sub-<subject>/.
#
# Then browse every subject processed so far with:
#   python -m abstract_values.visualize.webshow_surface_maps --serve-all

set -euo pipefail

SUBJECT="${1:-}"
if [ -z "$SUBJECT" ]; then
    echo "usage: $0 <subject-label> [--flatten] [--reimport] [--session N]" >&2
    exit 1
fi
shift

DO_FLATTEN=0
REIMPORT=0
SESSION=1
while [ $# -gt 0 ]; do
    case "$1" in
        --flatten) DO_FLATTEN=1; shift ;;
        --reimport) REIMPORT=1; shift ;;
        --session) SESSION="$2"; shift 2 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

CLUSTER=sciencecluster
REMOTE_BIDS=/shares/zne.uzh/gdehol/ds-abstractvalue
LOCAL_BIDS=/data/ds-abstractvalue
FS_REL=derivatives/fmriprep/sourcedata/freesurfer
FS_SUBJ="sub-${SUBJECT}_ses-${SESSION}"
PYCORTEX_PY=$HOME/mambaforge/envs/pycortex2/bin/python
REMOTE_PY='$HOME/data/conda/envs/abstract_values/bin/python'

say() { printf '\n=== %s ===\n' "$*"; }

# ── 1. surface sampling (cluster) ────────────────────────────────────────────
say "1/5  surface sampling for sub-${SUBJECT}"
NEED_SAMPLING=$(ssh -T $CLUSTER "ls ${REMOTE_BIDS}/derivatives/encoding_models/aprf/sub-${SUBJECT}/func/*fsnative*desc-mode_pe.func.gii 2>/dev/null | wc -l" | tr -d ' ')
NEED_NULL=$(ssh -T $CLUSTER "ls ${REMOTE_BIDS}/derivatives/encoding_models/aprf-null.cv/sub-${SUBJECT}/func/*fsnative*cvr2_pe.func.gii 2>/dev/null | wc -l" | tr -d ' ')

if [ "$NEED_SAMPLING" = "0" ] || [ "$NEED_NULL" = "0" ]; then
    echo "sampling on the cluster (aPRF parameters + null model)..."
    ssh -T $CLUSTER "cd \$HOME/git/abstract_values && \
      export FREESURFER_HOME=/shares/zne.uzh/containers/fmriprep-25.2.3/opt/freesurfer && \
      export PATH=\"\$FREESURFER_HOME/bin:\$PATH\" && \
      export FS_LICENSE=/shares/zne.uzh/containers/freesurfer/license.txt && \
      TMPDIR=/scratch/gdehol srun -c4 --mem 16G --time 60 --account=zne.uzh bash -c '
        for SM in \"\" \"--smoothed\"; do
          BIDS_FOLDER=${REMOTE_BIDS} PYTHONUNBUFFERED=1 ${REMOTE_PY} -u \
            -m abstract_values.surface.sample_aprf_to_surface ${SUBJECT} --session ${SESSION} \$SM
          BIDS_FOLDER=${REMOTE_BIDS} PYTHONUNBUFFERED=1 ${REMOTE_PY} -u \
            -m abstract_values.surface.sample_r2_to_surface --subjects ${SUBJECT} \
            --models aprf-null.cv --desc cvr2 --session ${SESSION} \$SM
        done'"
else
    echo "already sampled — skipping"
fi

# ── 2. flat patches (cluster, slow) ─────────────────────────────────────────
say "2/5  flat patches"
HAS_PATCH=$(ssh -T $CLUSTER "ls ${REMOTE_BIDS}/${FS_REL}/${FS_SUBJ}/surf/lh.autoflatten.flat.patch.3d 2>/dev/null | wc -l" | tr -d ' ')
if [ "$HAS_PATCH" = "0" ]; then
    if [ "$DO_FLATTEN" = "1" ]; then
        echo "submitting autoflatten (~35 min; rerun this script when it finishes)"
        ssh -T $CLUSTER "cat > /tmp/autoflatten_${SUBJECT}.sh <<EOF
#!/bin/bash
#SBATCH --job-name=autoflat_${SUBJECT}
#SBATCH --output=/home/gdehol/logs/autoflatten_${SUBJECT}_%j.txt
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --account=zne.uzh
export FREESURFER_HOME=/shares/zne.uzh/containers/fmriprep-25.2.3/opt/freesurfer
export PATH=\\\$FREESURFER_HOME/bin:\\\$PATH
export FS_LICENSE=\\\$HOME/freesurfer/license.txt
export SUBJECTS_DIR=${REMOTE_BIDS}/${FS_REL}
source ~/data/miniforge3/etc/profile.d/conda.sh
conda activate autoflatten
time autoflatten \\\$SUBJECTS_DIR/${FS_SUBJ} --parallel --overwrite
EOF
sbatch /tmp/autoflatten_${SUBJECT}.sh"
        echo
        echo "Autoflatten submitted. The bundle built below will be INFLATED-ONLY."
        echo "Re-run this script once the job finishes to get the flatten view."
    else
        echo "no flat patches for ${FS_SUBJ} — bundle will be inflated-only."
        echo "pass --flatten to submit the autoflatten job (~35 min)."
    fi
else
    echo "flat patches present"
fi

# ── 3. sync to local ────────────────────────────────────────────────────────
say "3/5  syncing FreeSurfer + surfaces to local"
mkdir -p "${LOCAL_BIDS}/${FS_REL}/${FS_SUBJ}"
rsync -a -e "ssh -T -o ServerAliveInterval=30 -o ServerAliveCountMax=3" \
    "${CLUSTER}:${REMOTE_BIDS}/${FS_REL}/${FS_SUBJ}/" \
    "${LOCAL_BIDS}/${FS_REL}/${FS_SUBJ}/"

for M in aprf aprf.cv aprf-null.cv aprf-linear.cv vonmises.cv; do
    SRC="${REMOTE_BIDS}/derivatives/encoding_models/${M}/sub-${SUBJECT}/func/"
    DST="${LOCAL_BIDS}/derivatives/encoding_models/${M}/sub-${SUBJECT}/func/"
    if ssh -T $CLUSTER "test -d ${SRC}"; then
        mkdir -p "$DST"
        rsync -a --include='*.gii' --exclude='*' \
            -e "ssh -T -o ServerAliveInterval=30" "${CLUSTER}:${SRC}" "$DST"
    fi
done

# fmriprep anat is needed by import_freesurfer_subject (T1w, xfm, boldref)
rsync -a --include='*/' --include='*_desc-preproc_T1w.nii.gz' \
    --include='*from-fsnative_to-T1w*' --include='*space-T1w_boldref.nii.gz' \
    --exclude='*' -e "ssh -T -o ServerAliveInterval=30" \
    "${CLUSTER}:${REMOTE_BIDS}/derivatives/fmriprep/sub-${SUBJECT}/" \
    "${LOCAL_BIDS}/derivatives/fmriprep/sub-${SUBJECT}/"

# ── 4. pycortex import ──────────────────────────────────────────────────────
say "4/5  importing into pycortex"
# import_subj on an existing subject prompts, and answering yes DELETES the
# filestore entry — flat maps included. Re-importing after a 35 min flatten
# would silently throw it away, so only import when the subject is new.
ALREADY=$($PYCORTEX_PY -c "import cortex,sys; \
    sys.stdout.write('1' if 'abstractvalue.sub-${SUBJECT}' in cortex.db.subjects else '0')" 2>/dev/null || echo 0)
if [ "$ALREADY" = "1" ] && [ "$REIMPORT" = "0" ]; then
    echo "abstractvalue.sub-${SUBJECT} already in the filestore — skipping import"
    echo "(pass --reimport to rebuild it; this discards existing flat maps)"
else
    $PYCORTEX_PY -m abstract_values.visualize.import_freesurfer_subject \
        "$SUBJECT" "$SESSION" --fmriprep-deriv fmriprep
fi

HAS_FLAT=$($PYCORTEX_PY -c "import cortex,os,sys; \
    d=os.path.join(cortex.database.default_filestore,'abstractvalue.sub-${SUBJECT}','surfaces'); \
    sys.stdout.write('1' if os.path.exists(os.path.join(d,'flat_lh.gii')) else '0')" 2>/dev/null || echo 0)
if [ "$HAS_FLAT" = "1" ]; then
    echo "flat map already imported — skipping"
elif [ -f "${LOCAL_BIDS}/${FS_REL}/${FS_SUBJ}/surf/lh.autoflatten.flat.patch.3d" ]; then
    echo "importing flat map ..."
    # NB: import_flat appends '.flat' to `patch` itself — passing
    # "autoflatten.flat" looks for a nonexistent ...flat.flat.patch.3d.
    $PYCORTEX_PY - "$FS_SUBJ" "$SUBJECT" "${LOCAL_BIDS}/${FS_REL}" <<'PYEOF'
import sys
from cortex import freesurfer
fs_subj, subject, fs_dir = sys.argv[1:4]
freesurfer.import_flat(fs_subj, patch='autoflatten', hemis=['lh', 'rh'],
                       cx_subject=f'abstractvalue.sub-{subject}',
                       freesurfer_subject_dir=fs_dir, auto_overwrite=True)
PYEOF
else
    echo "no local flat patch — skipping import_flat"
fi

# ── 5. build the bundle ─────────────────────────────────────────────────────
say "5/5  building the webgl bundle"
$PYCORTEX_PY -m abstract_values.visualize.webshow_surface_maps \
    "$SUBJECT" --both-smoothing

cat <<EOF

Done. Browse every processed subject with:

    python -m abstract_values.visualize.webshow_surface_maps --serve-all

EOF
