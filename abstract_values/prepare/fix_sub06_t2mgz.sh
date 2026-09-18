#!/bin/bash
#SBATCH --job-name=fix_t2mgz
#SBATCH --output=/home/gdehol/logs/fix_t2mgz_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --account=zne.uzh

# One-shot fix for sub-06: replace bad T2.mgz in FreeSurfer dir with a correctly-
# registered version resampled from fmriprep's preproc T2w.

set -euo pipefail

source /etc/profile.d/lmod.sh
module load apptainer/1.4.1

export APPTAINERENV_FS_LICENSE=$HOME/freesurfer/license.txt

BIDS=/shares/zne.uzh/gdehol/ds-abstractvalue
FS=$BIDS/derivatives/fmriprep/sourcedata/freesurfer/sub-06_ses-1

FMRIPREP_T2=$BIDS/derivatives/fmriprep/sub-06/ses-2/anat/sub-06_ses-2_desc-preproc_T2w.nii.gz
T1_MGZ=$FS/mri/T1.mgz

# Backup originals
for f in T2.mgz T2.norm.mgz T2.prenorm.mgz; do
    if [[ ! -f $FS/mri/${f}.bak ]]; then
        cp $FS/mri/$f $FS/mri/${f}.bak
        echo "Backed up $f"
    fi
done

# Resample fmriprep's preproc T2w onto FS conformed T1 grid using --regheader
# (--regheader = geometry is already correct, just reslice onto target voxel grid)
apptainer exec \
    -B $BIDS:/data \
    /shares/zne.uzh/containers/fmriprep-25.2.5 \
    mri_vol2vol \
        --mov $FMRIPREP_T2 \
        --targ $T1_MGZ \
        --regheader \
        --o $FS/mri/T2.new.mgz \
        --cubic

echo "=== Summary ==="
apptainer exec /shares/zne.uzh/containers/fmriprep-25.2.5 \
    mri_info $FS/mri/T2.new.mgz | head -20
