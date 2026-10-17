#!/bin/bash
# LOGIN NODE ONLY (needs the Kerberos ticket in /tmp/krb5cc_<uid>): pull the ds-risk
# GLMsingle outputs (glm_stim1.denoise, ses-3t2 / ses-7t2) from the department share
# into /scratch, read-only (smbclient get only).  The cluster copy under
# gdehol/ds-risk/derivatives/glm_stim1.denoise is a duplicate of ds-tmsrisk's, not ds-risk.
#   nohup bash fetch_risk_glm.sh > ~/logs/fetch_risk_glm.txt 2>&1 &
SHARE='//idnas32.uzh.ch/g_econ_department$'
SRC="projects/2020/dehollander_ruff_a neural account of trial-to-trial fluctuations in risk preference/data/ds-risk/derivatives/glm_stim1.denoise"
DST=/scratch/gdehol/dropout_survey/ds-risk/derivatives/glm_stim1.denoise
for ses in 3t2 7t2; do
    list=/scratch/gdehol/dropout_survey/subjects_risk${ses:0:2}.txt
    for s in $(cat $list); do
        d=$DST/sub-$s/ses-$ses/func
        mkdir -p $d
        for f in TYPED_FITHRF_GLMDENOISE_RR.npy sub-${s}_ses-${ses}_task-task_space-T1w_desc-stims1_pe.nii.gz; do
            [ -s $d/$f ] && continue
            smbclient --use-kerberos=required "$SHARE" \
                -c "cd \"$SRC/sub-$s/ses-$ses/func\"; lcd $d; get $f" 2>&1 | grep -v -e deprecated -e Password
        done
        echo "$(date +%T) sub-$s ses-$ses $(ls $d | wc -l) files"
    done
done
echo done
