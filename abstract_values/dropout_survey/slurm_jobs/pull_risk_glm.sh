#!/bin/bash
#SBATCH --job-name=pull_risk_glm
#SBATCH --output=/home/gdehol/logs/pull_risk_glm_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=08:00:00
#SBATCH --account=zne.uzh

# Restore ds-risk's own GLMsingle outputs on the cluster from the department
# share (read-only: smbclient get/mget only). The cluster copies had been
# overwritten with ds-tmsrisk's GLM and were moved aside to
# <glm>.tmsrisk_duplicate on 2026-10-17.
#
# Needs a Kerberos ticket that a compute node can read: Gilles runs `kinit` on
# the login node, then
#   cp /tmp/krb5cc_$(id -u) /scratch/gdehol/.krb5cc && chmod 600 /scratch/gdehol/.krb5cc
#   sbatch --export=GLM=glm_stim1.denoise pull_risk_glm.sh

export KRB5CCNAME=/scratch/gdehol/.krb5cc
export TMPDIR=/scratch/gdehol/tmp
SHARE='//idnas32.uzh.ch/g_econ_department$'
SRC="projects/2020/dehollander_ruff_a neural account of trial-to-trial fluctuations in risk preference/data/ds-risk/derivatives"
DEST=/shares/zne.uzh/gdehol/ds-risk/derivatives
[ -e "$DEST/$GLM" ] || mkdir -p "$DEST/$GLM"

for i in 1 2 3; do
    smbclient --use-kerberos=required "$SHARE" -D "$SRC" \
        -c "prompt OFF; recurse ON; lcd $DEST; mget $GLM" </dev/null && break
    echo "attempt $i failed, retrying"; sleep 30
done

# completeness check against the share listing (file count and total bytes)
smbclient --use-kerberos=required "$SHARE" -D "$SRC/$GLM" -c "recurse ON; ls" </dev/null 2>/dev/null \
    | grep -E '^  [^ ]+ +[ANRH]+ +[0-9]+ ' | awk '{n++; s+=$3} END {print "share:   " n " files, " s " bytes"}'
(cd "$DEST/$GLM" && find . -type f -printf "%s\n") | awk '{n++; s+=$1} END {print "cluster: " n " files, " s " bytes"}'
echo "DONE"
