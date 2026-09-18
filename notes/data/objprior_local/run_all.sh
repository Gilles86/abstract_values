#!/bin/bash
# Local objective-prior sweep (cluster blocked by cephfs_emergency).
# Serialized so only ONE TF/Metal process runs at a time on the M1 GPU.
set -u
cd /Users/gdehol/git/abstract_values
export BIDS_FOLDER=/data/ds-abstractvalue
PY=~/mambaforge/envs/abstract_values/bin/python
LOGDIR=notes/data/objprior_local

# Wait for any in-flight decode (sub-03/04) to finish first.
while pgrep -f decode_value_modelcomp >/dev/null; do sleep 30; done

SUBS_ALL="03 04 05 06 07 08 09 10 13 14 pil01 pil02"
SUBS_REST="05 06 07 08 09 10 13 14 pil01 pil02"   # 03/04 already done

# 1) EU uncertainty sim: fwhm-shift, NPCr, 250 vox, 1000 sims, spherical.
for S in $SUBS_ALL; do
  for P in flat objective; do
    echo "===== EU sub-$S prior=$P $(date +%T) =====" >> $LOGDIR/eu.log
    PYTHONUNBUFFERED=1 $PY -u -m abstract_values.encoding_models.compute_expected_decoded_value_aprf \
      $S --model fwhm-shift --n-voxels 250 --n-simulations 1000 \
      --spherical-noise --prior $P --bids-folder /data/ds-abstractvalue \
      >> $LOGDIR/eu.log 2>&1
  done
done
echo "EU DONE $(date +%T)" >> $LOGDIR/eu.log

# 2) Remaining decode subjects (03/04 ran separately).
for S in $SUBS_REST; do
  for P in none objective; do
    echo "===== DEC sub-$S prior=$P $(date +%T) =====" >> $LOGDIR/decode.log
    PYTHONUNBUFFERED=1 $PY -u -m abstract_values.encoding_models.decode_value_modelcomp \
      $S --prior $P --bids-folder /data/ds-abstractvalue \
      >> $LOGDIR/decode.log 2>&1
  done
done
echo "DECODE DONE $(date +%T)" >> $LOGDIR/decode.log
echo "ALL_COMPLETE $(date +%T)" >> $LOGDIR/run_all.done
