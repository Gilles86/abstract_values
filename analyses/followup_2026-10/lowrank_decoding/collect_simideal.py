#!/usr/bin/env python3
"""Collect simulate_ideal_decoding.py outputs (run on the cluster).

  derivatives/decoding_simideal/ideal_all.tsv.gz            all ideal-observer trials
  derivatives/decoding_simideal/pipeline_<code>.tsv.gz      pipeline posteriors summarised
                                                            on both axes (mapping_profiles_collect)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mapping_profiles_collect import main as collect_profiles  # noqa: E402


def main(bids):
    root = Path(bids) / 'derivatives' / 'decoding_simideal'
    fs = sorted((root / 'ideal').glob('sub-*/*_code-*.tsv.gz'))
    if fs:
        d = pd.concat([pd.read_csv(f, sep='\t', dtype={'subject': str}) for f in fs])
        d.to_csv(root / 'ideal_all.tsv.gz', sep='\t', index=False, float_format='%.5g')
        print('ideal', len(fs), 'files', d.groupby(['code', 'roi']).subject.nunique().to_string())
    for code in ('vonmises', 'lam0', 'lam1'):
        if (root / f'pipeline_{code}').exists():
            collect_profiles(bids, f'decoding_simideal/pipeline_{code}', 'cv', 'lowrankcv',
                             root / f'pipeline_{code}.tsv.gz')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-abstractvalue')
    main(p.parse_args().bids_folder)
