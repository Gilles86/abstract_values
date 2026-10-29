#!/usr/bin/env python3
"""Collect simulate_ideal_decoding.py outputs (run on the cluster).

  derivatives/decoding_simideal/ideal[_cvmatched]_all.tsv.gz         all ideal-observer trials
  derivatives/decoding_simideal/pipeline_<code>[_cvmatched].tsv.gz   pipeline posteriors summarised
                                                                     on both axes
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mapping_profiles_collect import main as collect_profiles  # noqa: E402


def main(bids):
    root = Path(bids) / 'derivatives' / 'decoding_simideal'
    for tag in ('', '_cvmatched'):
        fs = sorted((root / f'ideal{tag}').glob('sub-*/*_code-*.tsv.gz'))
        if fs:
            d = pd.concat([pd.read_csv(f, sep='\t', dtype={'subject': str}) for f in fs])
            d.to_csv(root / f'ideal{tag}_all.tsv.gz', sep='\t', index=False, float_format='%.5g')
            print(f'ideal{tag}', len(fs), 'files',
                  d.groupby(['code', 'roi']).subject.nunique().to_string())
        # the in-sample lam0/lam1 pipeline runs were cancelled part-way: not collected
        for code in (('vonmises',) if tag == '' else ('vonmises', 'lam0', 'lam1')):
            if (root / f'pipeline_{code}{tag}').exists():
                collect_profiles(bids, f'decoding_simideal/pipeline_{code}{tag}', 'cv',
                                 'lowrankcv', root / f'pipeline_{code}{tag}.tsv.gz')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--bids-folder', default='/shares/zne.uzh/gdehol/ds-abstractvalue')
    main(p.parse_args().bids_folder)
