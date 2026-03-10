import argparse
from pathlib import Path
from collections import defaultdict

SHOW_UP_FEE = 10.0

def main(bids_folder):
    behavior_dir = Path(bids_folder) / 'sourcedata' / 'behavior'

    # Collect totals: rewards[(subject, session)] = list of per-run rewards
    rewards = defaultdict(list)

    for reward_file in sorted(behavior_dir.glob('sub-*/ses-*/reward_*.txt')):
        value = float(reward_file.read_text().strip())
        # Filename: reward_{subject}_{session}_{run}.txt
        parts = reward_file.stem.split('_')
        subject, session = int(parts[1]), int(parts[2])
        rewards[(subject, session)].append(value)

    if not rewards:
        print(f'No reward files found under {behavior_dir}')
        return

    subjects = sorted({s for s, _ in rewards})
    sessions = sorted({ses for _, ses in rewards})

    # Header (tab-separated for easy paste into Google Sheets)
    header_cols = ['subject']
    for ses in sessions:
        header_cols += [f'ses{ses}_variable_reward', f'ses{ses}_total_payment', f'ses{ses}_n_runs']
    print('\t'.join(header_cols))

    for sub in subjects:
        row = [f'sub-{sub}']
        for ses in sessions:
            if (sub, ses) in rewards:
                runs = rewards[(sub, ses)]
                total_variable = sum(runs)
                total_payment = SHOW_UP_FEE + total_variable
                row += [f'{total_variable:.2f}', f'{total_payment:.2f}', str(len(runs))]
            else:
                row += ['', '', '']
        print('\t'.join(row))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Summarize per-subject reward totals from reward files.')
    parser.add_argument('--bids_folder', default='/Users/gdehol/data/ds-abstract_values_pilot',
                        help='Path to BIDS root folder (default: /Users/gdehol/data/ds-abstract_values_pilot)')
    args = parser.parse_args()
    main(args.bids_folder)
