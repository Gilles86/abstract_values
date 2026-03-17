"""Calculate total earnings per participant from reward files across sessions.

Reward files are expected under:
  {bids_folder}/sourcedata/behavior/sub-{subject}/ses-{session}/reward_{subject}_{session}_{run}.txt

Falls back to a local logs directory when --bids_folder is not given.

Warns if a participant does not have all EXPECTED_RUNS_TOTAL runs combined
across both sessions (default: 16, i.e. 8 per session × 2 sessions).
"""
import argparse
import warnings
from collections import defaultdict
from pathlib import Path

SHOW_UP_FEE_PER_SESSION = 10.0
EXPECTED_RUNS_TOTAL = 16  # 8 runs × 2 sessions (fMRI); override with --expected_runs for behavioral (16)
DEFAULT_BIDS_FOLDER = Path('/data/ds-abstract_values_pilot')


def load_rewards(behavior_dir: Path) -> dict:
    """Return rewards[(subject, session)] = {run: value}."""
    rewards = defaultdict(dict)

    for reward_file in sorted(behavior_dir.glob('sub-*/*/reward_*.txt')):
        try:
            value = float(reward_file.read_text().strip())
        except ValueError:
            warnings.warn(f'Could not parse {reward_file} — skipping.')
            continue

        # Filename: reward_{subject}_{session}_{run}.txt
        parts = reward_file.stem.split('_')
        if len(parts) < 4:
            warnings.warn(f'Unexpected filename format: {reward_file.name} — skipping.')
            continue

        subject = parts[1]
        session = int(parts[2])
        run = int(parts[3])
        rewards[(subject, session)][run] = value

    return rewards


def summarize(behavior_dir: Path, expected_runs: int = EXPECTED_RUNS_TOTAL) -> None:
    rewards = load_rewards(behavior_dir)

    if not rewards:
        print(f'No reward files found under {behavior_dir}')
        return

    subjects = sorted({s for s, _ in rewards})
    sessions = sorted({ses for _, ses in rewards})

    # ── per-subject summary ────────────────────────────────────────────────
    for subject in subjects:
        print(f'\n{"="*50}')
        print(f'Subject: {subject}')

        total_variable = 0.0
        total_runs = 0
        sessions_present = []

        for session in sessions:
            if (subject, session) not in rewards:
                continue

            run_rewards = rewards[(subject, session)]
            sessions_present.append(session)
            session_variable = sum(run_rewards.values())
            n_runs = len(run_rewards)
            session_total = SHOW_UP_FEE_PER_SESSION + session_variable

            total_variable += session_variable
            total_runs += n_runs

            print(f'  Session {session}: {n_runs} run(s)  |  '
                  f'variable: {session_variable:.2f} CHF  |  '
                  f'total (incl. show-up fee): {session_total:.2f} CHF')

        # Grand total across all sessions
        n_sessions = len(sessions_present)
        grand_total = n_sessions * SHOW_UP_FEE_PER_SESSION + total_variable
        print(f'  ── {n_sessions} session(s) | {total_runs} run(s) total ──')
        print(f'  Variable reward:  {total_variable:.2f} CHF')
        print(f'  Show-up fees:     {n_sessions * SHOW_UP_FEE_PER_SESSION:.2f} CHF  ({n_sessions} × {SHOW_UP_FEE_PER_SESSION:.2f})')
        print(f'  GRAND TOTAL:      {grand_total:.2f} CHF')

        # ── warnings ──────────────────────────────────────────────────────
        if total_runs < expected_runs:
            warnings.warn(
                f'Subject {subject} has only {total_runs}/{expected_runs} runs '
                f'across {n_sessions} session(s).'
            )
        elif total_runs > expected_runs:
            warnings.warn(
                f'Subject {subject} has {total_runs} runs, more than the '
                f'expected {expected_runs}.'
            )

    print(f'\n{"="*50}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Calculate participant earnings from reward files.'
    )
    parser.add_argument(
        '--bids_folder',
        default=None,
        type=Path,
        help=f'Path to BIDS root folder (default: {DEFAULT_BIDS_FOLDER}). '
             'Reward files are read from <bids_folder>/sourcedata/behavior/.',
    )
    parser.add_argument(
        '--expected_runs',
        default=EXPECTED_RUNS_TOTAL,
        type=int,
        help=f'Expected number of runs per participant in total (default: {EXPECTED_RUNS_TOTAL})',
    )
    args = parser.parse_args()

    if args.bids_folder is None:
        local_logs = Path(__file__).parent / 'logs'
        behavior_dir = local_logs if local_logs.exists() else DEFAULT_BIDS_FOLDER / 'sourcedata' / 'behavior'
    else:
        behavior_dir = args.bids_folder / 'sourcedata' / 'behavior'

    if not behavior_dir.exists():
        parser.error(f'Behavior directory not found: {behavior_dir}')

    summarize(behavior_dir, expected_runs=args.expected_runs)


if __name__ == '__main__':
    main()
