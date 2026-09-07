"""The project's one ridge penalty for closed-form basis-weight fits.

Every weighted-basis model -- von Mises in orientation space, log-Gaussian in
value space, encoding side and decoding side alike -- fits its per-voxel weights
by a closed-form ridge solve, and that solve needs a penalty. Two 29-subject
sweeps picked it:

  * orientation (``sweep_v1_k_kappa``): alpha=10 beat alpha=1 in 29/29 subjects
    and the near-unregularised fit in 29/29; alpha=100 collapses.
  * value (``sweep_npc_value``): the best of 150 joint cells is exactly
    k=8, width 2x basis spacing, alpha=10.

So alpha is settled, and analyses must not drift from it: a decoder fitting
weights at alpha=0 against an encoding model fitted at alpha=10 is not the same
model, and the mismatch is invisible in the output filenames. This module makes
the value single-sourced and makes departing from it deliberate.

The sweeps themselves are the one legitimate exception -- varying alpha is
their entire purpose -- and they set their own grids without calling in here.
"""

DEFAULT_RIDGE_ALPHA = 10.0


def enforce_default_alpha(alpha, allow_nondefault=False, context=''):
    """Return ``alpha``, refusing a non-default value unless it is opted into.

    Parameters
    ----------
    alpha : float
        The requested ridge penalty.
    allow_nondefault : bool
        Set by the caller's ``--allow-nondefault-alpha`` flag.
    context : str
        What is being fitted, for the message ('vonmises basis weights').

    Raises
    ------
    SystemExit
        When ``alpha`` differs from the project default and the override was
        not given.
    """
    if float(alpha) == DEFAULT_RIDGE_ALPHA:
        return float(alpha)

    where = f' for {context}' if context else ''
    if not allow_nondefault:
        raise SystemExit(
            f'ERROR: alpha={alpha}{where} differs from the project default '
            f'({DEFAULT_RIDGE_ALPHA}), which two 29-subject sweeps picked in '
            f'both stimulus spaces. Analyses must not silently drift from it. '
            f'Pass --allow-nondefault-alpha if you really mean to, and expect '
            f'the result to be incomparable with everything else on disk.')

    print(f'  !! WARNING: alpha={alpha}{where}, NOT the project default '
          f'({DEFAULT_RIDGE_ALPHA}). This fit is not comparable with the rest '
          f'of the analysis. !!', flush=True)
    return float(alpha)
