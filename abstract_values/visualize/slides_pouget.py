"""Slide-sized figures for a one-to-one with Alexandre Pouget (2026-10-22).

16:9 pages (13.33 x 7.5 in), each one figure + title + a few short bullets.
Panels are cropped from the existing house figures (rendered at 250 dpi), so
re-run those first if the data change:

  notes/figures/fig_summary.pdf          (visualize/figure_summary.py)
  notes/figures/fig_connectopy.pdf       (connective_fields/figure_connectopy.py)
  notes/figures/sim_v1_ips_mapping.pdf   (simulations/v1_ips_mapping.py)

    python -m abstract_values.visualize.slides_pouget
    -> notes/figures/slides_pouget.pdf (+ _pNN.png)
Speaker notes: notes/slides_pouget_notes.md
"""
from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

REPO = Path(__file__).resolve().parents[2]
FIG = REPO / "notes" / "figures"
W, H = 13.333, 7.5
INK, MUTED, ACC = "0.12", "0.42", "#D9730D"

plt.rcParams.update({"font.family": "Helvetica", "pdf.fonttype": 42,
                     "mathtext.fontset": "stixsans"})


def render(names, dpi=250, pages=None):
    """Render PDFs (in notes/figures, or a full path) to PNG; ``pages`` maps a
    name to the 1-based pages to render (default: all)."""
    tmp = Path(tempfile.mkdtemp())
    out = {}
    for n in names:
        src = Path(n) if Path(n).suffix == ".pdf" else FIG / f"{n}.pdf"
        key = src.stem
        cmd = ["pdftoppm", "-r", str(dpi), "-png"]
        for pg in (pages or {}).get(key, [None]):
            extra = ["-f", str(pg), "-l", str(pg)] if pg else []
            subprocess.run(cmd + extra + [str(src), str(tmp / key)], check=True)
        out[key] = sorted(tmp.glob(f"{key}-*.png"))
    return out


def crop(path, x0, y0, x1, y1):
    im = Image.open(path).convert("RGB")
    w, h = im.size
    return im.crop((int(x0 * w), int(y0 * h), int(x1 * w), int(y1 * h)))


def slide(pdf, title, images, bullets=(), kicker=None, img_box=None, note=None,
          bullet_box=(0.70, 0.80), out=None, n=None):
    """images: list of (PIL image, [left, bottom, width, height] in figure
    fractions); each is fitted inside its box keeping the aspect ratio."""
    fig = plt.figure(figsize=(W, H))
    fig.patch.set_facecolor("white")
    if kicker:
        fig.text(0.04, 0.935, kicker, fontsize=13, color=ACC, fontweight="bold",
                 va="bottom")
    fig.text(0.04, 0.925, title, fontsize=24, color=INK, fontweight="bold",
             va="top", wrap=True)
    for im, box in images:
        ax = fig.add_axes(box)
        ax.imshow(im, interpolation="lanczos")
        ax.set_axis_off()
        ax.set_anchor("C")
    x, y = bullet_box
    for b in bullets:
        if b.startswith("!"):     # emphasised line
            fig.text(x, y, b[1:], fontsize=15, color=ACC, fontweight="bold",
                     va="top", wrap=True)
        else:
            fig.text(x - 0.012, y, "•", fontsize=15, color=MUTED, va="top")
            fig.text(x, y, b, fontsize=15, color=INK, va="top",
                     wrap="$" not in b, linespacing=1.25)
        y -= 0.065 + 0.045 * b.count("\n")
    if note:
        fig.text(0.04, 0.03, note, fontsize=10, color=MUTED, va="bottom")
    if n is not None:
        fig.text(0.97, 0.03, f"{n}", fontsize=10, color=MUTED, ha="right")
    pdf.savefig(fig)
    if out:
        fig.savefig(out, dpi=110)
    plt.close(fig)


def main():
    paper = REPO / "notes" / "papers" / "bedi_et_al2026.pdf"
    pg = render(["fig_summary", "fig_connectopy", "fig_connectopy_npc",
                 "sim_v1_ips_mapping", "fig_model_comparison", "fig_gradient",
                 "fig_wrap_behavior", "fig_coupling_components", str(paper)],
                pages={"bedi_et_al2026": [6, 8, 12, 14]})
    S = pg["fig_summary"][0]
    C = pg["fig_connectopy"][0]
    N = pg["fig_connectopy_npc"][0]
    M = pg["fig_model_comparison"][0]
    G = pg["fig_gradient"][0]
    WB = pg["fig_wrap_behavior"][0]
    CC = pg["fig_coupling_components"][0]
    S1, S2 = pg["sim_v1_ips_mapping"]
    B1, B2, B3, B4 = pg["bedi_et_al2026"]          # paper Figs 1, 2, 3, 4
    out = FIG / "slides_pouget.pdf"
    for old in FIG.glob("slides_pouget_p*.png"):
        old.unlink()
    png = lambda i: FIG / f"slides_pouget_p{i:02d}.png"
    i = 0

    def nxt():
        nonlocal i
        i += 1
        return dict(out=png(i), n=i)

    paper_ref = "Bedi, de Hollander, Harl & Ruff (2026), preregistered, n = 119"
    with PdfPages(out) as pdf:
        # ── Part 1: the question and the behavioural logic (Bedi et al.) ────
        slide(pdf, "Where does efficient coding act when value\nis constructed from perception?",
              [(crop(C, 0, 0, 1, 0.345), [0.04, 0.12, 0.92, 0.56])],
              note="Gilles de Hollander · with Saurabh Bedi & Christian Ruff · UZH · "
                   "meeting with Alexandre Pouget, 22 Oct 2026", **nxt())

        slide(pdf, "Value is not in the stimulus: a goal maps a percept onto value",
              [(crop(B1, 0.15, 0.105, 0.86, 0.475), [0.03, 0.10, 0.62, 0.74])],
              ["Same feature, different value\nunder different goals",
               "Change the mapping G and\nthe value prior changes:\n"
               r"$p(v) = p(s)\,|ds/dv|$",
               "Both stages are noisy and\nresource-limited: perception,\nvaluation, or both coded\nefficiently?"],
              kicker="Question", note=paper_ref, **nxt())

        slide(pdf, "Task: learn what each orientation is worth, then bid",
              [(crop(B2, 0.18, 0.103, 0.85, 0.44), [0.03, 0.08, 0.60, 0.76])],
              ["23 orientations, 7.5–172.5°,\nshown equally often",
               "Learn the orientation-to-CHF\nmapping with feedback",
               "Test: low-contrast Gabor,\nBDM auction, no feedback",
               "BDM is incentive-compatible:\nthe bid IS the subjective\nvalue"],
              kicker="Task", note=paper_ref, **nxt())

        slide(pdf, "The trick: change only the mapping, so only the value prior changes",
              [(crop(B3, 0.18, 0.104, 0.85, 0.228), [0.03, 0.22, 0.64, 0.56])],
              ["Orientations uniform in the\nexperiment; the long-term\n(natural) prior favours cardinals",
               "Aligned: dense values at\ncardinal orientations",
               "Misaligned: dense values at\nobliques",
               "Perceptual input statistics\nidentical across mappings"],
              kicker="Design logic", note=paper_ref, **nxt())

        slide(pdf, "Three architectures predict different bias patterns",
              [(crop(B3, 0.18, 0.226, 0.85, 0.338), [0.03, 0.22, 0.64, 0.56])],
              ["Efficient perception: bias\nkeeps its sign across mappings",
               "Efficient valuation: bias flips\nbetween Aligned and Misaligned",
               "Both stages: flips, and is\namplified when the priors agree\n(Aligned), damped when not"],
              kicker="Predictions", note=paper_ref, **nxt())

        slide(pdf, "Bids: efficient coding at both stages, on different timescales",
              [(crop(B4, 0.18, 0.104, 0.85, 0.46), [0.03, 0.06, 0.58, 0.80])],
              ["Bias reverses between Aligned\nand Misaligned (t = 27)",
               "Asymmetric, and persists under\nUniform: a perceptual stage too",
               "Perception: long-term prior.\nValue: adapts to the session",
               "!So where in the brain is the\nvalue code, and how does it\nfollow the mapping?"],
              kicker="Result", note=paper_ref, bullet_box=(0.66, 0.80), **nxt())

        # ── Part 2: the fMRI study ──────────────────────────────────────────
        slide(pdf, "fMRI: the same task, both mappings within each person",
              [(crop(S1, 0, 0.012, 1, 0.224), [0.03, 0.14, 0.64, 0.66])],
              ["35 participants, two fMRI\nsessions, one mapping each",
               "Cdf = Misaligned (values\nbunch at 12 and 32 CHF)",
               "Inverse cdf = Aligned (2, 22\nand 42 CHF, the cardinals)",
               "Single-trial betas, 23\norientations × 8 runs"],
              kicker="fMRI study", **nxt())

        slide(pdf, "In the scanner, bids show the same multi-stage signature",
              [(crop(S, 0, 0.231, 1, 0.307), [0.03, 0.14, 0.64, 0.66])],
              ["Bias and spread swing with\nthe mapping (violet vs pink)",
               "Two-stage model + category\ngate at 90° / 22 CHF\n(ΔELPD = 270)",
               "!Open: per person, perceptual\nvs value noise is weakly\nidentified from bids alone"],
              kicker="Behaviour", **nxt())

        slide(pdf, "How could the brain implement a mapping that changes?",
              [(crop(S1, 0, 0.228, 1, 0.48), [0.03, 0.14, 0.62, 0.66])],
              ["A. Parietal cortex codes\norientation; value is read\nout downstream",
               "B. Parietal value code with\nfixed labels; the V1 input\nis re-routed (left: toy model)",
               "C. Parietal neurons re-allocate\nto the value prior (an efficient\ncode in the tuning itself)"],
              kicker="Hypotheses", bullet_box=(0.68, 0.80), **nxt())

        slide(pdf, "Where the codes live: V1 orientation, parietal cortex value",
              [(crop(S, 0, 0.019, 1, 0.088), [0.03, 0.47, 0.94, 0.36]),
               (crop(S, 0, 0.098, 1, 0.19), [0.03, 0.06, 0.94, 0.40])],
              kicker="Encoding models", **nxt())

        slide(pdf, "V1 is a circle, parietal cortex a line: IPS codes value,\nnot orientation",
              [(crop(S, 0, 0.503, 1, 0.5695), [0.03, 0.12, 0.64, 0.64])],
              ["V1: 7.5° and 172.5° are\nneighbours. IPS: the far ends",
               "Geometry over orientations is\nthe same in both sessions, as\na quantile value code predicts",
               "Orientation has no edge at\n0°/180°. Value does: 42 CHF\nnext to 2 CHF",
               "!The IPS axis is cut where\nthe task puts it, not where\nthe stimulus does"],
              kicker="Representational geometry", **nxt())

        slide(pdf, "The catch: a fixed orientation code already looks 'efficient'",
              [(crop(S2, 0, 0.03, 1, 0.47), [0.03, 0.14, 0.62, 0.66])],
              [r"Uniform θ, monotone f:" "\n" r"quantile of v = rank of θ",
               r"$I_v = I_\theta\,(d\theta/dv)^2 \propto p(v)^2$",
               "Efficient-coding signatures\nalone can't tell A from B",
               "!What separates them:\nwrap-around. A sensory code\nis a circle, a value code\nis a line"],
              kicker="Theory", bullet_box=(0.69, 0.80), **nxt())

        slide(pdf, "Tuning: circle in V1, line in IPS, and the IPS line tracks\nthe session's value distribution",
              [(crop(M, 0, 0, 1, 1), [0.02, 0.05, 0.64, 0.78])],
              ["Same parameter count, cvR²,\nall voxels, n = 35",
               "Circle vs line flips: V1–3\ncircle, IPS line (IPS vs V1–3\nt(34) = −4.7, p < .001)",
               "Preferred values follow the\nvalue distribution (quantile)\nbetter than fixed CHF, incl.\nIPS1–2 and IPS3–5",
               "!An efficient value code in\nthe tuning itself (C)"],
              kicker="Model comparison", bullet_box=(0.69, 0.80), **nxt())

        slide(pdf, "What B predicts that A does not: the preferred orientation moves",
              [(crop(S1, 0, 0.51, 1, 0.752), [0.03, 0.42, 0.64, 0.42]),
               (crop(S1, 0, 0.755, 1, 1.0), [0.03, 0.03, 0.64, 0.37])],
              ["Re-routing alone gives an\nefficient value code\n(Fisher info ~ p(v) to the 1.6–1.9)",
               "IPS stays bell-shaped in value",
               "!Test: which V1 orientation\ndoes a parietal voxel listen to,\nsession by session?"],
              kicker="Toy model", **nxt())

        slide(pdf, "Parietal–V1 coupling follows the mapping in force",
              [(crop(C, 0, 0.0, 1, 0.345), [0.03, 0.14, 0.66, 0.68])],
              ["Trial-to-trial residuals:\nstimulus means removed",
               "Parietal voxels binned by\npreferred value × V1\norientation channel",
               "Fixed coupling cancels in the\nscore: A predicts zero"],
              kicker="Connectivity", bullet_box=(0.71, 0.80), **nxt())

        slide(pdf, "The effect grows along IPS and survives the neighbour control",
              [(crop(C, 0, 0.36, 1, 0.68), [0.03, 0.14, 0.66, 0.66])],
              ["IPS0–5: +0.013 ± 0.003,\np = .0001 (±1 lag; n = 35)",
               "IPS0 n.s., rising to\nIPS3–5 +0.020",
               "Mapping read out from the\ncoupling map: 60% of sessions",
               "Gain confound: partly\naddressed, not closed"],
              kicker="Connectivity", bullet_box=(0.71, 0.80), **nxt())

        slide(pdf, "It replicates in NPC1–3 and with the full tuning curve as template",
              [(crop(N, 0, 0.33, 1, 0.635), [0.03, 0.44, 0.64, 0.42]),
               (crop(N, 0, 0.655, 1, 1.0), [0.03, 0.02, 0.64, 0.42])],
              ["NPC1–3: +0.012, p = .0002\n(±1 lag); flat across NPC1–3,\nno gradient as along IPS",
               "Multi-peak (value-basis)\ntemplate: a bit weaker,\nsignificant everywhere,\nincl. IPS0",
               "Orientation-winning voxels\nalso route (IPS0–5 +0.010)",
               "NPC and Wang IPS overlap:\nnot independent"],
              kicker="Robustness", **nxt())

        slide(pdf, "Most coupling is a stable, orientation-locked link: exactly what\na quantile value code needs",
              [(crop(CC, 0, 0, 1, 0.47), [0.02, 0.43, 0.66, 0.38]),
               (crop(CC, 0, 0.5, 1, 1), [0.02, 0.03, 0.66, 0.38])],
              ["Fit both at once: a stable\ntemplate (a) + one that moves\nwith the mapping (b)",
               "Stable part: a = +.037,\nt(34) = 6.4. Survives gain\nremoval and low-drive trials",
               "Moving part: on strongly\ndriven trials (b = +.016,\np = .02)",
               "!Uniform θ + monotone map:\nfixed wiring from orientation\nrank already yields the\nefficient value code"],
              kicker="Connectivity, two components", bullet_box=(0.71, 0.80),
              note="IPS0–5, n = 34–35. Moving part: n.s. over all trials (+.006) "
                   "or after gain removal; tracks the original score (r = .43).", **nxt())

        slide(pdf, "Brain maps onto the model's stages: V1 with perception,\nIPS with valuation",
              [(crop(G, 0, 0, 1, 1), [0.02, 0.05, 0.64, 0.76])],
              ["Code strength vs the cognitive\nmodel's precision per stage\n(partial r, n = 35)",
               "V1 orientation code and\nperceptual precision: +.40",
               "IPS0–5 value code and value\nprecision: +.31 to +.39",
               "!Inside the model as covariates:\na stronger IPS value code means\nless value noise (+1.8 [0.7, 3.1]\nlogit share), in both fits"],
              kicker="Individual differences", bullet_box=(0.69, 0.80),
              note="Sequential fit, n = 35. The IPS half holds in every fit variant; "
                   "the V1 half depends on how the fit splits the noise.", **nxt())

        slide(pdf, "Code shape predicts bid precision, with opposite signs\nin V1 and IPS",
              [(crop(WB, 0, 0, 1, 1), [0.02, 0.18, 0.66, 0.58])],
              ["Precise bidders: a circular\ncode in V1–3 (ρ ≈ −.4) and a\nline-like code in IPS0–2\n(ρ ≈ +.4 to +.5)",
               "Behaviour already says the\nreadout doesn't wrap: 1° is\nnever bid as 42 CHF",
               "Who routes more has less\nvalue noise (IPS0–5,\nρ = −.38, p = .02)"],
              kicker="Individual differences", bullet_box=(0.71, 0.80), **nxt())

        slide(pdf, "Summary and questions", [],
              ["!V1 codes orientation on a circle; IPS codes value on a line",
               "!The IPS code tracks the value distribution, and its strength and\n"
               "shape predict value-stage precision",
               "!V1 to IPS: a stable orientation-rank link, plus routing that\n"
               "follows the mapping on strongly driven trials",
               "Quantile value = orientation rank by design: which experiment\n"
               "separates an efficient value code from a relabelled sensory axis?\n"
               "(non-uniform orientation sampling, non-monotone mapping)",
               "Re-routing vs a fixed population with a flexible readout:\n"
               "what distinguishes them in population activity?",
               "Should routing show up in (information-limiting) noise correlations?"],
              bullet_box=(0.08, 0.78), **nxt())
    print(f"Wrote {out} ({i} slides)")


if __name__ == "__main__":
    main()
