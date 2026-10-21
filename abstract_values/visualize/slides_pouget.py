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


def render(names, dpi=250):
    tmp = Path(tempfile.mkdtemp())
    pages = {}
    for n in names:
        subprocess.run(["pdftoppm", "-r", str(dpi), "-png", str(FIG / f"{n}.pdf"),
                        str(tmp / n)], check=True)
        pages[n] = sorted(tmp.glob(f"{n}-*.png"))
    return pages


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
    pg = render(["fig_summary", "fig_connectopy", "sim_v1_ips_mapping"])
    S = pg["fig_summary"][0]
    C = pg["fig_connectopy"][0]
    S1, S2 = pg["sim_v1_ips_mapping"]
    out = FIG / "slides_pouget.pdf"
    png = lambda i: FIG / f"slides_pouget_p{i:02d}.png"
    i = 0

    def nxt():
        nonlocal i
        i += 1
        return dict(out=png(i), n=i)

    with PdfPages(out) as pdf:
        slide(pdf, "Abstract value from orientation:\nfixed codes, flexible routing",
              [(crop(C, 0, 0, 1, 0.345), [0.04, 0.12, 0.92, 0.56])],
              note="Gilles de Hollander · Bedi, Ruff & de Hollander · UZH · "
                   "meeting with Alexandre Pouget, 22 Oct 2026", **nxt())

        slide(pdf, "Same orientation, different value: two mappings",
              [(crop(S1, 0, 0.012, 1, 0.224), [0.03, 0.14, 0.64, 0.66])],
              ["35 participants learn an\norientation-to-CHF mapping",
               "BDM auction: the bid\nis the value (truth-telling)",
               "23 orientations, uniform;\none mapping per fMRI session",
               "Mappings set where values\nbunch: 12/32 vs 2/22/42 CHF"],
              kicker="Design", **nxt())

        slide(pdf, "Bids: efficient coding in value space, after a perceptual stage",
              [(crop(S, 0, 0.231, 1, 0.307), [0.03, 0.14, 0.64, 0.66])],
              ["Two stages: orientation noise\n(cardinal prior), then value\nnoise σ_rep (efficient code)",
               "Category gate at 90° / 22 CHF:\nΔELPD = 270 (LOO)",
               "!Open: per person, perceptual\nvs value noise is weakly\nidentified from bids"],
              kicker="Behaviour", note="Bedi, Ruff & de Hollander (2026)",
              **nxt())

        slide(pdf, "Where the codes live: V1 orientation, parietal cortex value",
              [(crop(S, 0, 0.019, 1, 0.088), [0.03, 0.47, 0.94, 0.36]),
               (crop(S, 0, 0.098, 1, 0.19), [0.03, 0.06, 0.94, 0.40])],
              kicker="Encoding models", **nxt())

        slide(pdf, "V1 is a circle, parietal cortex an ordered line",
              [(crop(S, 0, 0.503, 1, 0.578), [0.03, 0.14, 0.64, 0.66])],
              ["Crossnobis RDMs over the\n23 orientations",
               "V1: 7.5° and 172.5° are\nneighbours. IPS: they are\nthe far ends",
               "IPS geometry does not\nre-scale with the mapping"],
              kicker="Representational geometry", **nxt())

        slide(pdf, "The catch: a fixed orientation code already looks 'efficient'",
              [(crop(S2, 0, 0.03, 1, 0.47), [0.03, 0.14, 0.62, 0.66])],
              [r"Uniform θ, monotone f:" "\n" r"quantile of v = rank of θ",
               r"$I_v = I_\theta\,(d\theta/dv)^2 \propto p(v)^2$",
               "Efficient-coding signatures\ncan't tell value from\norientation here",
               "Tuning comparison neither:\norientation model wins\n~55–60% in V1 and in IPS"],
              kicker="Theory", bullet_box=(0.69, 0.80), **nxt())

        slide(pdf, "A mechanism: rewire V1-to-IPS, keep IPS value labels",
              [(crop(S1, 0, 0.228, 1, 0.48), [0.03, 0.40, 0.64, 0.45]),
               (crop(S1, 0, 0.755, 1, 1.0), [0.03, 0.04, 0.64, 0.35])],
              ["Only the weights depend\non the session's mapping",
               "IPS stays bell-shaped\nin value (R² ≈ .93)",
               "Fisher info ~ p(v) to the\n1.6–1.9 (ideal 2); fixed\ncode: flat",
               "!Prediction: the V1 orientation\nan IPS voxel listens to\nmoves with the mapping"],
              kicker="Toy model", **nxt())

        slide(pdf, "Parietal–V1 coupling follows the mapping in force",
              [(crop(C, 0, 0.0, 1, 0.345), [0.03, 0.14, 0.66, 0.68])],
              ["Trial-to-trial residuals:\nstimulus means removed",
               "Parietal voxels binned by\npreferred value × V1\norientation channel",
               "Ridge sits on the mapping\nin force, not the other one"],
              kicker="Connectivity", bullet_box=(0.71, 0.80), **nxt())

        slide(pdf, "The effect grows along IPS and survives neighbour control",
              [(crop(C, 0, 0.36, 1, 0.68), [0.03, 0.14, 0.66, 0.66])],
              ["IPS0–5: +0.013 ± 0.003,\np = .0001 (±1 lag; n = 35)",
               "IPS0 n.s., rising to\nIPS3–5 +0.020",
               "Mapping read out from the\ncoupling map: 60% of sessions",
               "Gain confound: partly\naddressed, not closed"],
              kicker="Connectivity", bullet_box=(0.71, 0.80), **nxt())

        slide(pdf, "Who routes better has less value noise",
              [(crop(C, 0, 0.70, 0.37, 1.0), [0.03, 0.14, 0.33, 0.66]),
               (crop(S, 0, 0.905, 1, 0.99), [0.37, 0.20, 0.60, 0.50])],
              note="Left: routing score vs model value precision (ρ = .35, p = .04; "
                   "r = .48). Right: value decoding precision vs model noise, per region.",
              kicker="Individual differences", **nxt())

        slide(pdf, "Questions for you", [],
              ["How would you test value vs orientation coding when\n"
               "efficient-coding signatures are identical by design?",
               "Flexible routing vs a fixed population + flexible readout:\n"
               "what distinguishes them in population activity?",
               "Can decoded (posterior) uncertainty in V1 / IPS separate\n"
               "perceptual from value noise where bids cannot?",
               "Information-limiting correlations: should routing\n"
               "show up in noise correlations, as we see?",
               "Single-peak vs multi-peak tuning: most voxels prefer\n"
               "a basis model; what label should the routing test use?"],
              bullet_box=(0.08, 0.76), **nxt())
    print(f"Wrote {out} ({i} slides)")


if __name__ == "__main__":
    main()
