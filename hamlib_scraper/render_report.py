"""Render a Markdown summary and a PDF report from the scaling analysis."""

import argparse
from collections import Counter
import json
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import EngFormatter, MaxNLocator
import numpy as np

from fit_scaling import best_by, predict_log

STATUS = {
    "superpolynomial_candidate": "Super-polynomial candidate",
    "inconclusive": "Inconclusive validation",
    "no_superpolynomial_signal": "No super-polynomial signal",
    "descriptive_fit": "Descriptive fit",
    "one_fitted_size": "One positive-term qubit size",
    "no_positive_data": "No positive term counts",
}
SCREENED = {"superpolynomial_candidate", "inconclusive", "no_superpolynomial_signal"}
OBSERVED = "#176b73"
SELECTED = "#274e9b"
ALTERNATIVE = "#a34c1c"


def number(value):
    return "—" if value is None else f"{value:.3g}"


def overview(analysis):
    families = analysis["families"]
    counts = Counter(family["status"] for family in families)
    total = sum(family["hamiltonians"] for family in families)
    screened = sum(counts[status] for status in SCREENED)
    minimum = analysis["min_comparison_sizes"]
    improvement = analysis["min_improvement"]
    lines = [
        f"{len(families):,} families and {total:,} encoded Hamiltonians.",
        f"{screened:,} families met the {minimum}-size minimum for screening. "
        f"{counts['superpolynomial_candidate']:,} passed the screen.",
        f"A candidate must reduce both holdout errors by at least {improvement:.0%}. "
        "Each error is compared with the best polynomial-scaling baseline for that metric.",
        "Fits use the median positive term count at each qubit size. Each size has equal weight.",
    ]
    if analysis["scrape_errors"]:
        lines.append(f"Partial crawl. The manifest lists {len(analysis['scrape_errors'])} errors.")
    return lines, counts


def write_markdown(analysis, path):
    paragraphs, _ = overview(analysis)
    lines = ["# HamLib chemistry term-count scaling", ""]
    for paragraph in paragraphs:
        lines.extend([paragraph, ""])
    if analysis["source_url"]:
        lines.extend([f"[HamLib chemistry metadata]({analysis['source_url']})", ""])
    lines.extend([
        "| Family | Hamiltonians | Qubit sizes | Fitted function | Log R² | CV log RMSE | Largest-size log error | Status |",
        "|---|---:|---:|---|---:|---:|---:|---|",
    ])
    for family in analysis["families"]:
        fit = family["selected"] or {}
        equation = f"`{fit['equation']}`" if fit else "—"
        lines.append(
            f"| {family['family']} | {family['hamiltonians']:,} | "
            f"{family['distinct_qubit_sizes']} | {equation} | {number(fit.get('log_r2'))} | "
            f"{number(fit.get('cv_log_rmse'))} | {number(fit.get('largest_holdout_log_error'))} | "
            f"{STATUS[family['status']]} |"
        )
    lines.extend([
        "",
        "Cross-validation holds out one size at a time. Largest-size error is the absolute "
        "log prediction error when training on smaller sizes only.",
        "",
        "The vibrational directory, when present, pools molecules, encodings, and basis truncations.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def save(pdf, fig):
    pdf.savefig(fig)
    plt.close(fig)


def coverage_page(pdf, analysis):
    notes, counts = overview(analysis)
    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    fig.subplots_adjust(top=0.46, left=0.34, right=0.93, bottom=0.12)
    fig.suptitle("HamLib chemistry term-count scaling", fontsize=21, y=0.95)
    prose = "\n\n".join(textwrap.fill(note, 100) for note in notes)
    fig.text(0.08, 0.87, prose, va="top", fontsize=11)
    keys = [status for status in reversed(STATUS) if counts[status]]
    bars = ax.barh([STATUS[key] for key in keys], [counts[key] for key in keys], color=OBSERVED)
    ax.bar_label(bars, padding=4)
    ax.set_xlim(0, max(counts.values()) * 1.18)
    ax.set_xlabel("Number of families")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    save(pdf, fig)


def draw_fit(ax, fit, n, label, color, linestyle):
    prediction = predict_log(
        fit["model"], fit["reference_qubits"], fit["log_coefficients"], n,
    )
    ax.plot(n, np.exp(prediction), label=f"{label} · {fit['model']}",
            color=color, linestyle=linestyle)


def fit_page(pdf, family):
    fig, ax = plt.subplots(figsize=(11.7, 8.3))
    fig.subplots_adjust(top=0.82, bottom=0.27, left=0.10, right=0.94)
    fig.suptitle(family["family"], fontsize=18, y=0.95)
    fig.text(
        0.10, 0.88,
        f"{family['hamiltonians']:,} Hamiltonians · {family['fit_qubit_sizes']} fitted sizes "
        f"· {STATUS[family['status']]}",
    )

    points = [point for point in family["points"] if point["median"] is not None]
    n = np.array([point["qubits"] for point in points])
    medians = np.array([point["median"] for point in points])
    q1 = np.array([point["q1"] for point in points])
    q3 = np.array([point["q3"] for point in points])
    ax.errorbar(
        n, medians, yerr=[medians - q1, q3 - medians], fmt="o", markersize=4,
        color=OBSERVED, elinewidth=1, capsize=2, label="Median and interquartile range",
    )

    selected = family["selected"]
    alternative = family["superpolynomial_comparison"]
    if alternative and alternative["model"] == selected["model"]:
        alternative = best_by(
            [fit for fit in family["fits"] if not fit["superpolynomial"]],
            "cv_log_rmse",
        )
    grid = np.geomspace(n[0], n[-1], 250)
    draw_fit(ax, selected, grid, "Selected", SELECTED, "-")
    if alternative:
        draw_fit(ax, alternative, grid, "Comparison", ALTERNATIVE, "--")
    ax.set(xscale="log", yscale="log", xlabel="Qubits", ylabel="Pauli terms")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)

    for index, fit in enumerate([selected, alternative]):
        if fit is None:
            continue
        color = SELECTED if index == 0 else ALTERNATIVE
        y = 0.17 - index * 0.077
        fig.text(0.10, y, fit["equation"], color=color, fontsize=11)
        fig.text(
            0.10, y - 0.03,
            f"Log R² {number(fit['log_r2'])} · Log RMSE {number(fit['log_rmse'])} · "
            f"CV log RMSE {number(fit['cv_log_rmse'])} · "
            f"Largest-size log error {number(fit['largest_holdout_log_error'])}",
            fontsize=10,
        )
    save(pdf, fig)


def descriptive_pages(pdf, families):
    per_page = 6
    for start in range(0, len(families), per_page):
        subset = families[start:start + per_page]
        fig, axes = plt.subplots(3, 2, figsize=(11.7, 8.3))
        fig.subplots_adjust(hspace=1.0, wspace=0.35, top=0.83, bottom=0.10)
        fig.suptitle("Families without a scaling fit", fontsize=18)
        for ax, family in zip(axes.flat, subset):
            ax.set_title(family["family"], loc="left", fontsize=9)
            ax.set_yticks([])
            point = next((p for p in family["points"] if p["median"] is not None), None)
            if point is None:
                ax.set_xticks([])
                detail = f"{family['hamiltonians']:,} Hamiltonians · No positive term counts"
            else:
                median = point["median"]
                ax.errorbar(
                    median, 0, xerr=[[median - point["q1"]], [point["q3"] - median]],
                    fmt="o", color=OBSERVED, capsize=4,
                )
                margin = max((point["q3"] - point["q1"]) / 2, median * 0.05, 0.5)
                ax.set_xlim(max(0, point["q1"] - margin), point["q3"] + margin)
                ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
                ax.xaxis.set_major_formatter(EngFormatter(sep=""))
                ax.set_xlabel("Pauli terms, median and interquartile range", fontsize=8)
                detail = f"{point['qubits']} qubits · {family['hamiltonians']:,} Hamiltonians"
            ax.text(0.02, 0.07, detail, transform=ax.transAxes, fontsize=8)
        for ax in list(axes.flat)[len(subset):]:
            ax.set_axis_off()
        save(pdf, fig)


def render(out):
    analysis = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    write_markdown(analysis, out / "report.md")
    families = analysis["families"]
    with PdfPages(out / "scaling_report.pdf") as pdf:
        coverage_page(pdf, analysis)
        for family in families:
            if family["selected"] is not None:
                fit_page(pdf, family)
        descriptive_pages(pdf, [family for family in families if family["selected"] is None])
    print(f"Reports: {out / 'report.md'} and {out / 'scaling_report.pdf'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("results"))
    try:
        render(parser.parse_args().report_dir)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
