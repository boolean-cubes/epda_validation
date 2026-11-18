"""Fit and compare Hamiltonian term-count scaling across distinct qubit sizes."""

import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from scrape_metadata import ACCEPTED, MANIFEST, atomic_write, column_keys, find_counts

MIN_COMPARISON_SIZES = 6
SCORE_TOLERANCE = 1e-8


@dataclass(frozen=True)
class Model:
    name: str
    basis: tuple
    expression: str
    superpolynomial: bool = False


MODELS = (
    Model("constant", (), "{a:.6g}"),
    Model("power law", (np.log,), "{a:.6g} x^({b:.6g})"),
    Model(
        "power law with finite-size correction",
        (np.log, lambda x: 1 / x - 1),
        "{a:.6g} x^({b:.6g}) exp({c:.6g} (1/x - 1))",
    ),
    Model("exponential", (lambda x: x - 1,), "{a:.6g} exp({b:.6g} (x - 1))", True),
    Model(
        "stretched exponential", (lambda x: np.sqrt(x) - 1,),
        "{a:.6g} exp({b:.6g} (sqrt(x) - 1))", True,
    ),
    Model(
        "quasipolynomial", (np.log, lambda x: np.log(x)**2),
        "{a:.6g} exp({b:.6g} ln(x) {c:+.6g} ln(x)^2)", True,
    ),
)
MODEL_BY_NAME = {model.name: model for model in MODELS}

FIT_COLUMNS = [
    "model", "equation", "log_r2", "log_rmse", "cv_log_rmse",
    "largest_holdout_log_error",
]


def input_files(root):
    if root.is_file():
        return {root.stem: [root]}, None
    if not root.is_dir():
        raise ValueError(f"Metadata path does not exist: {root}")
    groups = defaultdict(list)
    manifest_path = root / MANIFEST
    manifest = None
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        paths = sorted({
            root / entry["path"] for entry in manifest["files"]
            if entry["status"] in ACCEPTED
        })
    else:
        paths = sorted(
            path for path in root.rglob("*")
            if path.is_file() and path.suffix.casefold() == ".csv"
        )
    for path in paths:
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Metadata path leaves the input directory: {path}")
        with path.open(encoding="utf-8-sig", newline="") as stream:
            if all(find_counts(next(csv.reader(stream), []))):
                groups[path.parent.relative_to(root).as_posix()].append(path)
            elif manifest is not None:
                raise ValueError(f"Metadata columns missing from manifest-listed file: {path}")
    return dict(sorted(groups.items())), manifest


def read_family(paths):
    by_qubits = defaultdict(list)
    counts = dict(
        hamiltonians=0, metadata_rows=0, invalid_rows=0,
        duplicate_rows=0, zero_term_rows=0,
    )
    instances = {}
    for path in paths:
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            headers = reader.fieldnames or []
            q_key, t_key = find_counts(headers)
            if not q_key or not t_key:
                raise ValueError(f"Metadata columns missing from CSV: {path}")
            ids = column_keys(headers)
            for row in reader:
                counts["metadata_rows"] += 1
                try:
                    qubits, terms = int(row[q_key]), int(row[t_key])
                    if qubits <= 0 or terms < 0:
                        raise ValueError
                except (ValueError, TypeError):
                    counts["invalid_rows"] += 1
                    continue

                if "file" in ids and "dataset" in ids:
                    file_id = (row[ids["file"]] or "").strip()
                    dataset_id = (row[ids["dataset"]] or "").strip()
                    if file_id and dataset_id:
                        identity = (file_id, dataset_id)
                        previous = instances.get(identity)
                        if previous is not None:
                            if previous[:2] != (qubits, terms):
                                raise ValueError(
                                    f"Conflicting counts for {identity} in "
                                    f"{previous[2]}:{previous[3]} and {path}:{reader.line_num}"
                                )
                            counts["duplicate_rows"] += 1
                            continue
                        instances[identity] = (qubits, terms, path, reader.line_num)

                by_qubits[qubits].append(terms)
                counts["hamiltonians"] += 1
                counts["zero_term_rows"] += terms == 0

    points = []
    for qubits, terms in sorted(by_qubits.items()):
        positive = [count for count in terms if count > 0]
        q1, median, q3 = (None, None, None)
        if positive:
            q1, median, q3 = map(float, np.percentile(positive, (25, 50, 75)))
        points.append(dict(
            qubits=qubits, count=len(terms), positive_count=len(positive),
            q1=q1, median=median, q3=q3,
        ))
    return counts, points


def design(model, n, reference):
    x = n / reference
    return np.column_stack([np.ones_like(x), *(basis(x) for basis in model.basis)])


def fit_model(model, n, y):
    reference = float(np.exp(np.mean(np.log(n))))
    matrix = design(model, n, reference)
    beta, _, rank, _ = np.linalg.lstsq(matrix, y, rcond=None)
    if rank != matrix.shape[1] or abs(beta[0]) > 700:
        return None
    if model.superpolynomial:
        growth = beta[-1]
        if growth <= SCORE_TOLERANCE:
            return None
        if model.name == "quasipolynomial":
            largest_log_slope = beta[1] + 2 * growth * np.log(n[-1] / reference)
            if largest_log_slope <= 0:
                return None
    return reference, beta


def predict_log(name, reference, coefficients, n):
    return design(MODEL_BY_NAME[name], np.asarray(n, dtype=float), reference) @ coefficients


def rmse(residuals):
    return float(np.sqrt(np.mean(np.square(residuals))))


def score_model(model, n, y):
    parameters = 1 + len(model.basis)
    if len(n) < parameters:
        return None
    fitted = fit_model(model, n, y)
    if fitted is None:
        return None
    reference, beta = fitted
    residuals = y - predict_log(model.name, reference, beta, n)
    total = np.sum((y - y.mean())**2)
    cv, tail = None, None

    # Each training fold needs residual degrees of freedom.
    if len(n) >= parameters + 2:
        errors = []
        for index in range(len(n)):
            train = np.arange(len(n)) != index
            heldout = fit_model(model, n[train], y[train])
            if heldout is None:
                continue
            prediction = predict_log(model.name, *heldout, n[index:index + 1])[0]
            error = float(y[index] - prediction)
            errors.append(error)
            if index == len(n) - 1:
                tail = abs(error)
        if len(errors) == len(n):
            cv = rmse(errors)

    a = float(np.exp(beta[0]))
    b, c = (*beta[1:], 0, 0)[:2]
    equation = f"T(n) = {model.expression.format(a=a, b=b, c=c)}"
    if model.basis:
        equation += f", x = n / {reference:.6g}"
    return dict(
        model=model.name, superpolynomial=model.superpolynomial, parameters=parameters,
        equation=equation, reference_qubits=reference, log_coefficients=beta.tolist(),
        log_r2=float(1 - np.sum(residuals**2) / total) if total > SCORE_TOLERANCE**2 else None,
        log_rmse=rmse(residuals), cv_log_rmse=cv, largest_holdout_log_error=tail,
    )


def best_by(fits, metric):
    scored = [fit for fit in fits if fit[metric] is not None]
    if not scored:
        return None
    minimum = min(fit[metric] for fit in scored)
    return min(
        (fit for fit in scored if fit[metric] <= minimum + SCORE_TOLERANCE),
        key=lambda fit: (fit["parameters"], fit[metric]),
    )


def wins_by(candidate, baseline, fraction):
    return (
        candidate is not None
        and baseline - candidate > SCORE_TOLERANCE
        and candidate <= (1 - fraction) * baseline
    )


def assess(points, min_comparison_sizes, min_improvement):
    positive = [point for point in points if point["median"] is not None]
    if not positive:
        return "no_positive_data", None, [], None
    if len(positive) == 1:
        return "one_fitted_size", None, [], None

    n = np.array([point["qubits"] for point in positive], dtype=float)
    y = np.log([point["median"] for point in positive])
    comparable = len(n) >= min_comparison_sizes
    models = MODELS if comparable else MODELS[:2]
    fits = [fit for model in models if (fit := score_model(model, n, y)) is not None]
    baseline = [fit for fit in fits if not fit["superpolynomial"]]
    best_cv = best_by(baseline, "cv_log_rmse")
    if not comparable:
        selected = best_cv or best_by(baseline, "log_rmse")
        return "descriptive_fit", selected, fits, None

    best_tail = best_by(baseline, "largest_holdout_log_error")
    screened = [
        fit for fit in fits if fit["superpolynomial"] and fit["cv_log_rmse"] is not None
    ]
    passed = [
        fit for fit in screened
        if wins_by(fit["cv_log_rmse"], best_cv["cv_log_rmse"], min_improvement)
        and wins_by(
            fit["largest_holdout_log_error"],
            best_tail["largest_holdout_log_error"],
            min_improvement,
        )
    ]
    winner = best_by(passed, "cv_log_rmse")
    strongest = winner or best_by(screened, "cv_log_rmse")
    if winner:
        return "superpolynomial_candidate", winner, fits, strongest
    if strongest and wins_by(strongest["cv_log_rmse"], best_cv["cv_log_rmse"], min_improvement):
        return "inconclusive", best_cv, fits, strongest
    return "no_superpolynomial_signal", best_cv, fits, strongest


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def analyze(root, out, *, min_comparison_sizes=MIN_COMPARISON_SIZES, min_improvement=0.10):
    groups, manifest = input_files(root)
    if not groups:
        raise ValueError(f"No CSV with qubit and term count columns under {root}")

    families = []
    for family, paths in groups.items():
        counts, points = read_family(paths)
        status, selected, fits, strongest = assess(points, min_comparison_sizes, min_improvement)
        families.append(dict(
            family=family, metadata_files=[path.name for path in paths], **counts,
            distinct_qubit_sizes=len(points),
            fit_qubit_sizes=sum(point["positive_count"] > 0 for point in points),
            qubits_min=points[0]["qubits"] if points else None,
            qubits_max=points[-1]["qubits"] if points else None,
            status=status, selected=selected, fits=fits,
            superpolynomial_comparison=strongest, points=points,
        ))

    analysis = dict(
        source_url=manifest["root_url"] if manifest else None,
        scrape_errors=manifest["errors"] if manifest else [],
        min_comparison_sizes=min_comparison_sizes,
        min_improvement=min_improvement,
        families=families,
    )
    out.mkdir(parents=True, exist_ok=True)
    atomic_write(out / "analysis.json", (json.dumps(analysis, indent=2, allow_nan=False) + "\n").encode())

    summary_columns = [
        "family", "status", "hamiltonians", "distinct_qubit_sizes", "fit_qubit_sizes",
        "qubits_min", "qubits_max", "metadata_rows", "invalid_rows",
        "duplicate_rows", "zero_term_rows",
    ]
    write_csv(
        out / "family_summary.csv", summary_columns + FIT_COLUMNS,
        ({**family, **(family["selected"] or {})} for family in families),
    )
    write_csv(
        out / "model_comparison.csv", ["family", "superpolynomial", "parameters", *FIT_COLUMNS],
        ({"family": family["family"], **fit} for family in families for fit in family["fits"]),
    )
    print(f"Analyzed {len(families)} families. Tables: {out}")
    return analysis


def add_fit_options(parser):
    parser.add_argument("--report-dir", type=Path, default=Path("results"))
    parser.add_argument("--min-comparison-sizes", type=int, default=MIN_COMPARISON_SIZES)
    parser.add_argument("--min-improvement", type=float, default=0.10)


def analyze_from_args(args):
    if args.min_comparison_sizes < 5:
        raise ValueError("Use at least five sizes for model comparison.")
    if not 0 < args.min_improvement < 1:
        raise ValueError("Minimum improvement must be between zero and one.")
    return analyze(
        args.metadata_dir, args.report_dir,
        min_comparison_sizes=args.min_comparison_sizes,
        min_improvement=args.min_improvement,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=Path("metadata"))
    add_fit_options(parser)
    try:
        analyze_from_args(parser.parse_args())
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(1, f"{exc}\n")
