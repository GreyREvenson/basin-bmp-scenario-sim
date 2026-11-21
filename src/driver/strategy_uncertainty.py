"""Phase 16: sampling-uncertainty comparisons of scientifically comparable ensembles.

No scientific models are initialized, no original result is modified, and no
inferential ranking is produced without the analyst's explicit assertions.
Bootstrap percentile intervals quantify *Monte Carlo sampling precision* only;
they cannot measure structural, observational, or input-data uncertainty.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any
from html import escape as xml_escape

import numpy as np

from src.driver.ensemble_analysis import DecisionSupportConfig
from src.driver.ensemble_checkpoints import _atomic_json
from src.driver.ensemble_comparison import (
    StrategyComparisonConfig, _atomic_text, _load_and_compare,
)

UNCERTAINTY_CONFIG_SCHEMA_VERSION = 1
UNCERTAINTY_RESULT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class UncertaintyConfig:
    """Independent or verified common-random-number paired sampling design."""

    sampling_design: str
    sampling_design_confirmed: bool
    n_bootstrap: int = 2000
    confidence_level: float = 0.95
    random_seed: int = 0
    pairing_id: str | None = None

    @classmethod
    def read(cls, filename: str | Path) -> "UncertaintyConfig":
        try:
            data = json.loads(Path(filename).expanduser().resolve(strict=True).read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid strategy uncertainty configuration JSON") from exc
        known = {"schema_version", "sampling_design", "sampling_design_confirmed",
                 "n_bootstrap", "confidence_level", "random_seed", "pairing_id"}
        if (not isinstance(data, dict) or set(data) - known or
                type(data.get("schema_version")) is not int or
                data["schema_version"] != UNCERTAINTY_CONFIG_SCHEMA_VERSION):
            raise ValueError("Unsupported uncertainty configuration schema or unknown fields")
        design = data.get("sampling_design")
        if type(design) is not str or design not in ("independent", "paired"):
            raise ValueError("sampling_design must be independent or paired")
        if data.get("sampling_design_confirmed") is not True:
            raise ValueError("sampling_design_confirmed must be explicitly true")
        n = data.get("n_bootstrap", 2000)
        if type(n) is not int or not (200 <= n <= 50000):
            raise ValueError("n_bootstrap must be an integer between 200 and 50000")
        cl = data.get("confidence_level", 0.95)
        if type(cl) not in (int, float) or not math.isfinite(cl) or not (0.5 <= cl < 1.0):
            raise ValueError("confidence_level must be a finite number in [0.5, 1)")
        seed = data.get("random_seed", 0)
        if type(seed) is not int or not (0 <= seed < 2**64):
            raise ValueError("random_seed must be an unsigned 64-bit integer")
        pairing = data.get("pairing_id")
        if design == "paired":
            if type(pairing) is not str or not pairing.strip():
                raise ValueError("paired sampling requires nonempty pairing_id")
        elif pairing is not None:
            raise ValueError("pairing_id is only valid for paired sampling")
        return cls(design, True, n, float(cl), seed, pairing)


def _validate_sampling_design(records: list[dict[str, Any]], config: UncertaintyConfig) -> None:
    """Metadata checks supplement, but never replace, analyst design validation."""
    if any(len(record["worker_seeds"]) < 2 for record in records):
        raise ValueError("Bootstrap inference requires at least two realizations per strategy")
    first = records[0]
    if config.sampling_design == "paired":
        for record in records[1:]:
            if (record["scenario_ids"] != first["scenario_ids"] or
                    record["worker_seeds"] != first["worker_seeds"]):
                raise ValueError("Paired sampling requires matching ordered scenario IDs and worker seeds")
    else:
        seen: set[int] = set()
        for record in records:
            worker_seeds = set(record["worker_seeds"])
            if seen.intersection(worker_seeds):
                raise ValueError(
                    "Independent sampling requires non-overlapping worker seeds; "
                    "shared seeds do not prove pairing. Use paired only for a verified common-random-number design"
                )
            seen.update(worker_seeds)


def _single_realization_win_probability(a: np.ndarray, b: np.ndarray, *,
                                        paired: bool, larger_is_better: bool) -> float:
    """P(B wins) + 0.5 P(tie) for matched or independent random realizations.

    Independent comparisons use all cross-product pairs without materializing
    an N x M matrix. Pairing uses the same realization index in both arrays.
    """
    if not larger_is_better:
        a, b = -a, -b
    if paired:
        return float((np.count_nonzero(b > a) + 0.5 * np.count_nonzero(b == a)) / len(b))
    sorted_a = np.sort(a)
    below = np.searchsorted(sorted_a, b, side="left")
    equal = np.searchsorted(sorted_a, b, side="right") - below
    return float((sum(below) + 0.5 * sum(equal)) / (len(a) * len(b)))


def _metric(samples: np.ndarray, indices: int, bootstrap_differences: np.ndarray,
            confidence_level: float, *, beneficial_direction: int) -> dict[str, Any]:
    """Difference is always B minus A; direction is +1 for benefit, -1 for cost."""
    difference = float(samples[indices])
    alpha = (1.0 - confidence_level) / 2
    lower, upper = np.quantile(bootstrap_differences[:, indices], [alpha, 1 - alpha])
    wins = np.count_nonzero(bootstrap_differences[:, indices] * beneficial_direction > 0)
    ties = np.count_nonzero(bootstrap_differences[:, indices] == 0)
    return {
        "difference_b_minus_a": difference,
        "bootstrap_percentile_interval": [float(lower), float(upper)],
        "bootstrap_fraction_favoring_b_with_half_ties": float((wins + 0.5 * ties) / len(bootstrap_differences)),
    }


def _matrix(record: dict[str, Any], target_config: DecisionSupportConfig) -> np.ndarray:
    """[cost USD, target1 kg, target1 hit, ..., joint hit, budget hit]."""
    n = len(record["costs"])
    arrays = [record["costs"]]
    joint = np.ones(n, dtype=bool)
    for target in target_config.targets:
        values = record["target_values"][target.name]
        event = values >= target.minimum_removed_mass_kg
        arrays.extend((values, event.astype(np.float64)))
        joint &= event
    arrays.append(joint.astype(np.float64))
    if target_config.budget_usd is not None:
        arrays.append((record["costs"] <= target_config.budget_usd).astype(np.float64))
        arrays.append((joint & (record["costs"] <= target_config.budget_usd)).astype(np.float64))
    return np.column_stack(arrays)


def _bootstrap_differences(a: np.ndarray, b: np.ndarray, *, rng: np.random.Generator,
                           n_bootstrap: int, paired: bool) -> np.ndarray:
    """Bound peak memory by resampling in batches, preserving RNG determinism."""
    result = np.empty((n_bootstrap, a.shape[1]), dtype=np.float64)
    for start in range(0, n_bootstrap, 64):
        count = min(64, n_bootstrap - start)
        ia = rng.integers(0, len(a), size=(count, len(a)))
        ib = ia if paired else rng.integers(0, len(b), size=(count, len(b)))
        result[start:start + count] = b[ib].mean(axis=1) - a[ia].mean(axis=1)
    return result


def analyze_strategy_uncertainty(
    comparison_config: StrategyComparisonConfig,
    target_config: DecisionSupportConfig,
    uncertainty_config: UncertaintyConfig,
) -> dict[str, Any]:
    """Perform pairwise Monte Carlo precision analysis, never scientific ranking."""
    if (comparison_config.comparison_mode != "comparable" or
            not comparison_config.scientific_comparability_confirmed or
            comparison_config.scientific_model_id is None):
        raise ValueError("Uncertainty comparisons require explicitly confirmed comparable mode")
    comparison, records = _load_and_compare(comparison_config, target_config)
    _validate_sampling_design(records, uncertainty_config)
    matrices = [_matrix(r, target_config) for r in records]
    rng = np.random.default_rng(uncertainty_config.random_seed)
    pairs: list[dict[str, Any]] = []
    paired = uncertainty_config.sampling_design == "paired"
    for i, a in enumerate(records):
        for j in range(i + 1, len(records)):
            b = records[j]
            ma, mb = matrices[i], matrices[j]
            differences = mb.mean(axis=0) - ma.mean(axis=0)
            boot = _bootstrap_differences(
                ma, mb, rng=rng, n_bootstrap=uncertainty_config.n_bootstrap,
                paired=paired,
            )
            cost_row = _metric(differences, 0, boot, uncertainty_config.confidence_level,
                               beneficial_direction=-1)
            cost_row["single_realization_probability_b_cheaper_with_half_ties"] = (
                _single_realization_win_probability(ma[:, 0], mb[:, 0],
                                                    paired=paired, larger_is_better=False))
            target_rows = []
            for idx, target in enumerate(target_config.targets):
                mass_index = 1 + idx * 2
                mass = _metric(differences, mass_index, boot, uncertainty_config.confidence_level,
                               beneficial_direction=1)
                mass["single_realization_probability_b_greater_with_half_ties"] = (
                    _single_realization_win_probability(
                        ma[:, mass_index], mb[:, mass_index], paired=paired,
                        larger_is_better=True))
                probability = _metric(differences, mass_index + 1, boot,
                                      uncertainty_config.confidence_level, beneficial_direction=1)
                target_rows.append({
                    "name": target.name,
                    "signed_annual_removed_mass_kg": mass,
                    "attainment_probability": probability,
                })
            next_index = 1 + 2 * len(target_config.targets)
            joint = _metric(differences, next_index, boot, uncertainty_config.confidence_level,
                            beneficial_direction=1)
            next_index += 1
            budget = None
            if target_config.budget_usd is not None:
                budget = {
                    "within_budget_probability": _metric(
                        differences, next_index, boot, uncertainty_config.confidence_level,
                        beneficial_direction=1),
                    "joint_targets_within_budget_probability": _metric(
                        differences, next_index + 1, boot, uncertainty_config.confidence_level,
                        beneficial_direction=1),
                }
            pairs.append({
                "strategy_a": a["spec"].name, "strategy_b": b["spec"].name,
                "n_a": len(ma), "n_b": len(mb),
                "difference_definition": "B minus A",
                "implementation_cost_usd": cost_row,
                "targets": target_rows,
                "joint_attainment_probability": joint,
                "budget_outcomes": budget,
            })
    return {
        "uncertainty_schema_version": UNCERTAINTY_RESULT_SCHEMA_VERSION,
        "scientific_model_id": comparison["scientific_model_id"],
        "sampling_design": uncertainty_config.sampling_design,
        "sampling_design_confirmed": True,
        "pairing_id": uncertainty_config.pairing_id,
        "bootstrap": {
            "method": "nonparametric percentile",
            "resamples": uncertainty_config.n_bootstrap,
            "confidence_level": uncertainty_config.confidence_level,
            "random_seed": uncertainty_config.random_seed,
        },
        "primary_target": comparison["primary_target"],
        "targets": comparison["targets"],
        "budget_usd": comparison["budget_usd"],
        "strategies": [{
            "name": row["name"], "backend": row["backend"],
            "n_scenarios": row["n_scenarios"], "source_sha256": row["source_sha256"],
        } for row in comparison["strategies"]],
        "pairwise_differences": pairs,
        "interpretation": {
            "scope": "Conditional Monte Carlo sampling precision only: no structural/model, forcing, input-data, or observational uncertainty bounds.",
            "differences": "All differences are strategy B minus strategy A. Positive mass or attainment differences favor B; negative cost differences favor B.",
            "bootstrap_fraction": "Fraction of bootstrap mean differences favorable to B (half credit for ties); NOT a posterior probability that B is scientifically better.",
            "single_realization_probability": "Empirical P(B outperforms A) with half credit for ties: all cross-product draws for independent designs, matched scenario indices for confirmed paired designs.",
            "pairing": "Paired results require both analyst-confirmed common underlying draws and matching ordered scenario IDs/seeds. Matching seeds alone do not prove common-random-number pairing.",
            "assumptions": "Independent bootstrap requires independent realizations within and across strategies; paired bootstrap requires independent pairs across scenario IDs. Neither independence nor scientific comparability is established by metadata alone.",
            "multiplicity": "Intervals are unadjusted for the number of strategies, targets, and comparisons. Descriptive comparisons are not formal multiple-hypothesis significance tests.",
        },
    }


def render_primary_target_intervals_svg(report: dict[str, Any]) -> str:
    """Deterministic interval plot for signed annual removal mean differences."""
    primary = report["primary_target"]
    rows = report["pairwise_differences"]
    if len(rows) > 60:
        raise ValueError("SVG interval chart supports at most 60 strategy pairs")
    vals = [0.0]
    for row in rows:
        metric = next(t["signed_annual_removed_mass_kg"] for t in row["targets"]
                      if t["name"] == primary)
        vals.extend(metric["bootstrap_percentile_interval"])
    low, high = min(vals), max(vals)
    span = high - low
    pad = max(1e-9, 0.07 * span)
    low -= pad
    high += pad
    width, height = 1050, max(220, 110 + 36 * len(rows))
    x = lambda v: 300 + 690 * (v - low) / (high - low)
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="{xml_escape("Bootstrap differences in mean removal", quote=True)}">',
             '<rect width="100%" height="100%" fill="white"/>',
             f'<title>Monte Carlo intervals for {xml_escape(primary)}</title>',
             f'<text x="300" y="35" font-family="sans-serif" font-size="20">Difference in mean annual removal (B minus A)</text>',
             f'<line x1="{x(0):.2f}" y1="60" x2="{x(0):.2f}" y2="{height-40}" stroke="#64748b" stroke-dasharray="5 5"/>']
    for k, row in enumerate(rows):
        metric = next(t["signed_annual_removed_mass_kg"] for t in row["targets"]
                      if t["name"] == primary)
        lower, upper = metric["bootstrap_percentile_interval"]
        y = 75 + 36 * k
        title = f'{row["strategy_b"]} minus {row["strategy_a"]}'
        parts.append(f'<text x="290" y="{y+4}" text-anchor="end" font-family="sans-serif" font-size="12">{xml_escape(title)}</text>')
        parts.append(f'<path d="M{x(lower):.2f} {y} H{x(upper):.2f}" stroke="#2563eb" stroke-width="3"/>')
        for v in (lower, upper):
            parts.append(f'<path d="M{x(v):.2f} {y-5} V{y+5}" stroke="#2563eb" stroke-width="2"/>')
        parts.append(f'<circle cx="{x(metric["difference_b_minus_a"]):.2f}" cy="{y}" r="5" fill="#111827"/>')
    parts.append(f'<text x="300" y="{height-15}" font-family="sans-serif" font-size="12">Signed annual mass removed (kg); {report["bootstrap"]["confidence_level"]:.1%} percentile intervals (MC precision only)</text>')
    parts.append('</svg>')
    return '\n'.join(parts) + '\n'


def write_strategy_uncertainty(
    comparison_config_file: str | Path,
    target_config_file: str | Path,
    uncertainty_config_file: str | Path,
    destination: str | Path,
    *,
    chart_path: str | Path | None = None,
) -> Path:
    """Publish a separate report; never overwrite sources or config files."""
    configs = [Path(p).expanduser().resolve(strict=True) for p in (
        comparison_config_file, target_config_file, uncertainty_config_file)]
    dest = Path(destination).expanduser().resolve()
    selection = StrategyComparisonConfig.read(configs[0])
    forbidden = set(configs) | {s.ensemble_result for s in selection.strategies}
    chart = None if chart_path is None else Path(chart_path).expanduser().resolve()
    if dest in forbidden or (chart is not None and (chart in forbidden or chart == dest)):
        raise ValueError("Uncertainty report/chart destinations must not overwrite source/configuration files")
    report = analyze_strategy_uncertainty(
        selection, DecisionSupportConfig.read(configs[1]), UncertaintyConfig.read(configs[2]))
    svg = render_primary_target_intervals_svg(report) if chart is not None else None
    _atomic_json(dest, report)
    if chart is not None:
        _atomic_text(chart, svg)
    return dest


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Phase 16: bootstrap uncertainty for comparable BMP strategies")
    parser.add_argument("comparison_config", type=Path)
    parser.add_argument("decision_support_config", type=Path)
    parser.add_argument("uncertainty_config", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--chart", type=Path, help="Optional primary-target interval SVG")
    args = parser.parse_args()
    print(write_strategy_uncertainty(
        args.comparison_config, args.decision_support_config, args.uncertainty_config,
        args.result, chart_path=args.chart))


if __name__ == "__main__":
    main()
