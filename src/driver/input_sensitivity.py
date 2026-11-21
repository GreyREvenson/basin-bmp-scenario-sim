"""Phase 17: auditable *exploratory* input/output association diagnostics.

A separately supplied, realization-aligned trace is required. The Phase 12/13
ensemble result does not expose all sampled model inputs, so deriving global
sensitivity or variance attribution from it would be scientifically unjustified.
This module never executes the model, resamples input uncertainties, or alters
simulation, replay, checkpoint, or earlier decision-support outputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from html import escape as xml_escape
import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.driver.ensemble_analysis import DecisionSupportConfig, _validated_rows
from src.driver.ensemble_checkpoints import _atomic_json
from src.driver.ensemble_comparison import _atomic_text

TRACE_SCHEMA_VERSION = 1
SENSITIVITY_CONFIG_SCHEMA_VERSION = 1
SENSITIVITY_RESULT_SCHEMA_VERSION = 1


def _strict_number(value: Any, label: str) -> float:
    if type(value) not in (float, int):
        raise ValueError(f"{label} must be a finite JSON number (not a boolean)")
    try:
        number = float(value)
    except (OverflowError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite JSON number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite JSON number")
    return number


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda s: (_ for _ in ()).throw(ValueError(s)))
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Invalid {label} JSON") from exc
    if not isinstance(data, dict):
        raise ValueError(f"{label} must be a JSON object")
    return data


@dataclass(frozen=True)
class SensitivityConfig:
    """Bootstrap configuration for *associations*, not variance decomposition."""

    n_bootstrap: int = 400
    confidence_level: float = 0.95
    random_seed: int = 0
    primary_target: str | None = None

    @classmethod
    def read(cls, path: str | Path) -> "SensitivityConfig":
        data = _read_json(Path(path).expanduser().resolve(strict=True), "sensitivity configuration")
        allowed = {"schema_version", "n_bootstrap", "confidence_level", "random_seed", "primary_target"}
        if (set(data) - allowed or type(data.get("schema_version")) is not int or
                data["schema_version"] != SENSITIVITY_CONFIG_SCHEMA_VERSION):
            raise ValueError("Unsupported sensitivity configuration schema or unknown fields")
        n = data.get("n_bootstrap", 400)
        if type(n) is not int or not 200 <= n <= 5000:
            raise ValueError("n_bootstrap must be an integer from 200 to 5000")
        level = _strict_number(data.get("confidence_level", 0.95), "confidence_level")
        if not 0.5 <= level < 1.0:
            raise ValueError("confidence_level must be in [0.5, 1)")
        seed = data.get("random_seed", 0)
        if type(seed) is not int or not 0 <= seed < 2**64:
            raise ValueError("random_seed must be a uint64 integer")
        primary = data.get("primary_target")
        if primary is not None and (type(primary) is not str or not primary.strip()):
            raise ValueError("primary_target must be a nonempty string or omitted")
        return cls(n, level, seed, primary)


def _input_trace(trace: Mapping[str, Any], ensemble: Mapping[str, Any]) -> tuple[list[dict[str, str]], np.ndarray]:
    """Match every recorded draw to the ensemble's actual IDs *and* worker seeds."""
    if (not isinstance(trace, dict) or set(trace) !=
            {"schema_version", "trace_provenance", "sampling_design",
             "sampling_design_confirmed", "features", "realizations"} or
            type(trace.get("schema_version")) is not int or
            trace["schema_version"] != TRACE_SCHEMA_VERSION):
        raise ValueError("Unsupported input trace schema or unknown/missing fields")
    if (trace["sampling_design"] != "independent" or
            trace["sampling_design_confirmed"] is not True):
        raise ValueError("Bootstrap sensitivity requires an explicitly confirmed independent-realization design")
    provenance = trace["trace_provenance"]
    if (type(provenance) is not str or not provenance.strip()):
        raise ValueError("trace_provenance must document how input values were recorded")
    raw_features = trace["features"]
    if not isinstance(raw_features, list) or not 1 <= len(raw_features) <= 100:
        raise ValueError("input trace must contain 1 to 100 feature definitions")
    names: set[str] = set()
    features: list[dict[str, str]] = []
    for i, feature in enumerate(raw_features):
        if not isinstance(feature, dict) or set(feature) != {"name", "units", "role"}:
            raise ValueError(f"Feature {i} must contain exactly name, units, and role")
        name, units, role = feature["name"], feature["units"], feature["role"]
        if (type(name) is not str or not name.strip() or name in names or
                type(units) is not str or not units.strip() or
                role not in ("sampled_input", "realized_diagnostic")):
            raise ValueError(f"Invalid or duplicated feature metadata at index {i}")
        names.add(name)
        features.append({"name": name, "units": units, "role": role})
    realizations = trace["realizations"]
    rows = ensemble["scenario_summaries"]
    if not isinstance(realizations, list) or len(realizations) != len(rows):
        raise ValueError("Input trace must have one realization per ensemble scenario")
    X = np.empty((len(rows), len(features)), dtype=np.float64)
    for i, (item, expected) in enumerate(zip(realizations, rows)):
        if (not isinstance(item, dict) or set(item) != {"scenario_id", "worker_seed", "values"} or
                type(item["scenario_id"]) is not int or
                type(item["worker_seed"]) is not int or
                item["scenario_id"] != expected["scenario_id"] or
                item["worker_seed"] != expected["worker_seed"]):
            raise ValueError(f"Input trace scenario identity/worker seed mismatch at index {i}")
        values = item["values"]
        if not isinstance(values, list) or len(values) != len(features):
            raise ValueError(f"Input trace feature count mismatch at index {i}")
        X[i, :] = [_strict_number(value, f"Trace realization {i} feature {j}")
                    for j, value in enumerate(values)]
    return features, X


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Stable, tie-aware midranks without SciPy or arbitrary jitter."""
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    ranks = np.empty(values.size, dtype=np.float64)
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1]) + 1]
    ends = np.r_[starts[1:], values.size]
    for start, end in zip(starts, ends):
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float | None:
    """Spearman rho with midranks and null for zero-variance input/output."""
    if len(a) < 3:
        return None
    ra, rb = _average_ranks(a), _average_ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = float(np.linalg.norm(ra) * np.linalg.norm(rb))
    return None if denom == 0.0 else float(np.clip(np.dot(ra, rb) / denom, -1.0, 1.0))


def _association(x: np.ndarray, y: np.ndarray, *, config: SensitivityConfig,
                 rng: np.random.Generator) -> dict[str, Any]:
    coefficient = _spearman(x, y)
    if coefficient is None:
        return {
            "spearman_rho": None, "bootstrap_percentile_interval": None,
            "valid_bootstrap_replicates": 0, "undefined_bootstrap_replicates": config.n_bootstrap,
        }
    draws: list[float] = []
    for _ in range(config.n_bootstrap):
        indices = rng.integers(0, len(x), size=len(x))
        value = _spearman(x[indices], y[indices])
        if value is not None:
            draws.append(value)
    # A percentile interval is not reported if too many bootstrap draws are
    # undefined (e.g. an attainment event occurs in only one realization).
    valid = len(draws)
    alpha = (1 - config.confidence_level) / 2
    interval = ([float(t) for t in np.quantile(draws, [alpha, 1 - alpha])]
                if valid >= math.ceil(config.n_bootstrap * 0.95) else None)
    return {
        "spearman_rho": coefficient, "bootstrap_percentile_interval": interval,
        "valid_bootstrap_replicates": valid,
        "undefined_bootstrap_replicates": config.n_bootstrap - valid,
    }


def analyze_input_sensitivity(ensemble: Mapping[str, Any], trace: Mapping[str, Any],
                              targets: DecisionSupportConfig, config: SensitivityConfig,
                              *, ensemble_sha256: str, trace_sha256: str) -> dict[str, Any]:
    """Associate recorded per-realization features with signed annual outcomes.

    Results are unconditional observational associations and never described as
    Sobol indices, variance fractions, independent effects, or causal drivers.
    """
    if not isinstance(config, SensitivityConfig) or not isinstance(targets, DecisionSupportConfig):
        raise ValueError("Expected SensitivityConfig and DecisionSupportConfig")
    rows = ensemble.get("scenario_summaries") if isinstance(ensemble, dict) else None
    if not isinstance(rows, list) or len(rows) < 4:
        raise ValueError("Input sensitivity requires at least four realizations")
    manifest, costs, mass = _validated_rows(ensemble)
    features, X = _input_trace(trace, ensemble)
    names = [target.name for target in targets.targets]
    if config.primary_target is not None and config.primary_target not in names:
        raise ValueError("primary_target does not match a configured target")
    primary = config.primary_target or names[0]
    pollutants, outlets = manifest["pollutants"], manifest["outlet_oids"]
    outcomes: list[tuple[str, str, np.ndarray]] = [("implementation_cost_usd", "USD", costs)]
    for target in targets.targets:
        if target.pollutant not in pollutants or target.outlet_oid not in outlets:
            raise ValueError(f"Unknown pollutant or outlet for target {target.name!r}")
        column = pollutants.index(target.pollutant) * len(outlets) + outlets.index(target.outlet_oid)
        annual = mass[:, column]
        outcomes.append((f"signed_annual_removed_mass_kg::{target.name}", "kg", annual))
        outcomes.append((f"target_attained::{target.name}", "boolean (0 or 1)",
                         (annual >= target.minimum_removed_mass_kg).astype(np.float64)))
    rng = np.random.default_rng(config.random_seed)
    outcome_rows: list[dict[str, Any]] = []
    for name, units, y in outcomes:
        entries = []
        for j, feature in enumerate(features):
            entries.append({"feature": feature["name"], **_association(X[:, j], y, config=config, rng=rng)})
        outcome_rows.append({"outcome": name, "units": units,
                             "constant_outcome": bool(np.all(y == y[0])),
                             "associations": entries})
    feature_correlations = []
    for i in range(len(features)):
        for j in range(i + 1, len(features)):
            feature_correlations.append({
                "feature_a": features[i]["name"], "feature_b": features[j]["name"],
                "spearman_rho": _spearman(X[:, i], X[:, j]),
            })
    return {
        "sensitivity_schema_version": SENSITIVITY_RESULT_SCHEMA_VERSION,
        "source_ensemble_sha256": ensemble_sha256,
        "input_trace_sha256": trace_sha256,
        "n_scenarios": len(costs),
        "backend": ensemble["backend"],
        "seed_policy": ensemble["seed_policy"],
        "trace_provenance": trace["trace_provenance"],
        "sampling_design": "independent",
        "sampling_design_confirmed": True,
        "primary_target": primary,
        "n_bootstrap": config.n_bootstrap,
        "confidence_level": config.confidence_level,
        "random_seed": config.random_seed,
        "features": features,
        "outcomes": outcome_rows,
        "input_input_correlations": feature_correlations,
        "interpretation": {
            "method": "Unconditional Spearman rank association (tie-aware); paired-row percentile bootstrap over realizations.",
            "scope": "Exploratory associations within the sampled ensemble, not Sobol indices, variance attribution, a causal effect, or independent parameter importance.",
            "constraints": "Confounding, correlated parameters, selection/implementation feedback, structural uncertainty, and stochastic conditional sampling can affect associations. Source roles and trace provenance are analyst-declared, not independently verified.",
            "intervals": "Bootstrap intervals quantify finite-ensemble precision conditional on recorded draws; null indicates zero variance or insufficient valid resamples. Not simultaneous/multiplicity-adjusted.",
        },
    }


def render_sensitivity_svg(report: Mapping[str, Any]) -> str:
    """Deterministic, XML-escaped, signed-rho dot plot for the primary target."""
    wanted = f'signed_annual_removed_mass_kg::{report["primary_target"]}'
    outcome = next(x for x in report["outcomes"] if x["outcome"] == wanted)
    values = [(r["feature"], r["spearman_rho"]) for r in outcome["associations"]]
    if len(values) > 40:
        raise ValueError("Sensitivity SVG supports at most 40 features; omit --chart for more")
    values.sort(key=lambda row: (row[1] is None, -abs(row[1] or 0), row[0]))
    height = max(250, 145 + len(values) * 28)
    title = f'Exploratory rank association — {report["primary_target"]}'
    lines = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 {height}" role="img" aria-label="{xml_escape(title, quote=True)}">',
             f'<rect width="1000" height="{height}" fill="white"/>',
             f'<title>{xml_escape(title)}</title>',
             f'<text x="40" y="38" font-family="sans-serif" font-size="22">{xml_escape(title)}</text>',
             f'<path d="M610 67 V{height-48}" stroke="#64748b" stroke-dasharray="4 4"/>']
    for i, (name, rho) in enumerate(values):
        y = 82 + i * 28
        lines.append(f'<text x="385" y="{y+4}" font-size="13" text-anchor="end" font-family="sans-serif">{xml_escape(name)}</text>')
        if rho is None:
            lines.append(f'<text x="645" y="{y+4}" font-size="12" font-family="sans-serif">undefined (constant ranks)</text>')
        else:
            px = 610 + 205 * rho
            lines.append(f'<path d="M610 {y} H{px:.2f}" stroke="#2563eb" stroke-width="3"/>')
            lines.append(f'<circle cx="{px:.2f}" cy="{y}" r="5" fill="#1d4ed8"/>')
            lines.append(f'<text x="{px+9:.2f}" y="{y-4}" font-size="11" font-family="sans-serif">{rho:+.3f}</text>')
    for number in (-1, 0, 1):
        lines.append(f'<text x="{610+205*number}" y="{height-27}" text-anchor="middle" font-size="12" font-family="sans-serif">{number}</text>')
    lines.append(f'<text x="40" y="{height-12}" font-size="11" font-family="sans-serif">Descriptive only: not causal sensitivity or variance attribution</text>')
    lines.append('</svg>')
    return '\n'.join(lines) + '\n'


def write_input_sensitivity(ensemble_file: str | Path, trace_file: str | Path,
                            targets_file: str | Path, config_file: str | Path,
                            destination: str | Path, *, chart_path: str | Path | None = None) -> Path:
    """Validate before publishing; never overwrite an input file or the ensemble."""
    sources = [Path(p).expanduser().resolve(strict=True) for p in
               (ensemble_file, trace_file, targets_file, config_file)]
    dest = Path(destination).expanduser().resolve()
    chart = None if chart_path is None else Path(chart_path).expanduser().resolve()
    if dest in sources or (chart is not None and (chart in sources or chart == dest)):
        raise ValueError("Sensitivity destinations must not overwrite source or configuration files")
    ensemble_bytes, trace_bytes = sources[0].read_bytes(), sources[1].read_bytes()
    ensemble = _read_json(sources[0], "ensemble result")
    trace = _read_json(sources[1], "input trace")
    report = analyze_input_sensitivity(
        ensemble, trace, DecisionSupportConfig.read(sources[2]), SensitivityConfig.read(sources[3]),
        ensemble_sha256=sha256(ensemble_bytes).hexdigest(),
        trace_sha256=sha256(trace_bytes).hexdigest(),
    )
    graphic = None if chart is None else render_sensitivity_svg(report)
    _atomic_json(dest, report)
    if chart is not None and graphic is not None:
        _atomic_text(chart, graphic)
    return dest


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Phase 17: exploratory realization-aligned input sensitivity")
    parser.add_argument("ensemble_result", type=Path)
    parser.add_argument("input_trace", type=Path)
    parser.add_argument("decision_support_config", type=Path)
    parser.add_argument("sensitivity_config", type=Path)
    parser.add_argument("result", type=Path)
    parser.add_argument("--chart", type=Path, default=None)
    args = parser.parse_args()
    print(write_input_sensitivity(args.ensemble_result, args.input_trace,
                                  args.decision_support_config, args.sensitivity_config,
                                  args.result, chart_path=args.chart))


if __name__ == "__main__":
    main()
