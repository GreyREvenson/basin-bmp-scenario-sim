"""Phase 15: compare completed, model-neutral BMI ensembles without rerunning science.

Pareto frontiers are descriptive two-objective summaries of *strategy means*,
not policy optimization, statistical significance tests or proof of scientific
interchangeability. Independently generated ensembles are never treated as
paired observations, even if their seeds coincide.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from html import escape as xml_escape
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import os
from typing import Any

import numpy as np

from basin_bmp_sim.analysis.ensemble_analysis import (
    DecisionSupportConfig, _validated_rows, analyze_ensemble,
)
from basin_bmp_sim.execution.ensemble_checkpoints import _atomic_json

COMPARISON_CONFIG_SCHEMA_VERSION = 1
COMPARISON_RESULT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StrategySource:
    name: str
    ensemble_result: Path


@dataclass(frozen=True)
class StrategyComparisonConfig:
    strategies: tuple[StrategySource, ...]
    primary_target: str | None = None
    comparison_mode: str = "descriptive"
    scientific_comparability_confirmed: bool = False
    scientific_model_id: str | None = None

    @classmethod
    def read(cls, filename: str | Path) -> "StrategyComparisonConfig":
        source = Path(filename).expanduser().resolve(strict=True)
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Invalid strategy comparison JSON") from exc
        known = {"schema_version", "strategies", "primary_target", "comparison_mode",
                 "scientific_comparability_confirmed", "scientific_model_id"}
        if (not isinstance(data, dict) or set(data) - known or
                type(data.get("schema_version")) is not int or
                data["schema_version"] != COMPARISON_CONFIG_SCHEMA_VERSION):
            raise ValueError("Unsupported strategy comparison schema or unknown fields")
        incoming = data.get("strategies")
        if not isinstance(incoming, list) or len(incoming) < 2:
            raise ValueError("Comparison requires at least two strategies")
        strategies, names, paths = [], set(), set()
        for i, item in enumerate(incoming):
            if not isinstance(item, dict) or set(item) != {"name", "ensemble_result"}:
                raise ValueError(f"Strategy {i} must have name and ensemble_result only")
            name, filename = item["name"], item["ensemble_result"]
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Strategy {i} name must be nonempty")
            if not isinstance(filename, str) or not filename.strip():
                raise ValueError(f"Strategy {i} ensemble_result must be a file path")
            if name in names:
                raise ValueError(f"Duplicate strategy name: {name}")
            path = (source.parent / filename).resolve(strict=True)
            if not path.is_file():
                raise ValueError(f"Strategy {i} ensemble_result must be a file")
            if path in paths:
                raise ValueError("Each strategy must have a distinct ensemble result file")
            names.add(name)
            paths.add(path)
            strategies.append(StrategySource(name, path))
        mode = data.get("comparison_mode", "descriptive")
        if mode not in ("descriptive", "comparable") or not isinstance(mode, str):
            raise ValueError("comparison_mode must be descriptive or comparable")
        asserted = data.get("scientific_comparability_confirmed", False)
        if type(asserted) is not bool:
            raise ValueError("scientific_comparability_confirmed must be boolean")
        model_id = data.get("scientific_model_id")
        if model_id is not None and (not isinstance(model_id, str) or not model_id.strip()):
            raise ValueError("scientific_model_id must be nonempty when supplied")
        if mode == "comparable" and (not asserted or model_id is None):
            raise ValueError("Comparable frontiers require confirmed scientific comparability and scientific_model_id")
        if mode == "descriptive" and asserted:
            raise ValueError("Comparability confirmation is only valid in comparable mode")
        primary = data.get("primary_target")
        if primary is not None and (not isinstance(primary, str) or not primary.strip()):
            raise ValueError("primary_target must be a nonempty target name")
        return cls(tuple(strategies), primary, mode, asserted, model_id)


def _pareto_flags(costs: list[float], benefits: list[float]) -> list[bool]:
    """Minimize mean cost and maximize benefit; ties do not dominate."""
    return [not any(
        j != i and costs[j] <= costs[i] and benefits[j] >= benefits[i] and
        (costs[j] < costs[i] or benefits[j] > benefits[i])
        for j in range(len(costs))) for i in range(len(costs))]


def _load_and_compare(config: StrategyComparisonConfig,
                      target_config: DecisionSupportConfig) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate complete source files and calculate descriptive comparisons."""
    if config.primary_target is not None and config.primary_target not in {t.name for t in target_config.targets}:
        raise ValueError("primary_target must match a target name from decision-support config")
    primary = config.primary_target or target_config.targets[0].name
    records: list[dict[str, Any]] = []
    for spec in config.strategies:
        data_bytes = spec.ensemble_result.read_bytes()
        try:
            ensemble = json.loads(data_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Invalid ensemble JSON for strategy {spec.name!r}") from exc
        manifest, costs, removals = _validated_rows(ensemble)
        analysis = analyze_ensemble(ensemble, target_config, source_sha256=sha256(data_bytes).hexdigest())
        target_values = {}
        for target, analysis_row in zip(target_config.targets, analysis["targets"]):
            col = (manifest["pollutants"].index(target.pollutant) * len(manifest["outlet_oids"])
                   + manifest["outlet_oids"].index(target.outlet_oid))
            target_values[target.name] = removals[:, col].copy()
        records.append({"spec": spec, "manifest": manifest, "backend": ensemble["backend"],
                        "analysis": analysis, "costs": costs, "target_values": target_values,
                        # Phase 16 uses these validated identities to distinguish
                        # independently seeded runs from explicitly paired designs.
                        "scenario_ids": tuple(r["scenario_id"] for r in ensemble["scenario_summaries"]),
                        "worker_seeds": tuple(r["worker_seed"] for r in ensemble["scenario_summaries"])})

    base = records[0]
    pollutant_outlets = (base["manifest"]["pollutants"], base["manifest"]["outlet_oids"])
    for record in records[1:]:
        ids = (record["manifest"]["pollutants"], record["manifest"]["outlet_oids"])
        if ids != pollutant_outlets:
            raise ValueError("Cannot compare different pollutant or outlet identities/order")
    if config.comparison_mode == "comparable":
        for record in records[1:]:
            if record["backend"] != base["backend"] or record["manifest"] != base["manifest"]:
                raise ValueError("Comparable Pareto results require identical backend and full index manifest")

    strategy_rows = []
    for record in records:
        analysis = record["analysis"]
        strategy_rows.append({
            "name": record["spec"].name,
            "ensemble_result": str(record["spec"].ensemble_result),
            "source_sha256": analysis["source_ensemble_sha256"],
            "backend": record["backend"],
            "base_seed": analysis["base_seed"],
            "n_scenarios": analysis["n_scenarios"],
            "mean_implementation_cost_usd": float(np.mean(record["costs"])),
            "implementation_cost_usd": analysis["implementation_cost_usd"],
            "within_budget": analysis["within_budget"],
            "joint_all_targets": analysis["joint_all_targets"],
            "targets": [{
                "name": row["name"], "pollutant": row["pollutant"],
                "outlet_oid": row["outlet_oid"],
                "minimum_removed_mass_kg": row["minimum_removed_mass_kg"],
                "attainment": row["attainment"],
                "signed_removed_mass_kg": row["signed_removed_mass_kg"],
                "cost_effectiveness": row["cost_effectiveness"],
                **({"attainment_within_budget": row["attainment_within_budget"],
                    "conditional_attainment_given_within_budget": row["conditional_attainment_given_within_budget"]}
                   if "attainment_within_budget" in row else {}),
            } for row in analysis["targets"]],
        })
    costs = [row["mean_implementation_cost_usd"] for row in strategy_rows]
    fronts = {}
    if config.comparison_mode == "comparable":
        for target in target_config.targets:
            t_idx = next(i for i, t in enumerate(target_config.targets) if t.name == target.name)
            masses = [row["targets"][t_idx]["signed_removed_mass_kg"]["mean"] for row in strategy_rows]
            probabilities = [row["targets"][t_idx]["attainment"]["probability"] for row in strategy_rows]
            fronts[target.name] = {
                "mean_cost_vs_mean_annual_kg": dict(zip(
                    [s.name for s in config.strategies], _pareto_flags(costs, masses))),
                "mean_cost_vs_attainment_probability": dict(zip(
                    [s.name for s in config.strategies], _pareto_flags(costs, probabilities))),
            }
    return {
        "comparison_schema_version": COMPARISON_RESULT_SCHEMA_VERSION,
        "comparison_mode": config.comparison_mode,
        "scientific_comparability_confirmed": config.scientific_comparability_confirmed,
        "scientific_model_id": config.scientific_model_id,
        "primary_target": primary,
        "targets": [vars(target) for target in target_config.targets],
        "budget_usd": target_config.budget_usd,
        "pollutants": pollutant_outlets[0], "outlet_oids": pollutant_outlets[1],
        "strategies": strategy_rows, "pareto_frontiers": fronts if fronts else None,
        "interpretation": {
            "quantities": "Signed annual outlet mass removed (kg), and implementation costs (USD); not ambient concentrations or lifecycle costs.",
            "comparison": "Unpaired descriptive statistics conditional on each ensemble and its scientific backend; shared seeds do not imply paired outcomes.",
            "pareto": "Non-dominated STRATEGY MEANS only (minimize mean cost; maximize mean removal or attainment probability); no inference about individual realization dominance, statistical significance, or optimality.",
            "comparability": "Matching pollutant/outlet labels are not proof of matching geography, forcing, loading definitions, uncertainty or model validity. Comparable mode requires an explicit analyst assertion plus equal backend and full index manifest; it does not verify scientific equivalence.",
        },
    }, records


_PALETTE = ("#2563eb", "#d97706", "#059669", "#9333ea", "#dc2626", "#0891b2", "#be185d", "#475569")


def _scale(values: list[float], margin: float = 0.07) -> tuple[float, float]:
    low, high = float(min(values)), float(max(values))
    if low == high:
        pad = max(1.0, abs(low) * 0.1)
    else:
        pad = (high - low) * margin
    return low - pad, high + pad


def _svg_frame(title: str, x_label: str, y_label: str, body: list[str], legend: list[str]) -> str:
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1000 660" role="img" aria-label="'+xml_escape(title, quote=True)+'">',
             '<rect width="1000" height="660" fill="white"/>',
             f'<title>{xml_escape(title)}</title>',
             '<text x="90" y="45" font-family="sans-serif" font-size="24" font-weight="bold">'+xml_escape(title)+'</text>',
             '<path d="M90 80 V550 H820" fill="none" stroke="#334155" stroke-width="2"/>',
             f'<text x="460" y="610" text-anchor="middle" font-family="sans-serif" font-size="16">{xml_escape(x_label)}</text>',
             f'<text transform="translate(27 320) rotate(-90)" text-anchor="middle" font-family="sans-serif" font-size="16">{xml_escape(y_label)}</text>']
    lines.extend(body)
    lines.extend(legend)
    lines.append('</svg>')
    return '\n'.join(lines) + '\n'


def _legend(rows: list[dict[str, Any]]) -> list[str]:
    entries = []
    for i, row in enumerate(rows):
        y = 95 + 31 * i
        entries.append(f'<circle cx="855" cy="{y}" r="6" fill="{_PALETTE[i % len(_PALETTE)]}"/>')
        entries.append(f'<text x="870" y="{y+5}" font-family="sans-serif" font-size="12">{xml_escape(row["name"])}</text>')
    return entries


def _tick_labels(xlo: float, xhi: float, ylo: float, yhi: float) -> list[str]:
    result = []
    for i in range(5):
        x = 90 + 730 * i / 4
        y = 550 - 470 * i / 4
        xv = xlo + (xhi - xlo) * i / 4
        yv = ylo + (yhi - ylo) * i / 4
        result.extend((
            f'<text x="{x:.1f}" y="575" text-anchor="middle" font-family="sans-serif" font-size="11">{xv:.4g}</text>',
            f'<text x="80" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{yv:.3g}</text>',
            f'<path d="M{x:.1f} 550 V80" stroke="#e2e8f0" stroke-width="1"/>',
            f'<path d="M90 {y:.1f} H820" stroke="#e2e8f0" stroke-width="1"/>',
        ))
    return result


def render_cost_removal_svg(report: dict[str, Any]) -> str:
    """One point per strategy: mean implementation cost vs mean signed removal."""
    primary = report["primary_target"]
    target = next(t for t in report["targets"] if t["name"] == primary)
    t_idx = [t["name"] for t in report["targets"]].index(primary)
    rows = report["strategies"]
    xs = [float(s["mean_implementation_cost_usd"]) for s in rows]
    ys = [float(s["targets"][t_idx]["signed_removed_mass_kg"]["mean"]) for s in rows]
    xlo, xhi = _scale(xs + [0.0])
    ylo, yhi = _scale(ys + [target["minimum_removed_mass_kg"]])
    def xy(x, y):
        return (90 + 730 * (x - xlo) / (xhi - xlo), 550 - 470 * (y - ylo) / (yhi - ylo))
    body = _tick_labels(xlo, xhi, ylo, yhi)
    _, yt = xy(0, target["minimum_removed_mass_kg"])
    body.append(f'<path d="M90 {yt:.2f} H820" stroke="#64748b" stroke-dasharray="5 5" fill="none"/>')
    front = report["pareto_frontiers"]
    for i, (s, x, y) in enumerate(zip(rows, xs, ys)):
        xp, yp = xy(x, y)
        efficient = front is not None and front[primary]["mean_cost_vs_mean_annual_kg"][s["name"]]
        body.append(f'<circle cx="{xp:.2f}" cy="{yp:.2f}" r="{9 if efficient else 6}" fill="{_PALETTE[i%len(_PALETTE)]}" stroke="{ "#111827" if efficient else "white" }" stroke-width="2"><title>{xml_escape(s["name"])}: USD {x:.6g}, kg {y:.6g}</title></circle>')
    body.append('<text x="90" y="638" font-family="sans-serif" font-size="12" fill="#475569">Dashed: specified annual removal target; black ring: non-dominated strategy mean (when enabled)</text>')
    return _svg_frame(f'Strategy means — {primary}', 'Mean implementation cost (USD)',
                      'Mean signed annual outlet removal (kg)', body, _legend(rows))


def render_attainment_curve_svg(report: dict[str, Any], records: list[dict[str, Any]]) -> str:
    """Empirical exceedance curves, with exactly computed probabilities at sampled thresholds."""
    target = next(t for t in report["targets"] if t["name"] == report["primary_target"])
    values = [record["target_values"][report["primary_target"]] for record in records]
    all_values = [float(v) for a in values for v in a]
    xlo, xhi = _scale(all_values + [target["minimum_removed_mass_kg"]])
    xs = np.linspace(xlo, xhi, 121)
    body = _tick_labels(xlo, xhi, 0.0, 1.0)
    tx = 90 + 730 * (target["minimum_removed_mass_kg"] - xlo) / (xhi - xlo)
    body.append(f'<path d="M{tx:.2f} 80 V550" stroke="#64748b" stroke-dasharray="5 5"/>')
    for i, a in enumerate(values):
        sorted_values = np.sort(a)
        probabilities = 1 - np.searchsorted(sorted_values, xs, side="left") / len(sorted_values)
        coords = ' '.join(f'{90 + 730*(x-xlo)/(xhi-xlo):.2f},{550-470*p:.2f}'
                          for x, p in zip(xs, probabilities))
        body.append(f'<polyline points="{coords}" stroke="{_PALETTE[i%len(_PALETTE)]}" stroke-width="3" fill="none"><title>{xml_escape(records[i]["spec"].name)}</title></polyline>')
    body.append('<text x="90" y="638" font-family="sans-serif" font-size="12" fill="#475569">Fraction of independent Monte Carlo realizations meeting each threshold; dashed: configured target</text>')
    return _svg_frame(f'Annual reduction attainment — {report["primary_target"]}',
                      'Minimum signed annual removal threshold (kg)',
                      'Fraction meeting threshold', body, _legend(report["strategies"]))


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                prefix=f'.{path.name}.', suffix='.tmp', delete=False) as file:
            temporary = Path(file.name)
            file.write(value)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def write_strategy_comparison(comparison_config: str | Path, target_config: str | Path,
                              destination: str | Path, *, charts_dir: str | Path | None = None) -> Path:
    """Read and validate all sources before publishing a deterministic report/charts.

    This is post-processing: it NEVER modifies an ensemble, replay or checkpoint.
    Destinations are written atomically per file, not as a multi-file transaction.
    """
    config_path = Path(comparison_config).expanduser().resolve(strict=True)
    target_path = Path(target_config).expanduser().resolve(strict=True)
    dest = Path(destination).expanduser().resolve()
    config = StrategyComparisonConfig.read(config_path)
    targets = DecisionSupportConfig.read(target_path)
    forbidden = {config_path, target_path, *(s.ensemble_result for s in config.strategies)}
    figures = [] if charts_dir is None else [
        (Path(charts_dir).expanduser().resolve() / "cost_vs_reduction.svg"),
        (Path(charts_dir).expanduser().resolve() / "attainment_curves.svg"),
    ]
    if dest in forbidden or any(p in forbidden or p == dest for p in figures):
        raise ValueError("Comparison destinations must not overwrite source or configuration files")
    report, records = _load_and_compare(config, targets)
    graphics = ([render_cost_removal_svg(report), render_attainment_curve_svg(report, records)]
                if figures else [])
    _atomic_json(dest, report)
    for figure, contents in zip(figures, graphics):
        _atomic_text(figure, contents)
    return dest


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="Phase 15: compare completed BMI strategy ensembles")
    parser.add_argument("comparison_config", type=Path)
    parser.add_argument("decision_support_config", type=Path)
    parser.add_argument("comparison_result", type=Path)
    parser.add_argument("--charts-dir", type=Path, default=None,
                        help="Optionally write cost_vs_reduction.svg and attainment_curves.svg")
    args = parser.parse_args()
    print(write_strategy_comparison(args.comparison_config, args.decision_support_config,
                                    args.comparison_result, charts_dir=args.charts_dir))


if __name__ == "__main__":
    main()
