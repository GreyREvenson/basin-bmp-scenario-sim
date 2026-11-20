"""Scenario creation and replay, independent of pollutant load calculations.

Phase 2 retains the legacy RNG draw order by realizing *one placement at a
 time*.  Baseline load sampling precedes the first placement, and modeled AFO
 recovery draws precede its cost draw.  Completed scenarios can be serialized
 and replayed without any BMP selection/design/efficiency/cost random draws.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

from src.afo import feedlot_bmp_cost_quantities
from src.bmp.bmp_response_registry import response_family
from src.bmp_selection import BMPEligibilityPool
from src.constants import (
    CFG_BMP_FAIL_RATE, CFG_BMP_FAIL_REDUCTION,
    CFG_DETERMINISTIC_PARCEL_ORDER, CFG_DETERMINISTIC_BMP_TYPE_ORDER,
    DEFAULT_BMP_FAIL_REDUCTION,
    OUTPUT_WETLAND_AREA, OUTPUT_CATCHMENT_RATIO, OUTPUT_IMPACTED_PIDS,
    OUTPUT_BUFFER_AREA, OUTPUT_LINEAR_LENGTH, OUTPUT_PORTION_TREATED,
)
from src.scenarios.bmp_designs import BMPDesign, sample_bmp_design

SCENARIO_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PlacementChoice:
    parcel_index: int
    pid: str
    cps: int


@dataclass(frozen=True)
class RealizedBMPPlacement:
    """Immutable implementation parameters and stochastic response draws.

    ``cost_usd`` and ``recovery_gains`` are filled after model response to
    preserve the legacy order of AFO-recovery and cost random draws.
    """
    choice: PlacementChoice
    design: BMPDesign
    efficiency_by_pollutant: tuple[tuple[tuple[str, float], ...], ...]
    failed: bool
    recovery_effect_scale: float
    cost_usd: float | None = None
    recovery_gains: tuple[float, ...] | None = None

    def efficiency_maps(self) -> list[dict[str, float]]:
        return [dict(row) for row in self.efficiency_by_pollutant]

    def to_dict(self) -> dict[str, Any]:
        if self.cost_usd is None or self.recovery_gains is None:
            raise ValueError("Cannot serialize an incomplete BMP placement")
        return {
            "parcel_index": self.choice.parcel_index,
            "pid": self.choice.pid,
            "cps": self.choice.cps,
            "design": self.design.to_dict(),
            "efficiency_by_pollutant": [dict(row) for row in self.efficiency_by_pollutant],
            "failed": self.failed,
            "recovery_effect_scale": self.recovery_effect_scale,
            "cost_usd": self.cost_usd,
            "recovery_gains": list(self.recovery_gains),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RealizedBMPPlacement":
        cost = float(value["cost_usd"])
        if not math.isfinite(cost) or cost < 0:
            raise ValueError("Scenario placement cost must be finite and nonnegative")
        rows = tuple(tuple((str(k), float(v)) for k, v in row.items())
                     for row in value["efficiency_by_pollutant"])
        return cls(
            choice=PlacementChoice(int(value["parcel_index"]), str(value["pid"]), int(value["cps"])),
            design=BMPDesign.from_dict(value["design"]),
            efficiency_by_pollutant=rows,
            failed=bool(value["failed"]),
            recovery_effect_scale=float(value["recovery_effect_scale"]),
            cost_usd=cost,
            recovery_gains=tuple(float(v) for v in value["recovery_gains"]),
        )


@dataclass(frozen=True)
class BMPScenario:
    """Versioned realization of placements, designs, and response draws.

    Response efficiencies and AFO recovery are currently included to ensure
    exact replay. Future cross-model comparisons should separate these from
    the model-independent physical BMP scenario specification.
    """
    scenario_id: int
    seed: int
    parcel_ids_digest: str
    load_generation_mode: str
    placements: tuple[RealizedBMPPlacement, ...]
    schema_version: int = SCENARIO_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scenario_id": self.scenario_id,
            "seed": self.seed,
            "parcel_ids_digest": self.parcel_ids_digest,
            "load_generation_mode": self.load_generation_mode,
            "placements": [placement.to_dict() for placement in self.placements],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "BMPScenario":
        if int(value["schema_version"]) != SCENARIO_SCHEMA_VERSION:
            raise ValueError(f"Unsupported scenario schema version: {value['schema_version']}")
        return cls(
            scenario_id=int(value["scenario_id"]), seed=int(value["seed"]),
            parcel_ids_digest=str(value["parcel_ids_digest"]),
            load_generation_mode=str(value["load_generation_mode"]),
            placements=tuple(RealizedBMPPlacement.from_dict(v) for v in value["placements"]),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Save atomically and without numpy or private model-state objects.
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(self.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
        tmp.replace(path)

    @classmethod
    def load(cls, path: str | Path) -> "BMPScenario":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def parcel_digest(ctx: Any) -> str:
    return hashlib.sha256(json.dumps(list(map(str, ctx.parcel_ids)), separators=(",", ":")).encode()).hexdigest()


class StochasticScenarioGenerator:
    """Own BMP placement, efficiency, failure, design, and cost sampling.

    During generation each realization is completed after the model's AFO
    recovery operation.  This is necessary for exact legacy RNG equivalence;
    AFO recovery remains a model-dependent *effect*, not a placement design.
    """

    def __init__(self, ctx: Any, cfg: dict[str, Any], *,
                 replay: BMPScenario | None = None) -> None:
        self._pool = BMPEligibilityPool(
            selection_pids=ctx.parcel_selection_ids,
            bmp_cps=ctx.bmp_cps,
            bmp_probabilities=ctx.bmp_selection_probs,
            feedlot_bmp_cps=ctx.feedlot_bmp_cps,
            afo_host_pids=(op["pid"] for op in (ctx.afo_inputs or {}).get("operations", [])),
            parcel_probabilities=ctx.parcel_selection_probs,
            deterministic_parcels=bool(cfg.get(CFG_DETERMINISTIC_PARCEL_ORDER, False)),
            deterministic_bmps=bool(cfg.get(CFG_DETERMINISTIC_BMP_TYPE_ORDER, False)),
        )
        self._cursor = 0
        self._replay = replay
        self._replay_cursor = 0
        self._placements: list[RealizedBMPPlacement] = []
        self.total_bmp = 0
        self.total_cost = 0.0

    def validate_replay(self, ctx: Any, scenario_id: int, seed: int) -> None:
        """Reject mismatched or malformed scenario definitions before execution."""
        scenario = self._replay
        if scenario is None:
            return
        if scenario.scenario_id != scenario_id or scenario.seed != seed:
            raise ValueError("Replay scenario ID/seed must match requested execution")
        if scenario.parcel_ids_digest != parcel_digest(ctx):
            raise ValueError("Replay parcel IDs/order do not match the current watershed")
        if scenario.load_generation_mode != ctx.load_generation_mode:
            raise ValueError("Replay load-generation formulation does not match")
        seen: set[tuple[str, int]] = set()
        for placement in scenario.placements:
            c = placement.choice
            if (c.pid, c.cps) in seen:
                raise ValueError(f"Duplicate replay BMP placement: {c.pid}, CPS {c.cps}")
            seen.add((c.pid, c.cps))
            if c.parcel_index < 0 or c.parcel_index >= len(ctx.parcel_ids) or str(ctx.parcel_ids[c.parcel_index]) != c.pid:
                raise ValueError("Replay BMP host parcel index/PID mismatch")
            if c.cps not in ctx.bmp_cps or placement.design.family != response_family(c.cps):
                raise ValueError("Replay BMP practice/response family is not configured")
            if len(placement.efficiency_by_pollutant) != len(ctx.pollutants):
                raise ValueError("Replay pollutant efficiency array length mismatch")
            for maps in placement.efficiency_by_pollutant:
                if set(dict(maps)) != set(ctx.pathway_names):
                    raise ValueError("Replay efficiency pathways do not match selected formulation")
                if any(not math.isfinite(v) or v > 1 for _, v in maps):
                    raise ValueError("Replay efficiency values are invalid")
            if not placement.design.treated or placement.design.treated[0][0] != c.parcel_index:
                raise ValueError("Replay design must treat its host parcel first")
            affected_idxs = [i for i, _ in placement.design.treated]
            if len(set(affected_idxs)) != len(affected_idxs):
                raise ValueError("Replay design contains duplicate treated parcels")
            if any(i < 0 or i >= len(ctx.parcel_ids) or not math.isfinite(fraction)
                   or not 0 <= fraction <= 1 for i, fraction in placement.design.treated):
                raise ValueError("Replay treatment parcel/fraction is invalid")
            if not math.isfinite(placement.design.cost_quantity_ha) or placement.design.cost_quantity_ha < 0:
                raise ValueError("Replay cost quantity is invalid")
            metadata = placement.design.record_fields()
            if len(metadata) != len(placement.design.metadata):
                raise ValueError("Replay BMP design has duplicate metadata keys")
            if placement.design.family == "wetland":
                if not math.isfinite(float(metadata[OUTPUT_WETLAND_AREA])) or float(metadata[OUTPUT_WETLAND_AREA]) <= 0:
                    raise ValueError("Replay wetland area must be positive and finite")
                if not math.isfinite(float(metadata[OUTPUT_CATCHMENT_RATIO])) or float(metadata[OUTPUT_CATCHMENT_RATIO]) < 0:
                    raise ValueError("Replay wetland catchment ratio must be nonnegative and finite")
                impacted = ",".join(str(ctx.parcel_ids[i]) for i in affected_idxs) if len(affected_idxs) > 1 else ""
                if metadata[OUTPUT_IMPACTED_PIDS] != impacted:
                    raise ValueError("Replay wetland affected parcels disagree with recorded PIDs")
            elif placement.design.family == "grassed":
                if len(affected_idxs) != 1 or any(not math.isfinite(float(metadata[k])) or float(metadata[k]) < 0
                   for k in (OUTPUT_LINEAR_LENGTH, OUTPUT_BUFFER_AREA, OUTPUT_PORTION_TREATED)):
                    raise ValueError("Replay grassed-waterway geometry is invalid")
                if float(metadata[OUTPUT_PORTION_TREATED]) != placement.design.treated[0][1]:
                    raise ValueError("Replay grassed treatment fraction disagrees with design")
            elif placement.design.family == "infield":
                if placement.design.treated != ((c.parcel_index, 1.0),):
                    raise ValueError("Replay in-field treatment must cover its host parcel")
            if placement.cost_usd is None or not math.isfinite(placement.cost_usd) or placement.cost_usd < 0:
                raise ValueError("Replay cost is invalid")
            if placement.recovery_gains is None or any(not 0 <= v <= 1 for v in placement.recovery_gains):
                raise ValueError("Replay AFO recovery gains are invalid")
            if not math.isfinite(placement.recovery_effect_scale) or not 0 <= placement.recovery_effect_scale <= 1:
                raise ValueError("Replay BMP recovery effect scale is invalid")

    def next_placement(self, ctx: Any) -> Optional[PlacementChoice]:
        """Draw an eligible host and CPS using the existing selection policy."""
        indices = self._pool.active_selection_indices()
        if len(indices) == 0:
            return None
        selection_index = ctx._sample_parcel_index(eligible_indices=indices)
        parcel_index = int(ctx.selection_source_idxs[selection_index])
        pid = str(ctx.parcel_ids[parcel_index])
        cps, self._cursor = self._pool.select_bmp(pid, ctx.rng, self._cursor)
        return PlacementChoice(parcel_index=parcel_index, pid=pid, cps=int(cps))

    def next_realization(self, ctx: Any) -> RealizedBMPPlacement | None:
        """Generate the next complete *pre-response* BMP design and efficiency."""
        if self._replay is not None:
            if self._replay_cursor >= len(self._replay.placements):
                return None
            placement = self._replay.placements[self._replay_cursor]
            self._replay_cursor += 1
            return placement
        choice = self.next_placement(ctx)
        if choice is None:
            return None
        maps = [ctx._sample_efficiency_map(choice.cps, pol_idx)
                for pol_idx in range(len(ctx.pollutants))]
        failed_flag = False
        scale = 1.0
        fr_cfg = ctx.cfg[CFG_BMP_FAIL_RATE]
        fail_rate = float(fr_cfg if fr_cfg is not None else 0.0)
        if fail_rate > 0.0:
            failed = int(ctx.rng.choice([0, 1], p=[1.0 - fail_rate, fail_rate]))
            if failed == 1:
                red_cfg = ctx.cfg[CFG_BMP_FAIL_REDUCTION]
                reduction = float(red_cfg if red_cfg is not None else DEFAULT_BMP_FAIL_REDUCTION)
                maps = [{k: float(v) * reduction for k, v in emap.items()} for emap in maps]
                scale = reduction
                failed_flag = True
                ctx.logger.verbose(
                    f"BMP failure triggered for cps={choice.cps}; scaling efficiencies by {reduction:.2f}"
                )
        design = sample_bmp_design(ctx, choice.parcel_index, choice.cps)
        return RealizedBMPPlacement(
            choice=choice, design=design,
            efficiency_by_pollutant=tuple(tuple((k, float(v)) for k, v in row.items()) for row in maps),
            failed=failed_flag, recovery_effect_scale=scale,
        )

    def complete_placement(self, ctx: Any, placement: RealizedBMPPlacement,
                           *, recovery_gains: tuple[float, ...] = ()) -> RealizedBMPPlacement:
        """Draw cost after AFO recovery (or reuse recorded values during replay)."""
        if placement.cost_usd is None:
            cost_area_ha = placement.design.cost_quantity_ha
            cost_length_m = None
            feedlot = placement.choice.cps in ctx.feedlot_bmp_cps
            if feedlot:
                cost_area_ha, cost_length_m = feedlot_bmp_cost_quantities(
                    ctx.afo_inputs, placement.choice.pid, placement.choice.cps
                )
            cost_this = ctx._get_bmp_cost(
                placement.choice.cps, cost_area_ha,
                **({"feedlot": True, "cost_length_m": cost_length_m} if feedlot else {}),
            )
            placement = replace(placement, cost_usd=float(cost_this),
                                recovery_gains=tuple(recovery_gains))
        else:
            if tuple(recovery_gains) != placement.recovery_gains:
                raise ValueError("Replayed AFO recovery gains did not match the scenario definition")
        self.mark_applied(placement.choice)
        self._placements.append(placement)
        self.total_bmp += 1
        self.total_cost += float(placement.cost_usd)
        return placement

    def mark_applied(self, placement: PlacementChoice) -> None:
        self._pool.record_placement(placement.pid, placement.cps)

    def complete_scenario(self, ctx: Any, scenario_id: int, seed: int) -> BMPScenario:
        if self._replay is not None and self._replay_cursor != len(self._replay.placements):
            raise ValueError("Replay ended before consuming all scenario placements")
        return BMPScenario(
            scenario_id=scenario_id, seed=seed,
            parcel_ids_digest=parcel_digest(ctx),
            load_generation_mode=ctx.load_generation_mode,
            placements=tuple(self._placements),
        )
