"""Select eligible BMP placements independently of load calculations.

This transitional generator isolates the stochastic *selection* policy.  A
complete, replayable BMPScenario will also need design parameters, efficiency
realizations, costs, and cross-parcel treatments; those are still sampled in
the legacy execution loop and must not be claimed as independently generated.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional
from src.bmp_selection import BMPEligibilityPool
from src.constants import CFG_DETERMINISTIC_PARCEL_ORDER, CFG_DETERMINISTIC_BMP_TYPE_ORDER


@dataclass(frozen=True)
class PlacementChoice:
    parcel_index: int
    pid: str
    cps: int


class StochasticScenarioGenerator:
    """Maintain the original eligibility and placement selection state."""

    def __init__(self, ctx: Any, cfg: dict[str, Any]) -> None:
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

    def next_placement(self, ctx: Any) -> Optional[PlacementChoice]:
        """Draw the next eligible parcel and practice using the original RNG."""
        indices = self._pool.active_selection_indices()
        if len(indices) == 0:
            return None
        selection_index = ctx._sample_parcel_index(eligible_indices=indices)
        parcel_index = int(ctx.selection_source_idxs[selection_index])
        pid = str(ctx.parcel_ids[parcel_index])
        cps, self._cursor = self._pool.select_bmp(pid, ctx.rng, self._cursor)
        return PlacementChoice(parcel_index=parcel_index, pid=pid, cps=int(cps))

    def mark_applied(self, placement: PlacementChoice) -> None:
        """Enforce the existing once-per-(parcel, CPS) eligibility restriction."""
        self._pool.record_placement(placement.pid, placement.cps)
