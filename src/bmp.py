"""BMP selection and load-reduction helpers.

This module contains the logic used to choose best management practices
(BMPs), sample their effectiveness, and apply their impacts to parcel-level
pollutant loads. The functions are implemented as model helpers and operate
on shared model state such as parcel geometry, pollutant yields, and
simulation outputs.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from typing import Any, Callable, Dict, List, Optional, Sequence, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Model

from .constants import (
    CFG_BUFFER_DEPTH_FT,
    BMP_CPS_NAME_MAP,
    COL_CPS,
    COL_PROBABILITY,
    OUTPUT_BUFFER_AREA,
    OUTPUT_CATCHMENT_RATIO,
    OUTPUT_IMPACTED_PIDS,
    OUTPUT_LINEAR_LENGTH,
    OUTPUT_PORTION_TREATED,
    OUTPUT_REMOVED,
    OUTPUT_TREATED,
    OUTPUT_WETLAND_AREA,
    DATA_BMP_COST,
    DATA_CPS,
    DEFAULT_BUFFER_DEPTH_FT,
)
from .logging_utils import log_scope

ParcelRecordFn = Callable[[Union[int, str]], pd.Series]
ParcelUpListFn = Callable[[Union[int, str]], List[str]]

FT_TO_M = 0.3048  # meters per foot
PATHWAY_ORDER = ("surface", "shallow subsurface", "deep subsurface")


def _get_pathway_yields(self: "Model", parcel_idx: int, pol_idx: int, total_yield: float) -> Dict[str, float]:
    """Return the parcel yield split across flow pathways.

    The model may already track pathway-specific yields in
    ``self.current_pathway_yields``. When present, those values are returned
    directly. Otherwise, the total yield is partitioned using the model's
    configured surface and shallow-subsurface fractions, with any remainder
    assigned to deep subsurface flow.

    Parameters
    ----------
    self : Model
        Active simulation model instance.
    parcel_idx : int
        Index of the parcel within the model arrays.
    pol_idx : int
        Index of the pollutant within ``self.pollutants``.
    total_yield : float
        Total parcel yield for the pollutant.

    Returns
    -------
    dict[str, float]
        Mapping from pathway name to yield contribution. Keys are ``surface``,
        ``shallow subsurface``, and ``deep subsurface``.
    """
    pathway_yields = getattr(self, "current_pathway_yields", None)
    if pathway_yields is not None:
        values = pathway_yields[parcel_idx, pol_idx, :]
        return {PATHWAY_ORDER[i]: float(values[i]) for i in range(len(PATHWAY_ORDER))}

    y_surf = total_yield * float(self.pollutant_yield_frac_surface)
    y_shal = total_yield * float(self.pollutant_yield_frac_shallow)
    y_deep = max(0.0, total_yield - (y_surf + y_shal))
    return {
        "surface": float(y_surf),
        "shallow subsurface": float(y_shal),
        "deep subsurface": float(y_deep),
    }


def _apply_pathway_reduction(
    self: "Model",
    parcel_idx: int,
    pol_idx: int,
    treatment_fraction: float,
    eff_map: Dict[str, float],
) -> float:
    """Apply pathway-specific BMP reduction to in-memory pathway yields.

    The reduction is applied only when the model is tracking pathway-specific
    loads. Each pathway is reduced by ``treatment_fraction * eff_map[path]``.
    Hydrologic partitioning and BMP treatability are intentionally independent:
    whether a BMP affects shallow or deep subsurface load is controlled solely
    by that BMP's pathway-specific effectiveness values.

    Parameters
    ----------
    self : Model
        Active simulation model instance.
    parcel_idx : int
        Index of the parcel being treated.
    pol_idx : int
        Index of the pollutant being treated.
    treatment_fraction : float
        Fraction of the parcel or drainage area treated by the BMP.
    eff_map : dict[str, float]
        Pathway-specific effectiveness values keyed by pathway name.

    Returns
    -------
    float
        Total load removed across all pathways.
    """
    pathway_yields = getattr(self, "current_pathway_yields", None)
    if pathway_yields is None:
        return 0.0

    removed = 0.0
    for path_idx, path in enumerate(PATHWAY_ORDER):
        current = float(pathway_yields[parcel_idx, pol_idx, path_idx])
        eff = float(eff_map.get(path, 0.0))
        new_value = current * (1.0 - treatment_fraction * eff)
        removed += max(0.0, current - new_value)
        pathway_yields[parcel_idx, pol_idx, path_idx] = max(0.0, new_value)
    return float(removed)


def _select_bmp_type(self: "Model") -> int:
    """Randomly choose the next BMP CPS code.

    The selection is made from the probability table produced by
    :func:`_get_bmp_selection_probs` and stored on the model as
    ``self.bmp_selection_probs``.

    Returns
    -------
    int
        Selected BMP CPS code.
    """
    idx = self.rng.choice(len(self.bmp_cps), p=self.bmp_selection_probs)
    cps = int(self.bmp_cps[idx])
    self.logger.verbose(f"selected bmp {cps} ({self._get_bmp_name(cps)})")
    return cps


def _get_bmp_name(self: "Model", cps: Union[int, str]) -> str:
    """Return a human-readable BMP name for a CPS code.

    Parameters
    ----------
    self : Model
        Active simulation model instance.
    cps : int or str
        BMP CPS identifier.

    Returns
    -------
    str
        Known BMP name when available, otherwise ``"CPS {code}"``.
    """
    key = int(cps)
    return BMP_CPS_NAME_MAP.get(key, f"CPS {key}")


def _sample_efficiency(self: "Model", cps: Union[int, str], pol_idx: int) -> float:
    """Sample a single legacy BMP effectiveness value.

    This helper supports older data structures where BMP effectiveness is
    represented as one scalar distribution per pollutant rather than as
    pathway-specific values.

    Parameters
    ----------
    self : Model
        Active simulation model instance.
    cps : int or str
        BMP CPS identifier.
    pol_idx : int
        Index of the pollutant within ``self.pollutants``.

    Returns
    -------
    float
        Sampled effectiveness value in the ``[0, 1]`` range.
    """
    stats = self.bmp_efficiency_stats[int(cps)][pol_idx]
    eff = self._sample_from_stats(stats, kind="efficiency")
    self.logger.verbose(f"selected efficiency value {eff:.2f} for pollutant={self.pollutants[pol_idx]}")
    return eff


def _sample_efficiency_map(self: "Model", cps: Union[int, str], pol_idx: int) -> Dict[str, float]:
    """Sample BMP effectiveness values for each flow pathway.

    The function supports both the newer pathway-specific efficiency format
    and the older single-distribution format. When pathway-specific
    statistics are available, each pathway is sampled independently. When only
    a single distribution is available, the sampled value is reused for all
    pathways.

    Parameters
    ----------
    self : Model
        Active simulation model instance.
    cps : int or str
        BMP CPS identifier.
    pol_idx : int
        Index of the pollutant within ``self.pollutants``.

    Returns
    -------
    dict[str, float]
        Effectiveness values for ``surface``, ``shallow subsurface``, and
        ``deep subsurface``.
    """
    entry = self.bmp_efficiency_stats[int(cps)][pol_idx]
    pathway_names = ("surface", "shallow subsurface", "deep subsurface")
    is_pathway_entry = (
        isinstance(entry, dict)
        and any(path in entry for path in pathway_names)
        and all(isinstance(value, dict) for value in entry.values())
    )
    if is_pathway_entry:
        out: Dict[str, float] = {}
        for path in pathway_names:
            stats = entry.get(path)
            if stats is None:
                # Fallback: reuse any available pathway stats; else 0.0.
                stats = next(iter(entry.values())) if entry else {"value": 0.0}
            out[path] = float(self._sample_from_stats(stats, kind="efficiency"))
        return out

    # Legacy single efficiency distribution applied uniformly to all pathways.
    stats = entry if isinstance(entry, dict) else {"value": float(entry or 0.0)}
    val = float(self._sample_from_stats(stats, kind="efficiency"))
    return {"surface": val, "shallow subsurface": val, "deep subsurface": val}


def _simulate_wetland(
    self: "Model",
    parcel_idx: int,
    eff_maps: Sequence[Dict[str, float]],
    yields: np.ndarray,
    bmp_rec: Dict[str, Any],
    bmp_outputs: Dict[str, np.ndarray],
    cps: Union[int, str] = 656,
) -> None:
    """Apply a wetland BMP and update parcel loads.

    The wetland treatment area is sampled from heuristic area statistics and
    clipped to the parcel area. The function then allocates the wetland to a
    contributing set of upstream parcels, computes the impacted drainage area,
    and applies pathway-specific reductions to each affected parcel.

    Parameters
    ----------
    self : Model
        Active simulation model instance.
    parcel_idx : int
        Index of the parcel receiving the wetland BMP.
    eff_maps : sequence of dict[str, float]
        Effectiveness maps, one per pollutant, keyed by pathway name.
    yields : numpy.ndarray
        Parcel-by-pollutant yield array updated in place.
    bmp_rec : dict[str, Any]
        Record for the current BMP placement, updated with wetland metadata.
    bmp_outputs : dict[str, numpy.ndarray]
        Accumulators for treated and removed pollutant loads.
    cps : int or str, optional
        BMP CPS code associated with the wetland. Default is ``656``.

    Returns
    -------
    None
    """
    with log_scope(label="simulate_wetland", logger=self.logger):
        self.logger.verbose("calling simulate_wetland")

        # wetland area (ha), clipped by field area
        area_field_ha = float(self.parcel_area_ha[parcel_idx])
        wet_area_stats = {"min": 0.1, "p25": 0.4, "p50": 0.81, "p75": 2.0, "max": 4.0}  # heuristic
        wet_area = self._sample_from_stats(stats=wet_area_stats, kind=None)
        wet_area = min(wet_area, area_field_ha)
        # Restored from legacy: detailed diagnostics
        self.logger.verbose(
            f"selected wetland area of {wet_area:.2f} ha in parcel idx={parcel_idx} of area={area_field_ha:.2f} ha"
        )

        # catchment area ratio (dimensionless)
        ratio_stats = {"min": 1.0, "p25": 2.0, "p50": 5.0, "p75": 10.0, "max": 100.0}  # heuristic
        cat_ratio = self._sample_from_stats(stats=ratio_stats, kind=None)
        cat_ratio = max(0.0, float(cat_ratio))

        # Impacted area to satisfy ratio
        impacted_idxs: List[int] = [parcel_idx]
        impacted_area_ha: float = wet_area * (1.0 + cat_ratio)
        total_available_ha = float(self.parcel_area_ha[parcel_idx])

        for up_idx in self.parcel_up_idxs[parcel_idx]:
            if up_idx not in impacted_idxs:
                impacted_idxs.append(up_idx)
                total_available_ha += float(self.parcel_area_ha[up_idx])
                self.logger.verbose(
                    f"added upgradient parcel (pid={self.parcel_ids[up_idx]}) with area "
                    f"{self.parcel_area_ha[up_idx]:.2f} ha to wetland-impacted parcels"
                )
                if total_available_ha >= impacted_area_ha:
                    break

        # Adjust ratio when upstream area is insufficient
        if impacted_area_ha > total_available_ha:
            self.logger.verbose(
                f"total available upgradient area ({total_available_ha:.2f} ha) < impacted area "
                f"(wetland+catchment) ({impacted_area_ha:.2f} ha)"
            )
            impacted_area_ha = total_available_ha
            cat_ratio = max(0.0, (impacted_area_ha - wet_area) / max(wet_area, 1e-9))
            self.logger.verbose(
                f"reduced impacted area to {impacted_area_ha:.2f} ha and catchment ratio to {cat_ratio:.2f}"
            )

        bmp_rec[OUTPUT_WETLAND_AREA] = float(wet_area)
        bmp_rec[OUTPUT_CATCHMENT_RATIO] = float(cat_ratio)
        bmp_rec[OUTPUT_IMPACTED_PIDS] = ",".join([self.parcel_ids[idx] for idx in impacted_idxs] if len(impacted_idxs) > 1 else [])

        # Apply reductions across impacted parcels
        remaining = impacted_area_ha
        for p_idx in impacted_idxs:
            A = float(self.parcel_area_ha[p_idx])
            if remaining <= 0:
                frac = 0.0
            elif remaining < A:
                frac = remaining / A
            else:
                frac = 1.0
            self.logger.verbose(
                f"processing wetland-impacted parcel pid={self.parcel_ids[p_idx]}, "
                f"area={A:.2f} ha, fraction draining={frac:.2f}"
            )

            for pol_idx, pollutant in enumerate(self.pollutants):
                y = float(yields[p_idx, pol_idx])
                components = self._get_pathway_yields(p_idx, pol_idx, y)
                emap = eff_maps[pol_idx]

                treated = y * (A * frac)
                removed = (A * frac) * (
                    components["surface"] * emap["surface"] +
                    components["shallow subsurface"] * emap["shallow subsurface"] +
                    components["deep subsurface"] * emap["deep subsurface"]
                )
                removed_rate = self._apply_pathway_reduction(p_idx, pol_idx, frac, emap)

                bmp_outputs[OUTPUT_TREATED][pol_idx] += treated
                bmp_outputs[OUTPUT_REMOVED][pol_idx] += removed
                y_new = removed_rate if getattr(self, "current_pathway_yields", None) is not None else (y - removed / A)
                if getattr(self, "current_pathway_yields", None) is not None:
                    y_new = float(np.sum(self.current_pathway_yields[p_idx, pol_idx, :]))
                yields[p_idx, pol_idx] = max(0.0, y_new)

            remaining -= A


def _simulate_grassed(
    self: "Model",
    parcel_idx: int,
    eff_maps: Sequence[Dict[str, float]],
    yields: np.ndarray,
    bmp_rec: Dict[str, Any],
    bmp_outputs: Dict[str, np.ndarray],
) -> None:
    """Apply a grassed buffer BMP to the selected parcel.

    The buffer length is sampled as a fraction of parcel perimeter, converted
    to area using the configured buffer depth, and then used to compute the
    treated and removed pollutant loads for the parcel.

    Parameters
    ----------
    self : Model
        Active simulation model instance.
    parcel_idx : int
        Index of the parcel receiving the BMP.
    eff_maps : sequence of dict[str, float]
        Effectiveness maps, one per pollutant, keyed by pathway name.
    yields : numpy.ndarray
        Parcel-by-pollutant yield array updated in place.
    bmp_rec : dict[str, Any]
        Record for the current BMP placement, updated with buffer metadata.
    bmp_outputs : dict[str, numpy.ndarray]
        Accumulators for treated and removed pollutant loads.

    Returns
    -------
    None
    """
    with log_scope(label="simulate_grassed", logger=self.logger):
        self.logger.verbose("calling simulate_grassed")

        # Determine linear length as a fraction of parcel perimeter
        perim_m = float(self.parcel_perim_m[parcel_idx])
        frac_stats = {"min": 0.1, "max": 0.3, "mean": 0.2}  # heuristic
        perim_frac = self._sample_from_stats(stats=frac_stats, kind=None)
        length_m = perim_m * perim_frac
        self.logger.verbose(
            f"grassed buffer length={length_m:.2f} m from fraction={perim_frac:.2f} of perimeter={perim_m:.2f} m"
        )

        # Depth and area (length * depth -> m^2 -> ha)
        depth_ft = float(self.cfg.get(CFG_BUFFER_DEPTH_FT, DEFAULT_BUFFER_DEPTH_FT))
        depth_m = depth_ft * FT_TO_M
        area_ha = (length_m * depth_m) / 10000.0
        self.logger.verbose(
            f"grassed buffer depth={depth_ft:.2f} ft ({depth_m:.2f} m), area={area_ha:.4f} ha"
        )

        # Portion treated
        frac_stats = {"min": 0.2, "max": 0.4, "mean": 0.3}  # heuristic
        frac_treated = self._sample_from_stats(stats=frac_stats, kind=None)

        # Update record and outputs
        bmp_rec[OUTPUT_LINEAR_LENGTH] = float(length_m)
        bmp_rec[OUTPUT_BUFFER_AREA] = float(area_ha)
        bmp_rec[OUTPUT_PORTION_TREATED] = float(frac_treated)

        A = float(self.parcel_area_ha[parcel_idx])
        for pol_idx, pollutant in enumerate(self.pollutants):
            y = float(yields[parcel_idx, pol_idx])
            components = self._get_pathway_yields(parcel_idx, pol_idx, y)
            emap = eff_maps[pol_idx]

            treated = y * (A * frac_treated)
            removed = (A * frac_treated) * (
                components["surface"] * emap["surface"] +
                components["shallow subsurface"] * emap["shallow subsurface"] +
                components["deep subsurface"] * emap["deep subsurface"]
            )
            self._apply_pathway_reduction(parcel_idx, pol_idx, frac_treated, emap)

            bmp_outputs[OUTPUT_TREATED][pol_idx] += treated
            bmp_outputs[OUTPUT_REMOVED][pol_idx] += removed
            y_new = float(np.sum(self.current_pathway_yields[parcel_idx, pol_idx, :])) if getattr(self, "current_pathway_yields", None) is not None else (y - removed / A)
            yields[parcel_idx, pol_idx] = max(0.0, y_new)


def _simulate_infield(
    self: "Model",
    parcel_idx: int,
    eff_maps: Sequence[Dict[str, float]],
    yields: np.ndarray,
    bmp_rec: Dict[str, Any],
    bmp_outputs: Dict[str, np.ndarray],
) -> None:
    """Apply an in-field BMP to the selected parcel.

    In-field BMPs are treated as covering the whole parcel. The function
    therefore applies each pollutant's pathway-specific effectiveness across
    the full parcel area and updates both the output accumulators and the
    yield array in place.

    Parameters
    ----------
    self : Model
        Active simulation model instance.
    parcel_idx : int
        Index of the parcel receiving the BMP.
    eff_maps : sequence of dict[str, float]
        Effectiveness maps, one per pollutant, keyed by pathway name.
    yields : numpy.ndarray
        Parcel-by-pollutant yield array updated in place.
    bmp_rec : dict[str, Any]
        Record for the current BMP placement.
    bmp_outputs : dict[str, numpy.ndarray]
        Accumulators for treated and removed pollutant loads.

    Returns
    -------
    None
    """
    with log_scope(label="simulate_infield", logger=self.logger):
        self.logger.verbose("calling _simulate_infield")

        A = float(self.parcel_area_ha[parcel_idx])
        for pol_idx, pollutant in enumerate(self.pollutants):
            y = float(yields[parcel_idx, pol_idx])
            components = self._get_pathway_yields(parcel_idx, pol_idx, y)
            emap = eff_maps[pol_idx]

            treated = y * A
            removed = A * (
                components["surface"] * emap["surface"] +
                components["shallow subsurface"] * emap["shallow subsurface"] +
                components["deep subsurface"] * emap["deep subsurface"]
            )
            self._apply_pathway_reduction(parcel_idx, pol_idx, 1.0, emap)

            bmp_outputs[OUTPUT_TREATED][pol_idx] += treated
            bmp_outputs[OUTPUT_REMOVED][pol_idx] += removed
            y_new = float(np.sum(self.current_pathway_yields[parcel_idx, pol_idx, :])) if getattr(self, "current_pathway_yields", None) is not None else (y - removed / A)
            yields[parcel_idx, pol_idx] = max(0.0, y_new)


def _get_bmp_selection_probs(self: "Model", bmp_sel_path: Optional[str]) -> pd.DataFrame:
    """Build the BMP selection probability table.

    The function first attempts to read an explicit probability CSV. If no
    file is provided, it may estimate probabilities from cost heuristics when
    enabled in the configuration. Otherwise, all BMP types receive equal
    probability.

    Parameters
    ----------
    self : Model
        Active simulation model instance.
    bmp_sel_path : str or None
        Optional path to a CSV file containing BMP selection probabilities.

    Returns
    -------
    pandas.DataFrame
        Two-column dataframe with ``COL_CPS`` and ``COL_PROBABILITY``.

    Raises
    ------
    ValueError
        If the probability file is malformed, incomplete, or contains invalid
        probability values.
    """
    if bmp_sel_path:
        df = pd.read_csv(bmp_sel_path)
        df.columns = [c.lower() for c in df.columns]
        df = df[df[COL_CPS].astype(int).isin(self.data[DATA_CPS])].copy()
        if COL_PROBABILITY not in df.columns and "pr" in df.columns:
            df[COL_PROBABILITY] = df["pr"]
        elif COL_PROBABILITY not in df.columns and "p" in df.columns:
            df[COL_PROBABILITY] = df["p"]
        required = {COL_CPS, COL_PROBABILITY}
        if not required.issubset(df.columns):
            raise ValueError(
                f"bmp selection file must contain {sorted(required)} or an accepted probability alias"
            )
        probs = pd.to_numeric(df[COL_PROBABILITY], errors="coerce")
        invalid = (~np.isfinite(probs)) | (probs < 0.0)
        if invalid.any():
            bad_rows = df.loc[invalid, [COL_CPS, COL_PROBABILITY]].head(5).to_dict(orient="records")
            raise ValueError(
                "bmp selection probabilities must be finite and nonnegative; "
                f"example bad rows: {bad_rows}"
            )
        df[COL_PROBABILITY] = probs.astype(float)
        expected_cps = {int(c) for c in self.data[DATA_CPS]}
        found_cps = set(df[COL_CPS].astype(int).tolist())
        missing = sorted(expected_cps - found_cps)
        if missing:
            raise ValueError(f"bmp selection file is missing probability rows for cps values: {missing}")
        s = df[COL_PROBABILITY].sum()
        if s <= 0:
            raise ValueError("bmp_sel probabilities sum to zero or negative")
        df[COL_PROBABILITY] = df[COL_PROBABILITY] / s
        self.logger.verbose(
            f"Loaded explicit BMP selection probabilities from {bmp_sel_path}: "
            f"{df[[COL_CPS, COL_PROBABILITY]].to_dict(orient='records')}"
        )
        return df[[COL_CPS, COL_PROBABILITY]]
    else:
        est_via_costs = self.cfg.get("bmp_sel_prob_via_costs", False)
        if est_via_costs and self.data[DATA_BMP_COST] is not None:
            self.logger.info("estimating BMP selection probabilities via cost heuristics")
            df = self._estimate_costs_for_probabilities()
            return df[[COL_CPS, COL_PROBABILITY]]
        else:
            probs = np.full(len(self.data[DATA_CPS]), 1.0 / len(self.data[DATA_CPS]))
            return pd.DataFrame({COL_CPS: self.data[DATA_CPS], COL_PROBABILITY: probs})