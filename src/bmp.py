import pandas as pd
from numpy.random import Generator
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .costs import compute_bmp_cost_usd
from .sampling import sample_from_stats

ParcelRecordFn = Callable[[Union[int, str]], pd.Series]
ParcelUpListFn = Callable[[Union[int, str]], List[str]]

FT_TO_M = 0.3048  # meters per foot


def sample_efficiency(
    rng: Generator,
    bmp_eff_df: pd.DataFrame,
    cps: Union[int, str],
    pollutant: str,
    logger: Any,
) -> float:
    """Sample BMP efficiency for a specific CPS code and pollutant."""
    sub = bmp_eff_df[(bmp_eff_df["cps"].astype(int) == int(cps)) & (bmp_eff_df["pollutant"] == pollutant)]
    row = sub.iloc[0]
    stats = {
        k: row[k]
        for k in row.index
        if k in ("mean", "sd", "min", "max") or (str(k).startswith("p") and str(k)[1:].isdigit())
    }
    return sample_from_stats(rng, stats, kind="efficiency", verbose_logger=logger)


def sample_yield(
    rng: Generator,
    pollutant_yield_df: pd.DataFrame,
    pid: Union[int, str],
    pollutant: str,
    logger: Any,
) -> float:
    """Sample baseline pollutant yield for a parcel and pollutant."""
    sub = pollutant_yield_df[(pollutant_yield_df["pid"].astype(str) == str(pid)) & (pollutant_yield_df["pollutant"] == pollutant)]
    row = sub.iloc[0]
    stats = {
        k: row[k]
        for k in row.index
        if k in ("mean", "sd", "min", "max") or (str(k).startswith("p") and str(k)[1:].isdigit())
    }
    return sample_from_stats(rng, stats, kind="yield", verbose_logger=logger)


def compute_bmp_cost(
    rng: Generator,
    bmp_cost_df: Optional[pd.DataFrame],
    cps: Union[int, str],
    quantity: float,
    logger: Any,
) -> float:
    """Estimate BMP cost in USD from configured cost statistics."""
    if bmp_cost_df is None:
        return 0.0
    sub = bmp_cost_df[bmp_cost_df["cps"].astype(int) == int(cps)]
    if sub.empty:
        return 0.0
    unit_row = sub.iloc[0]
    return compute_bmp_cost_usd(rng, cps, unit_row, quantity, logger)


def simulate_wetland(
    rng: Generator,
    pid: Union[int, str],
    eff: Dict[str, float],
    yields_map: Dict[Tuple[str, str], float],
    bmp_rec: Dict[str, Any],
    bmp_outputs: Dict[str, Dict[str, float]],
    parcel_record: ParcelRecordFn,
    parcel_up_list: ParcelUpListFn,
    pollutants: List[str],
) -> None:
    """Simulate wetland BMP behavior and reduce yields across impacted parcels."""
    row = parcel_record(pid)
    area_field_ha = float(row["area_ha"])

    wet_area_stats = {"min": 0.1, "max": 10.0, "mean": 0.4}
    wet_area = sample_from_stats(rng, wet_area_stats, kind=None, verbose_logger=None)
    wet_area = min(wet_area, area_field_ha)

    ratio_stats = {"min": 2.0, "max": 100.0, "mean": 5.0}
    cat_ratio = sample_from_stats(rng, ratio_stats, kind=None, verbose_logger=None)
    catchment_area_ha = cat_ratio * wet_area
    impacted_area_ha = wet_area + catchment_area_ha

    up_list = parcel_up_list(pid)
    impacted_pids = [str(pid)]
    total_available_ha = area_field_ha
    if impacted_area_ha > area_field_ha and len(up_list):
        for up_pid in up_list:
            r = parcel_record(up_pid)
            impacted_pids.append(str(up_pid))
            total_available_ha += float(r["area_ha"])
            if total_available_ha >= impacted_area_ha:
                break

    if impacted_area_ha > total_available_ha:
        impacted_area_ha = total_available_ha
        cat_ratio = max(0.0, (impacted_area_ha - wet_area) / max(wet_area, 1e-9))

    bmp_rec["wetland_area_ha"] = wet_area
    bmp_rec["catchment_to_wetland_ratio"] = cat_ratio
    bmp_rec["impacted_pids"] = ",".join(impacted_pids if len(impacted_pids) > 1 else [])

    remaining = impacted_area_ha
    for p in impacted_pids:
        r = parcel_record(p)
        A = float(r["area_ha"])
        if remaining <= 0:
            frac = 0.0
        elif remaining < A:
            frac = remaining / A
        else:
            frac = 1.0

        for pollutant in pollutants:
            y = yields_map[(p, pollutant)]
            reduction = y * (A * frac) * eff[pollutant]
            bmp_outputs["treated"][pollutant] += y * (A * frac)
            bmp_outputs["removed"][pollutant] += reduction
            y_new = y - reduction / A
            yields_map[(p, pollutant)] = max(0.0, y_new)

        remaining -= A


def simulate_grassed(
    rng: Generator,
    pid: Union[int, str],
    eff: Dict[str, float],
    yields_map: Dict[Tuple[str, str], float],
    bmp_rec: Dict[str, Any],
    bmp_outputs: Dict[str, Dict[str, float]],
    parcel_record: ParcelRecordFn,
    cfg: Dict[str, Any],
    pollutants: List[str],
) -> None:
    """Simulate a grassed waterway or buffer BMP and update yield reductions."""
    row = parcel_record(pid)
    perim_m = float(row["perim_m"])

    frac_stats = {"min": 0.1, "max": 0.5, "mean": 0.25}
    frac = sample_from_stats(rng, frac_stats, kind=None, verbose_logger=None)
    length_m = perim_m * frac
    depth_m = float(cfg.get("buffer_depth_ft", 35.0)) * FT_TO_M
    area_ha = (length_m * depth_m) / 10000.0
    bmp_rec["linear_length_m"] = length_m
    bmp_rec["buffer_area_ha"] = area_ha
    bmp_rec["portion_treated"] = frac

    A = float(row["area_ha"])
    for pollutant in pollutants:
        y = yields_map[(str(pid), pollutant)]
        reduction = y * (A * frac) * eff[pollutant]
        bmp_outputs["treated"][pollutant] += y * (A * frac)
        bmp_outputs["removed"][pollutant] += reduction
        y_new = y - reduction / A
        yields_map[(str(pid), pollutant)] = max(0.0, y_new)


def simulate_infield(
    pid: Union[int, str],
    eff: Dict[str, float],
    yields_map: Dict[Tuple[str, str], float],
    bmp_rec: Dict[str, Any],
    bmp_outputs: Dict[str, Dict[str, float]],
    parcel_record: ParcelRecordFn,
    pollutants: List[str],
) -> None:
    """Simulate an in-field BMP and update the parcel yield state."""
    row = parcel_record(pid)
    A = float(row["area_ha"])
    for pollutant in pollutants:
        y = yields_map[(str(pid), pollutant)]
        reduction = y * A * eff[pollutant]
        bmp_outputs["treated"][pollutant] += y * A
        bmp_outputs["removed"][pollutant] += reduction
        y_new = y - reduction / A
        yields_map[(str(pid), pollutant)] = max(0.0, y_new)
