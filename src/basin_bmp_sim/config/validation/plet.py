"""Plet for the configuration."""

from __future__ import annotations
from typing import Dict, List, Optional, Sequence
import pandas as pd
from basin_bmp_sim.domain.constants import COL_PID, COL_POLLUTANT
from basin_bmp_sim.science.plet_rows import _rows_for_pid





def validate_plet_input_table(table: pd.DataFrame, parcel_ids: Sequence[str]) -> pd.DataFrame:
    """Validate and normalize required PLET classifications for every parcel.

        Parameters
        ----------
        table : pd.DataFrame
            Input table containing model data.
        parcel_ids : Sequence[str]
            Parcel identifiers in model order.

        Returns
        -------
        pd.DataFrame
            Validated and normalized PLET input table.

        Raises
        ------
        ValueError
            If PLET classifications or parameter rows violate the PLET input contract.

    """
    from basin_bmp_sim.domain.constants import PLET_CLASSIFICATION_PARAMETERS, PLET_DERIVED_PARAMETERS as _PLET_DERIVED_PARAMETERS
    from basin_bmp_sim.science.plet_classification import canonical_parameter_name, normalize_plet_hsg, normalize_plet_land_cover
    normalized = table.copy()
    raw_parameter_labels = normalized["parameter"].map(
        lambda value: str(value).strip().lower().replace("-", "_").replace(" ", "_")
    )
    removed_parameters = sorted(set(raw_parameter_labels[raw_parameter_labels.str.startswith("irrigat")]))
    if removed_parameters:
        raise ValueError(f"PLET irrigation parameters are no longer supported: {removed_parameters}")
    normalized["parameter"] = normalized["parameter"].map(canonical_parameter_name)
    supplied_derived = sorted(
        set(normalized.loc[normalized["parameter"].isin(_PLET_DERIVED_PARAMETERS), "parameter"])
    )
    if supplied_derived:
        raise ValueError(
            "PLET parcel parameter tables may not specify "
            f"{supplied_derived}; use input_curve_number and "
            "input_infiltration_fraction in parcels.gpkg instead"
        )
    classification_mask = normalized["parameter"].isin(PLET_CLASSIFICATION_PARAMETERS)
    if classification_mask.any() and "value" not in normalized.columns:
        raise ValueError("PLET land_cover and hsg classifications require a fixed value column")
    for row_index in normalized.index[classification_mask]:
        parameter = str(normalized.at[row_index, "parameter"])
        value = normalized.at[row_index, "value"]
        if pd.isna(value) or str(value).strip() == "":
            raise ValueError(f"PLET classification row {row_index} for {parameter} requires a fixed value")
        if parameter == "land_cover":
            normalized.at[row_index, "value"] = normalize_plet_land_cover(value)
        else:
            normalized.at[row_index, "value"] = normalize_plet_hsg(value)
    for pid in map(str, parcel_ids):
        effective = {canonical_parameter_name(row["parameter"]): row for row in _rows_for_pid(normalized, pid)}
        missing = [parameter for parameter in PLET_CLASSIFICATION_PARAMETERS if parameter not in effective]
        if missing:
            raise ValueError(f"PLET inputs for pid={pid} are missing required classifications: {missing}")
    return normalized


def validate_plet_runtime_inputs(
    plet_inputs: pd.DataFrame,
    rusle_inputs: Optional[pd.DataFrame],
    pollutant_concentrations: Optional[pd.DataFrame],
    groundwater_concentrations: Optional[pd.DataFrame],
    parcel_ids: Sequence[str],
    pollutants: Sequence[str],
) -> None:
    """Validate all PLET/RUSLE coverage before the simulation worker starts.

        Parameters
        ----------
        plet_inputs : pd.DataFrame
            PLET parameter input table.
        rusle_inputs : Optional[pd.DataFrame]
            RUSLE parameter input table, if configured.
        pollutant_concentrations : Optional[pd.DataFrame]
            Runtime surface-pathway concentration rows split from the unified
            pollutant concentration input table.
        groundwater_concentrations : Optional[pd.DataFrame]
            Runtime subsurface-pathway concentration rows split from the unified
            pollutant concentration input table.
        parcel_ids : Sequence[str]
            Parcel identifiers in model order.
        pollutants : Sequence[str]
            Pollutant names in model order.

        Raises
        ------
        ValueError
            If any parcel lacks required PLET/RUSLE parameters or pollutant concentrations for runtime load generation.

    """
    from basin_bmp_sim.domain.constants import PLET_REQUIRED_INPUTS as _REQUIRED_PLET_INPUTS, RUSLE_REQUIRED_INPUTS as _REQUIRED_RUSLE
    from basin_bmp_sim.science.plet_classification import canonical_parameter_name

    if rusle_inputs is not None and not rusle_inputs.empty and "parameter" in rusle_inputs.columns:
        raw_parameter_labels = rusle_inputs["parameter"].map(
            lambda value: str(value).strip().lower().replace("-", "_").replace(" ", "_")
        )
        removed_parameters = sorted(
            set(raw_parameter_labels[raw_parameter_labels.str.startswith("watershed_area")])
        )
        if removed_parameters:
            raise ValueError(
                f"RUSLE watershed-area parameters are no longer supported: {removed_parameters}; "
                "supply 'sdr' instead"
            )

    def effective_parameters(table: Optional[pd.DataFrame], pid: str) -> Dict[str, pd.Series]:
        """Return the effective parameter rows for a parcel.

                Parameters
                ----------
                table : Optional[pd.DataFrame]
                    Input table containing model data.
                pid : str
                    Parcel identifier.

                Returns
                -------
                Dict[str, pd.Series]
                    Effective parameter rows keyed by canonical parameter name.

        """
        if table is None:
            return {}
        return {canonical_parameter_name(row["parameter"]): row for row in _rows_for_pid(table, pid)}

    def has_concentration(table: Optional[pd.DataFrame], pid: str, pollutant: str) -> bool:
        """Return whether a parcel has a pollutant concentration input.

                Parameters
                ----------
                table : Optional[pd.DataFrame]
                    Input table containing model data.
                pid : str
                    Parcel identifier.
                pollutant : str
                    Pollutant name.

                Returns
                -------
                bool
                    ``True`` when a matching concentration input exists; otherwise ``False``.

        """
        if table is None or table.empty:
            return False
        pols = table[COL_POLLUTANT].astype(str)
        try:
            target_pid = int(pid)
            pids = pd.to_numeric(table[COL_PID], errors="coerce")
            pid_match = table[COL_PID].isna() | (pids == target_pid)
        except (TypeError, ValueError):
            pid_match = table[COL_PID].isna() | (table[COL_PID].astype(str) == str(pid))
        return bool((pid_match & (pols == pollutant)).any())

    for pid in map(str, parcel_ids):
        plet_effective = effective_parameters(plet_inputs, pid)
        missing_plet = [name for name in _REQUIRED_PLET_INPUTS if name not in plet_effective]
        if missing_plet:
            raise ValueError(f"PLET inputs for pid={pid} are missing required parameters: {missing_plet}")
        has_avg_rain = "avg_rain_in" in plet_effective
        has_legacy_precip = all(
            name in plet_effective
            for name in ("annual_precip_in", "rain_correction_fraction")
        )
        if not (has_avg_rain or has_legacy_precip):
            raise ValueError(
                f"PLET inputs for pid={pid} must provide input_avg_rain_in "
                "(PLET AVG_RAIN) or both input_annual_precip_in and "
                "input_rain_correction_fraction"
            )
        rusle_effective = effective_parameters(rusle_inputs, pid)
        if rusle_effective:
            missing_rusle = [name for name in _REQUIRED_RUSLE if name not in rusle_effective]
            if missing_rusle:
                raise ValueError(f"RUSLE inputs for pid={pid} are incomplete; missing: {missing_rusle}")
        for pollutant in pollutants:
            pol = str(pollutant).upper()
            if pol in {"TN", "TP"} and not has_concentration(pollutant_concentrations, pid, pol):
                raise ValueError(f"Surface concentration for pid={pid}, pollutant={pol} is required")
            if pol != "TSS" and not has_concentration(groundwater_concentrations, pid, pol):
                raise ValueError(
                    f"Subsurface concentration for pid={pid}, pollutant={pol} is required in plet_rusle mode"
                )
            if pol == "TSS" and not rusle_effective and not has_concentration(pollutant_concentrations, pid, pol):
                raise ValueError(f"TSS for pid={pid} requires complete RUSLE inputs or a TSS concentration")
