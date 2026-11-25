"""Scientifically defined pollutant delivery to outlets.

The driver consumes the resulting outlet reductions; it does not recompute
parcel delivery factors or decide which outlets receive BMP benefits.
"""
from __future__ import annotations
from basin_bmp_sim.domain.parcel import _get_delivery_coeffs, _get_parcel_out_oids

from typing import Any, Dict, List, Tuple
import numpy as np
from basin_bmp_sim.domain.constants import COL_SDR_F_TO_S, COL_SDR_S_TO_O, COL_NDR_F_TO_S, COL_NDR_S_TO_O

def _accumulate_delivered_bmp_reductions(
    ctx: Any,
    changes: List[Tuple[int, np.ndarray]],
    cumul: Dict[str, Dict[str, float]],
) -> None:
    """Credit each changed parcel's signed mass only to its linked outlets.

    The parcel-level delivery convention applies the two sediment factors for
    TSS and the two nutrient factors for other pollutants. A different wetland
    transport model would require an explicitly different accounting contract.
    """
    for idx, removed_mass_kg in changes:
        pid = str(ctx.parcel_ids[idx])
        for oid in _get_parcel_out_oids(ctx, idx):
            dr = _get_delivery_coeffs(ctx, pid, oid)
            for pol_idx, pol in enumerate(ctx.pollutants):
                if pol == 'TSS':
                    factor = dr[COL_SDR_F_TO_S] * dr[COL_SDR_S_TO_O]
                else:
                    factor = dr[COL_NDR_F_TO_S] * dr[COL_NDR_S_TO_O]
                cumul[pol][oid] += float(removed_mass_kg[pol_idx]) * factor
