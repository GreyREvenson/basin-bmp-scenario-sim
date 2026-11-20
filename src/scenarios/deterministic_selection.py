"""Legacy deterministic BMP type selection order."""
from __future__ import annotations
from typing import List, Optional, Tuple

def _next_deterministic_bmp_type(
    bmp_types: List[int], already_applied: set[int], cursor: int
) -> Tuple[Optional[int], int]:
    """Select the next BMP type in sequence that remains available."""
    for offset in range(len(bmp_types)):
        type_index = (cursor + offset) % len(bmp_types)
        cps = bmp_types[type_index]
        if cps not in already_applied:
            return cps, type_index + 1
    return None, cursor
