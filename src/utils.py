import re
from typing import Any, Dict, Iterable, Mapping, Optional

def ci_get(d: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """Case-insensitive dictionary lookup for config keys."""
    key_l = key.lower()
    for k, v in d.items():
        if str(k).lower() == key_l:
            return v
    return default

def normalize_columns(df: Any) -> Any:
    """Normalize DataFrame column labels to lowercase strings."""
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df

def parse_percent_keys(cols: Iterable[Any]) -> Dict[int, Any]:
    """Return a mapping from percentile column names to integer keys."""
    # Return a mapping for 'p5', 'p10', etc.
    percents: Dict[int, Any] = {}
    for c in cols:
        c_l = str(c).lower().strip()
        m = re.fullmatch(r"p(\d{1,2}|100)", c_l)
        if m:
            p = int(m.group(1))
            percents[p] = c
    return percents
