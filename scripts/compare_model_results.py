"""Compare two model run directories numerically, preserving row/column order.

Usage:
    python scripts/compare_model_results.py BEFORE AFTER [--rtol 1e-12]

Only canonical *.parquet tables are compared (log files and plots excluded).
A missing table, missing row/column, altered null mask, mismatched identifiers,
or numerical difference above tolerance causes a nonzero exit code.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import numpy as np
import pandas as pd


def compare(before: Path, after: Path, rtol: float, atol: float) -> int:
    files_before = {p.relative_to(before) for p in before.rglob('*.parquet')}
    files_after = {p.relative_to(after) for p in after.rglob('*.parquet')}
    differences = 0
    if files_before != files_after:
        missing = sorted(files_before - files_after)
        added = sorted(files_after - files_before)
        print('Missing output tables:', [str(x) for x in missing])
        print('Extra output tables:', [str(x) for x in added])
        differences += len(missing) + len(added)
    checked = 0
    for rel in sorted(files_before & files_after):
        a = pd.read_parquet(before / rel)
        b = pd.read_parquet(after / rel)
        checked += 1
        if a.shape != b.shape or list(a.columns) != list(b.columns):
            print(f'FAIL {rel}: shape or column order differs: {a.shape} vs {b.shape}')
            differences += 1
            continue
        for col in a.columns:
            x, y = a[col], b[col]
            if not x.isna().equals(y.isna()):
                print(f'FAIL {rel}/{col}: null mask differs')
                differences += 1
                continue
            valid = ~x.isna()
            if pd.api.types.is_numeric_dtype(x) and pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(x):
                ax = pd.to_numeric(x[valid]).to_numpy(dtype=float)
                by = pd.to_numeric(y[valid]).to_numpy(dtype=float)
                equal = bool(np.allclose(ax, by, rtol=rtol, atol=atol, equal_nan=True))
                if not equal:
                    delta = float(np.max(np.abs(ax - by))) if ax.size else 0.0
                    print(f'FAIL {rel}/{col}: numeric difference (max abs={delta:.12g})')
                    differences += 1
            elif not x.reset_index(drop=True).equals(y.reset_index(drop=True)):
                # Comparable categorical, string, bool and date columns.
                xx = x.astype('string').fillna('<NA>').tolist()
                yy = y.astype('string').fillna('<NA>').tolist()
                if xx != yy:
                    print(f'FAIL {rel}/{col}: values differ')
                    differences += 1
    print(f'Compared {checked} Parquet tables; {differences} differences; '
          f'rtol={rtol:g}, atol={atol:g}')
    return int(differences > 0)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('before', type=Path)
    p.add_argument('after', type=Path)
    p.add_argument('--rtol', type=float, default=1e-12)
    p.add_argument('--atol', type=float, default=1e-12)
    args = p.parse_args()
    if not args.before.is_dir() or not args.after.is_dir():
        p.error('Both before and after must be existing model output directories')
    if args.rtol < 0 or args.atol < 0:
        p.error('Tolerances must be nonnegative')
    try:
        return compare(args.before, args.after, args.rtol, args.atol)
    except (ImportError, ModuleNotFoundError) as exc:
        print(f'Parquet engine not installed: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
