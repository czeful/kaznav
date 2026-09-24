"""Summary tables from per-episode rows: n, mean, 95% CI, paired contrasts."""
from __future__ import annotations

from typing import Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd

from .stats import cohens_d_paired, mcnemar_exact, mean_ci, paired_bootstrap_diff


def summarize(
    df: pd.DataFrame,
    metrics: Sequence[str],
    by: Optional[Sequence[str]] = None,
    *,
    n_boot: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    """Long table: one row per (group, metric) with n, mean, CI and the CI method.

    Non-finite values (e.g. DTS = inf for an unreachable final pose) are
    excluded per metric and the exclusion count is reported in ``n_excluded``.
    """
    groups: Iterable = [((), df)] if not by else df.groupby(list(by), sort=True, dropna=False)
    rows: List[dict] = []
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        for m in metrics:
            vals = pd.to_numeric(g[m], errors="raise").astype(float).to_numpy()
            finite = np.isfinite(vals)
            row = dict(zip(by or [], key))
            row.update(metric=m, n=int(finite.sum()), n_excluded=int((~finite).sum()))
            if finite.sum() == 0:
                row.update(mean=np.nan, ci_low=np.nan, ci_high=np.nan, ci_method="")
            else:
                ci = mean_ci(vals[finite], n_boot=n_boot, seed=seed)
                row.update(mean=ci.estimate, ci_low=ci.low, ci_high=ci.high, ci_method=ci.method)
            rows.append(row)
    return pd.DataFrame(rows)


def paired_contrast(
    df: pd.DataFrame,
    *,
    arm_col: str,
    arm_a: str,
    arm_b: str,
    metric: str,
    pair_on: Sequence[str] = ("dataset", "split", "episode_id", "command_id"),
    n_boot: int = 10_000,
    seed: int = 0,
) -> dict:
    """Contrast arm_a - arm_b on the same episodes.

    Raises if the two arms do not cover exactly the same episodes: an unpaired
    comparison would silently mix episode difficulty into the model effect.
    """
    a = df[df[arm_col] == arm_a].set_index(list(pair_on))[metric]
    b = df[df[arm_col] == arm_b].set_index(list(pair_on))[metric]
    if a.index.has_duplicates or b.index.has_duplicates:
        raise ValueError("duplicate episodes within an arm; refine pair_on")
    if set(a.index) != set(b.index):
        missing = len(set(a.index) ^ set(b.index))
        raise ValueError(f"arms are not paired: {missing} episodes present in only one arm")
    b = b.loc[a.index]
    va, vb = a.astype(float).to_numpy(), b.astype(float).to_numpy()
    ci = paired_bootstrap_diff(va, vb, n_boot=n_boot, seed=seed)
    out = {
        "metric": metric, "arm_a": arm_a, "arm_b": arm_b, "n_pairs": int(va.size),
        "mean_a": float(va.mean()), "mean_b": float(vb.mean()),
        "diff": ci.estimate, "diff_ci_low": ci.low, "diff_ci_high": ci.high,
        "cohens_dz": cohens_d_paired(va, vb) if va.size >= 2 else float("nan"),
    }
    if set(np.unique(va)) <= {0.0, 1.0} and set(np.unique(vb)) <= {0.0, 1.0}:
        mc = mcnemar_exact(va.astype(bool), vb.astype(bool))
        out.update(mcnemar_only_a=mc.only_a, mcnemar_only_b=mc.only_b, mcnemar_p=mc.p_value)
    return out
