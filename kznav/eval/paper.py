"""Paper tables from run directories: main results, paired contrasts (Holm-adjusted), breakdowns, funnel,
failure stages and the decomposition of a success gap, verifier calibration, cost. Each function returns a
DataFrame; ``to_markdown`` / ``to_latex`` render them.

An arm is one or more run directories (several seeds of one configuration). Statistics follow
docs/analysis_plan.md (D-047): seeds are averaged per episode, the scene is the unit of inference
(scene-cluster bootstrap CIs, scene sign-flip permutation p-values, Holm over the table's success contrasts);
McNemar on pooled episode-seed pairs is a secondary column. Arms are paired on (dataset, split, episode_id,
seed): a contrast refuses arms that did not run the same episodes, except episodes listed as failed in a run's
failures.csv (D-046), which are dropped from every arm of the contrast and counted.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..run.logging import completed_events
from .calibration import brier_score, ece_binary
from .failures import STAGE_GROUP, STAGE_ORDER, Stage, attribute_row
from .stats import (cluster_bootstrap_mean, cluster_sign_flip_test, cohens_d_paired, holm_adjust, mcnemar_exact,
                    mean_ci)

PAIR = ("dataset", "split", "episode_id", "seed")
EPISODE = ("dataset", "split", "episode_id")
#: Below this many scenes a cluster bootstrap has nothing to resample: CIs fall back to episode-level ones.
MIN_SCENES_FOR_CLUSTER_CI = 5


@dataclass
class Arm:
    key: str
    label: str
    runs: List[Path]
    group: str = ""
    episodes: pd.DataFrame = field(default_factory=pd.DataFrame)
    exclusions: pd.DataFrame = field(default_factory=pd.DataFrame)
    failures: pd.DataFrame = field(default_factory=pd.DataFrame)       # episodes that never completed (D-046)
    verifications: List[dict] = field(default_factory=list)
    calls: List[dict] = field(default_factory=list)

    @classmethod
    def load(cls, key: str, label: str, runs: Sequence[str], group: str = "",
             subset: Optional[Sequence[str]] = None) -> "Arm":
        """``subset``: episode ids to keep (e.g. the nested 300-episode subset shared with secondary arms)."""
        arm = cls(key, label, [Path(r) for r in runs], group)
        eps, exc, fail = [], [], []
        for r in arm.runs:
            d = pd.read_csv(r / "episodes.csv")
            eps.append(d)
            if (r / "exclusions.csv").exists():
                exc.append(pd.read_csv(r / "exclusions.csv"))
            if (r / "failures.csv").exists():
                f = pd.read_csv(r / "failures.csv")
                f = f[~f.episode_id.isin(d.episode_id)].drop_duplicates("episode_id", keep="last")
                if len(f):
                    seed = int(d.seed.iloc[0]) if len(d) else 0
                    fail.append(f.assign(seed=seed))
            if (r / "events.jsonl").exists():     # events of failed or interrupted attempts are not results (D-046)
                arm.verifications += list(completed_events(r / "events.jsonl", "verification"))
            if (r / "vlm_calls.jsonl").exists():
                arm.calls += [json.loads(line) for line in open(r / "vlm_calls.jsonl", encoding="utf-8") if line.strip()]
        arm.episodes = pd.concat(eps, ignore_index=True).assign(arm=key)
        arm.episodes["failure_stage"] = stages_of(arm.episodes)
        arm.exclusions = pd.concat(exc, ignore_index=True) if exc else pd.DataFrame(columns=["episode_id", "reason"])
        arm.failures = pd.concat(fail, ignore_index=True) if fail else pd.DataFrame(columns=["episode_id", "seed"])
        if subset is not None:
            keep = set(subset)
            arm.episodes = arm.episodes[arm.episodes.episode_id.isin(keep)].reset_index(drop=True)
            arm.exclusions = arm.exclusions[arm.exclusions.episode_id.isin(keep)].reset_index(drop=True)
            arm.failures = arm.failures[arm.failures.episode_id.isin(keep)].reset_index(drop=True)
            arm.verifications = [e for e in arm.verifications if str(e.get("episode_id")) in keep]
            arm.calls = [c for c in arm.calls if str((c.get("context") or {}).get("episode_id")) in keep]
        return arm


def stages_of(df: pd.DataFrame, order: str = "pipeline", stop_label: str = "content") -> pd.Series:
    """The failure stage of every row: the logged one (schema v4, pipeline order, content stop labels) or, for older
    rows and for the sensitivity analyses (navigation-first order; ``stop_label="position"``: the position-based
    stop label instead of the content label, D-046), re-attributed from the row's columns."""
    if stop_label not in ("content", "position"):
        raise ValueError("stop_label must be content | position")
    if order != "pipeline" or stop_label == "position":
        def row_for(r):
            if stop_label == "position" and "stop_is_goal_by_position" in r and r.get("stop_label_source"):
                r = r.copy()
                r["stop_candidate_is_goal"] = r["stop_is_goal_by_position"]
            return r
        return pd.Series([attribute_row(row_for(r), order).value for _, r in df.iterrows()], index=df.index,
                         dtype=object)
    logged = df["failure_stage"] if "failure_stage" in df else pd.Series("", index=df.index)
    logged = logged.fillna("").astype(str)
    return pd.Series([v if v else attribute_row(r).value for v, (_, r) in zip(logged, df.iterrows())],
                     index=df.index, dtype=object)


def episode_means(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Seeds averaged per episode: columns dataset, split, episode_id, scene_id, command_id, value, n_seeds."""
    v = pd.to_numeric(df[metric].astype(float), errors="raise")
    d = df.assign(_v=v)
    if "command_id" not in d:
        d = d.assign(command_id="")
    g = d.groupby(list(EPISODE), sort=True)
    return g.agg(scene_id=("scene_id", "first"), command_id=("command_id", "first"), value=("_v", "mean"),
                 n_seeds=("_v", "size")).reset_index()


def metric_ci(df: pd.DataFrame, metric: str):
    """(ConfidenceInterval, method) of an arm's mean metric: scene-cluster bootstrap on seed-averaged episodes with
    at least MIN_SCENES_FOR_CLUSTER_CI scenes; otherwise Wilson (binary) or an episode bootstrap. None if empty."""
    clean = df.assign(**{metric: df[metric].astype(float)}).replace([np.inf, -np.inf], np.nan).dropna(subset=[metric])
    if not len(clean):
        return None, ""
    e = episode_means(clean, metric)
    if e.scene_id.nunique() >= MIN_SCENES_FOR_CLUSTER_CI:
        return cluster_bootstrap_mean(e.value, e.scene_id, n_boot=5000), "scene bootstrap"
    c = mean_ci(clean[metric].to_numpy(), n_boot=5000)
    return c, ("Wilson" if c.method == "wilson" else "episode bootstrap")


def sr_interval(arm: Arm):
    """SR of an arm with its 95% CI (as in the main table)."""
    d = arm.episodes
    return metric_ci(d.assign(success=d.success.astype(bool).astype(float)), "success")


def _scene_ci(df: pd.DataFrame, metric: str, scale: float = 1.0, digits: int = 3) -> Tuple[str, str]:
    c, method = metric_ci(df, metric)
    if c is None:
        return "—", ""
    return f"{scale * c.estimate:.{digits}f} [{scale * c.low:.{digits}f}, {scale * c.high:.{digits}f}]", method


def main_table(arms: Sequence[Arm]) -> pd.DataFrame:
    """One row per arm. ``attrs["note"]`` names the CI method(s) used (scene bootstrap needs >= 5 scenes)."""
    rows, methods = [], set()
    for a in arms:
        d = a.episodes
        n = len(d)
        lat = d.vlm_latency_total_s.astype(float)
        mt = d["translation_latency_s"].astype(float) if "translation_latency_s" in d else pd.Series(0.0, index=d.index)
        sr, m1 = _scene_ci(d.assign(success=d.success.astype(bool).astype(float)), "success", 100.0, 1)
        spl, m2 = _scene_ci(d, "spl")
        methods |= {m1, m2} - {""}
        rows.append({
            "Arm": a.label, "n": n, "scenes": d.scene_id.nunique(), "excluded": len(a.exclusions),
            "failed": len(a.failures), "seeds": d.seed.nunique(),
            "SR, % [95% CI]": sr, "SPL [95% CI]": spl, "SoftSPL": f"{d.soft_spl.mean():.3f}",
            "DTS, m": f"{d.dts.replace(np.inf, np.nan).mean():.2f}",
            "false stop, %": f"{100 * (d.outcome == 'false_stop').mean():.1f}",
            "stuck, %": f"{100 * (d.outcome == 'stuck').mean():.1f}",
            "collisions / ep": f"{d.collisions.mean():.2f}",
            "VLM calls / ep": f"{d.vlm_calls.mean():.1f}" if d.vlm_calls.sum() else "—",
            "VLM s / ep": f"{lat.mean():.1f}" if d.vlm_calls.sum() else "—",
            "MT s / ep": f"{mt.mean():.1f}" if mt.sum() else "—",
        })
    out = pd.DataFrame(rows)
    out.attrs["ci_methods"] = ", ".join(sorted(methods))
    out.attrs["note"] = f"95% CI: {out.attrs['ci_methods']}; seeds averaged per episode"
    return out


def _paired_frames(arms: Mapping[str, Arm], keys: Sequence[str], pair_on: Sequence[str]) -> Tuple[List[pd.DataFrame], int]:
    """The arms' rows on the same (episode, seed) keys. Episodes failed (D-046) in any of the arms are dropped from
    all; any other mismatch raises: an unpaired comparison would mix episode difficulty into the effect."""
    failed = set()
    for k in keys:
        failed |= set(arms[k].failures.get("episode_id", []))
    frames, dropped = [], 0
    for k in keys:
        d = arms[k].episodes
        dropped += int(d.episode_id.isin(failed).sum())
        frames.append(d[~d.episode_id.isin(failed)])
    idx = [f.set_index(list(pair_on)).index for f in frames]
    for k, ix in zip(keys, idx):
        if ix.has_duplicates:
            raise ValueError(f"duplicate episodes within arm {k!r}; refine pair_on")
    for k, ix in zip(keys[1:], idx[1:]):
        if set(ix) != set(idx[0]):
            raise ValueError(f"arms {keys[0]!r} and {k!r} are not paired: {len(set(idx[0]) ^ set(ix))} episode-seeds "
                             f"in one arm only")
    return frames, dropped


def _as_metric(d: pd.DataFrame, m: str) -> pd.DataFrame:
    return d.assign(success=d.success.astype(bool).astype(float)) if m == "success" else d


def contrast_table(arms: Sequence[Arm], pairs: Sequence[Sequence[str]], metrics: Sequence[str] = ("success", "spl"),
                   pair_on: Sequence[str] = PAIR, cluster: str = "scene_id",
                   drop_model_errors: bool = False) -> pd.DataFrame:
    """Paired contrasts on the same episodes (D-047). A pair (a, b) is a − b; a quadruple (a, b, c, d) is the
    interaction (a − b) − (c − d), e.g. whether the native model's advantage is larger in Kazakh than in English.
    Seeds are averaged per episode; CI by scene-cluster bootstrap (episode bootstrap below
    MIN_SCENES_FOR_CLUSTER_CI scenes); p by scene sign-flip permutation, Holm-adjusted over all success contrasts of
    the table (the family of primary contrasts); McNemar on pooled episode-seed pairs as a secondary test of a − b.
    Sensitivity analyses (docs/analysis_plan.md): ``cluster="command_id"`` makes the command the unit (each command
    serves many episodes); ``drop_model_errors`` drops episodes with a failed model call in any arm of the contrast."""
    if cluster not in ("scene_id", "command_id"):
        raise ValueError("cluster must be scene_id | command_id")
    by_key = {a.key: a for a in arms}
    rows = []
    for spec in pairs:
        spec = tuple(spec)
        if len(spec) not in (2, 4):
            raise ValueError(f"a contrast names 2 arms (a − b) or 4 arms ((a − b) − (c − d)), got {spec}")
        frames, dropped = _paired_frames(by_key, spec, pair_on)
        if drop_model_errors:
            bad = set()
            for f in frames:
                if "vlm_call_errors" in f:
                    bad |= set(f.loc[pd.to_numeric(f.vlm_call_errors, errors="coerce").fillna(0) > 0, "episode_id"])
            dropped += sum(int(f.episode_id.isin(bad).sum()) for f in frames)
            frames = [f[~f.episode_id.isin(bad)] for f in frames]
        coef = (1.0, -1.0) if len(spec) == 2 else (1.0, -1.0, -1.0, 1.0)
        la = by_key[spec[0]].label if len(spec) == 2 else f"({by_key[spec[0]].label} − {by_key[spec[1]].label})"
        lb = by_key[spec[1]].label if len(spec) == 2 else f"({by_key[spec[2]].label} − {by_key[spec[3]].label})"
        for m in metrics:
            em = [episode_means(_as_metric(f, m), m) for f in frames]
            j = em[0][list(EPISODE) + ["scene_id", "command_id", "n_seeds"]].copy()
            for i, e in enumerate(em):
                j = j.merge(e[list(EPISODE) + ["value"]].rename(columns={"value": f"v{i}"}), on=list(EPISODE))
            vals = [j[f"v{i}"].to_numpy() for i in range(len(spec))]
            d = sum(c * v for c, v in zip(coef, vals))
            side_a = vals[0] if len(spec) == 2 else vals[0] - vals[1]
            side_b = vals[1] if len(spec) == 2 else vals[2] - vals[3]
            units = j[cluster].astype(str).to_numpy()
            k = len(set(units))
            ci = cluster_bootstrap_mean(d, units, n_boot=10_000) if k >= MIN_SCENES_FOR_CLUSTER_CI \
                else mean_ci(d, n_boot=10_000, binary=False)
            perm = cluster_sign_flip_test(d, units)
            row = {"a": la, "b": lb, "metric": m, "episodes": len(j), "scenes": j.scene_id.nunique(),
                   "clusters": f"{k} {cluster.replace('_id', 's')}", "seeds": int(j.n_seeds.max()),
                   "mean_a": float(side_a.mean()), "mean_b": float(side_b.mean()), "diff": float(d.mean()),
                   "ci_low": ci.low, "ci_high": ci.high,
                   "d_z": cohens_d_paired(side_a, side_b) if len(j) >= 2 else float("nan"),
                   "p": perm.p_value, "only_a": None, "only_b": None, "p_mcnemar": np.nan, "dropped_failed": dropped}
            if m == "success" and len(spec) == 2:
                pa = _as_metric(frames[0], m).set_index(list(pair_on))["success"]
                pb = _as_metric(frames[1], m).set_index(list(pair_on))["success"].loc[pa.index]
                mc = mcnemar_exact(pa.to_numpy().astype(bool), pb.to_numpy().astype(bool))
                row.update(only_a=mc.only_a, only_b=mc.only_b, p_mcnemar=mc.p_value)
            rows.append(row)
    out = pd.DataFrame(rows)
    out[["only_a", "only_b"]] = out[["only_a", "only_b"]].astype("Int64")
    mask = out.metric == "success"
    out["p_holm"] = np.nan
    if mask.any():
        out.loc[mask, "p_holm"] = holm_adjust(out.loc[mask, "p"].astype(float).to_numpy())
    cols = ["a", "b", "metric", "episodes", "scenes", "seeds", "mean_a", "mean_b", "diff", "ci_low", "ci_high", "d_z",
            "p", "p_holm", "only_a", "only_b", "p_mcnemar", "dropped_failed"]
    if cluster != "scene_id":
        cols.insert(5, "clusters")
    return out[cols]


def stage_table(arms: Sequence[Arm]) -> pd.DataFrame:
    """Share of all episodes ending at each failure stage (D-045), pipeline order, and the stage groups.
    The stage shares and SR add up to 1 for every arm."""
    rows = []
    for a in arms:
        st = a.episodes["failure_stage"]
        row = {"Arm": a.label, "n": len(st), "SR": float((st == Stage.NONE.value).mean())}
        for s in STAGE_ORDER:
            row[s.value] = float((st == s.value).mean())
        for g in ("language", "grounding", "navigation"):
            row[f"[{g}]"] = float(st.map(lambda v: STAGE_GROUP[Stage(v)] == g).mean())
        rows.append(row)
    out = pd.DataFrame(rows)
    empty = [s.value for s in STAGE_ORDER if s.value in out and not out[s.value].any()]
    return out.drop(columns=empty)


def gap_decomposition(arms: Sequence[Arm], a: str, b: str, pair_on: Sequence[str] = PAIR,
                      n_boot: int = 10_000, order: str = "pipeline", stop_label: str = "content") -> pd.DataFrame:
    """SR(a) − SR(b) = Σ_stage [rate_b(stage) − rate_a(stage)]: how much of the success gap each failure stage
    explains (the H1 test, D-045). Per stage and stage group: both rates, the contribution with a scene-cluster
    bootstrap CI (episode bootstrap below MIN_SCENES_FOR_CLUSTER_CI scenes), and its share of the gap.
    ``order="navigation_first"`` / ``stop_label="position"``: the sensitivity analyses of the analysis plan."""
    by_key = {x.key: x for x in arms}
    (da, db), _ = _paired_frames(by_key, (a, b), pair_on)
    if order != "pipeline" or stop_label != "content":
        da = da.assign(failure_stage=stages_of(da, order, stop_label))
        db = db.assign(failure_stage=stages_of(db, order, stop_label))
    keys: List[Tuple[str, Sequence[str]]] = [(s.value, [s.value]) for s in STAGE_ORDER]
    for g in ("language", "grounding", "navigation"):
        keys.append((f"[{g}]", [s.value for s in STAGE_ORDER if STAGE_GROUP[s] == g]))
    keys.append(("[language + grounding]", [s.value for s in STAGE_ORDER if STAGE_GROUP[s] in ("language", "grounding")]))
    sr = episode_means(da.assign(_y=da.success.astype(bool).astype(float)), "_y").merge(
        episode_means(db.assign(_y=db.success.astype(bool).astype(float)), "_y"), on=list(EPISODE), suffixes=("_a", "_b"))
    gap = float((sr.value_a - sr.value_b).mean())
    rows = [{"stage": "SR gap (a − b)", "rate_a": float(sr.value_a.mean()), "rate_b": float(sr.value_b.mean()),
             "contribution": gap, "ci_low": np.nan, "ci_high": np.nan, "share_of_gap": 1.0 if gap else np.nan}]
    for name, members in keys:
        ea = episode_means(da.assign(_y=da.failure_stage.isin(members).astype(float)), "_y")
        eb = episode_means(db.assign(_y=db.failure_stage.isin(members).astype(float)), "_y")
        j = ea.merge(eb, on=list(EPISODE), suffixes=("_a", "_b"))
        if not (j.value_a.any() or j.value_b.any()):
            continue
        contrib = (j.value_b - j.value_a).to_numpy()
        scenes = j.scene_id_a.to_numpy()
        ci = cluster_bootstrap_mean(contrib, scenes, n_boot=n_boot) if len(set(scenes)) >= MIN_SCENES_FOR_CLUSTER_CI \
            else mean_ci(contrib, n_boot=n_boot, binary=False)
        rows.append({"stage": name, "rate_a": float(j.value_a.mean()), "rate_b": float(j.value_b.mean()),
                     "contribution": float(contrib.mean()), "ci_low": ci.low, "ci_high": ci.high,
                     "share_of_gap": float(contrib.mean()) / gap if gap else np.nan})
    return pd.DataFrame(rows)


def breakdown(arms: Sequence[Arm], by: str) -> pd.DataFrame:
    """Successes "k/n (SR %)" per arm and value of ``by`` (word_type, goal_category, command_lang, ...)."""
    rows = []
    for a in arms:
        row = {"Arm": a.label}
        for v, g in a.episodes.groupby(by):
            k, n = int(g.success.astype(bool).sum()), len(g)
            row[str(v)] = f"{k}/{n} ({100 * k / n:.0f}%)"
        rows.append(row)
    return pd.DataFrame(rows).fillna("—")


FUNNEL = (("goal in view", "goal_ever_in_view"), ("detected", "goal_ever_detected"),
          ("verified", "goal_ever_verified_positive"), ("success", "success"))


def funnel(arms: Sequence[Arm]) -> pd.DataFrame:
    rows = []
    for a in arms:
        d = a.episodes
        row = {"Arm": a.label}
        for name, col in FUNNEL:
            row[name] = float(d[col].astype(bool).mean()) if col in d else np.nan
        if not d.vlm_calls_verify.sum():
            row["verified"] = np.nan                          # no verifier in this arm
        rows.append(row)
    return pd.DataFrame(rows)


def verifier_samples(arm: Arm) -> Tuple[np.ndarray, np.ndarray, str]:
    """Raw P(yes) and labels of the arm's verifier calls: content labels (D-042) when logged, else positions."""
    calls = [e for e in arm.verifications if e.get("called") and e.get("p_raw") is not None]
    content = [e for e in calls if e.get("is_goal_by_content") is not None]
    use = content if content else calls
    key = "is_goal_by_content" if content else "is_goal_privileged"
    return (np.array([e["p_raw"] for e in use], dtype=float), np.array([bool(e[key]) for e in use]),
            "content" if content else "position")


def verifier_table(arms: Sequence[Arm]) -> pd.DataFrame:
    rows = []
    for a in arms:
        p, y, label = verifier_samples(a)
        if not p.size:
            continue
        yes = p >= 0.5
        rows.append({"Arm": a.label, "calls": int(p.size), "goal calls": int(y.sum()), "label": label,
                     "sensitivity": f"{int((yes & y).sum())}/{int(y.sum())}",
                     "specificity": f"{int((~yes & ~y).sum())}/{int((~y).sum())}",
                     "ECE (raw)": f"{ece_binary(p, y):.3f}", "Brier (raw)": f"{brier_score(p, y):.3f}"})
    return pd.DataFrame(rows)


def latency_table(arms: Sequence[Arm]) -> pd.DataFrame:
    """Per model call (warm-up excluded), by method: n, median and p95 of latency_total_s."""
    rows = []
    for a in arms:
        by: Dict[str, List[float]] = {}
        for c in a.calls:
            if (c.get("context") or {}).get("warmup"):
                continue
            by.setdefault(c["method"], []).append(float(c["latency_total_s"]))
        for m, v in sorted(by.items()):
            rows.append({"Arm": a.label, "call": m, "n": len(v), "median, s": f"{np.median(v):.2f}",
                         "p95, s": f"{np.percentile(v, 95):.2f}"})
    return pd.DataFrame(rows)


def exclusion_table(arms: Sequence[Arm]) -> pd.DataFrame:
    """Episodes run, failed (never completed, D-046) and excluded by reason (D-041)."""
    rows = []
    for a in arms:
        row = {"Arm": a.label, "run": len(a.episodes), "failed": len(a.failures)}
        row.update(a.exclusions["reason"].value_counts().to_dict() if len(a.exclusions) else {})
        rows.append(row)
    return pd.DataFrame(rows).fillna(0)


# ---------------------------------------------------------------------------- rendering
def to_markdown(df: pd.DataFrame, floatfmt: str = ".3f") -> str:
    """Floats rendered with ``floatfmt`` (NaN as a dash); strings stay as the table formatted them ("95.0")."""
    view = df.copy()
    for c in view.columns:
        if pd.api.types.is_float_dtype(view[c]):
            view[c] = [("—" if pd.isna(v) else format(v, floatfmt)) for v in view[c]]
        else:
            view[c] = ["—" if v is None or v is pd.NA else str(v) for v in view[c]]
    return view.to_markdown(index=False, disable_numparse=True)


def _tex(v: object) -> str:
    s = "—" if v is None or (isinstance(v, float) and np.isnan(v)) else (f"{v:.3f}" if isinstance(v, float) else str(v))
    for a, b in (("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#"), ("—", "--"), ("−", "-")):
        s = s.replace(a, b)
    return s


def to_latex(df: pd.DataFrame, caption: str, label: str) -> str:
    """booktabs table without pandas' Styler (no jinja2 needed)."""
    cols = list(df.columns)
    lines = [r"\begin{table}[t]", r"\centering", r"\small", rf"\caption{{{_tex(caption)}}}", rf"\label{{{label}}}",
             r"\begin{tabular}{l" + "r" * (len(cols) - 1) + "}", r"\toprule",
             " & ".join(_tex(c) for c in cols) + r" \\", r"\midrule"]
    lines += [" & ".join(_tex(v) for v in row) + r" \\" for row in df.itertuples(index=False)]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)
