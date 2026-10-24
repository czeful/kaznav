"""Tables of stage 4 (D-073): every arm of paper/matrices/dev_confirm.yaml on the same 72 episodes of 10 fresh dev
scenes (splits/hm3d_v1_dev3.txt). Dev data, not the test. Arms still running are left out (``--partial``).

    python reports/make_confirm_2026-10.py [--partial]     # -> reports/confirm_2026-10/tables/*.csv + tablitsy.md

CSV files are UTF-8 with BOM, so Excel opens the Cyrillic labels correctly.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs" / "matrix" / "dev_confirm"
OUT = REPO / "reports" / "confirm_2026-10" / "tables"

NEW = "новые"
# key: (вариант, модель, язык команды, настройки)
ARMS = {
    "vlfm_en": ("Английский робот без VLM (как VLFM)", "—", "en (слово категории)", "остановка 0.6 м"),
    "oracle": ("Идеальный робот (потолок навигации)", "—", "en (слово категории)", "остановка по видимости"),
    "qolda_kk": ("Qolda kk", "Qolda-4B", "kk", NEW),
    "qolda_kkru": ("Qolda kk_ru", "Qolda-4B", "kk_ru", NEW),
    "qolda_en": ("Qolda en", "Qolda-4B", "en", NEW),
    "qwen_kk": ("Qwen kk", "Qwen3-VL-4B", "kk", NEW),
    "qwen_kkru": ("Qwen kk_ru", "Qwen3-VL-4B", "kk_ru", NEW),
    "qwen_en": ("Qwen en", "Qwen3-VL-4B", "en", NEW),
    "mt_qwen_kk": ("Переводчик NLLB → Qwen (kk)", "NLLB + Qwen3-VL-4B", "kk → en", NEW),
    "mt_qwen_kkru": ("Переводчик NLLB → Qwen (kk_ru)", "NLLB + Qwen3-VL-4B", "kk_ru → en", NEW),
    "qolda_kk_nv": ("Qolda kk без проверки", "Qolda-4B", "kk", "без проверки"),
    "qwen_kk_nv": ("Qwen kk без проверки", "Qwen3-VL-4B", "kk", "без проверки"),
    "qolda_kk_cmdq": ("Qolda kk, вопрос по-казахски", "Qolda-4B", "kk", "новые, вопрос на языке команды"),
    "qwen_kk_cmdq": ("Qwen kk, вопрос по-казахски", "Qwen3-VL-4B", "kk", "новые, вопрос на языке команды"),
    "qolda_kk_old": ("Qolda kk — старые настройки (этап А)", "Qolda-4B", "kk", "как в этапе А"),
    "qwen_en_old": ("Qwen en — старые настройки (этап А)", "Qwen3-VL-4B", "en", "как в этапе А"),
}
STAGES = {  # failure stage -> (русское название, группа)
    "query": ("не понял команду", "язык"),
    "not_detected": ("цель в кадре, детектор не узнал", "восприятие"),
    "verify_reject": ("проверка отвергла настоящую цель", "восприятие"),
    "false_stop": ("остановился у чужого предмета", "восприятие"),
    "not_seen": ("цель ни разу не попала в кадр", "навигация"),
    "not_reached": ("видел цель, но не доехал", "навигация"),
    "stuck": ("застрял", "навигация"),
    "stop_too_far": ("остановился у цели, но далеко", "навигация"),
}
CATS = {"bed": "кровать", "chair": "стул", "plant": "растение", "sofa": "диван", "toilet": "унитаз",
        "tv_monitor": "телевизор"}
PAIRS = [  # (a, b, вопрос)
    ("qolda_kk", "qolda_kk_old", "было → стало: Qolda kk, новые против старых настроек"),
    ("qwen_en", "qwen_en_old", "было → стало: Qwen en, новые против старых настроек"),
    ("qolda_kk", "qolda_kk_cmdq", "вопрос по-английски против вопроса на языке команды (Qolda kk)"),
    ("qwen_kk", "qwen_kk_cmdq", "то же для Qwen kk"),
    ("qolda_kk", "qwen_kk", "H3: казахская модель против мультиязычной на kk"),
    ("qolda_kkru", "qwen_kkru", "H3: то же на смешанной kk_ru"),
    ("qolda_kk", "mt_qwen_kk", "родная модель против «перевести и отдать Qwen» (kk)"),
    ("qolda_kkru", "mt_qwen_kkru", "то же на kk_ru"),
    ("qolda_en", "qolda_kk", "H1: сколько Qolda теряет на казахском"),
    ("qwen_en", "qwen_kk", "H1: сколько Qwen теряет на казахском"),
    ("qolda_kk", "qolda_kk_nv", "H4: с проверкой против без (Qolda kk)"),
    ("qwen_kk", "qwen_kk_nv", "H4: с проверкой против без (Qwen kk)"),
    ("qolda_kk", "vlfm_en", "казахский робот на Qolda против английского робота без VLM"),
    ("qwen_en", "vlfm_en", "лучший английский VLM-вариант против робота без VLM"),
]


def wilson(k: int, n: int, z: float = 1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def mcnemar(a: pd.Series, b: pd.Series):
    oa, ob = int((a & ~b).sum()), int((~a & b).sum())
    n = oa + ob
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(oa, ob) + 1)) / 2 ** n) if n else 1.0
    return oa, ob, p


def load(key: str):
    f = RUNS / key / "seed0" / "episodes.csv"
    return pd.read_csv(f).set_index("episode_id").sort_index() if f.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--partial", action="store_true", help="leave out arms that have not finished all episodes")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data = {k: load(k) for k in ARMS}
    split = REPO / "splits" / "hm3d_v1_dev3.txt"
    n_full = sum(1 for line in split.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#"))
    missing = [k for k, d in data.items() if d is None or len(d) < n_full]
    if missing and not args.partial:
        raise SystemExit(f"arms not finished: {missing} (use --partial for a preview)")
    data = {k: d for k, d in data.items() if k not in missing}
    if not data:
        raise SystemExit("no arm has finished yet")
    ids = set(next(iter(data.values())).index)
    assert all(set(d.index) == ids for d in data.values()), "arms ran different episodes"

    rows = []
    for k, d in data.items():
        name, model, lang, setting = ARMS[k]
        s, n = int(d.success.sum()), len(d)
        lo, hi = wilson(s, n)
        st = d.failure_stage.value_counts()
        row = {"Вариант": name, "Модель": model, "Язык команды": lang, "Настройки": setting, "Эпизодов": n,
               "Успехов": s, "Успех, %": round(100 * s / n, 1), "95% ДИ, %": f"{lo:.0f}–{hi:.0f}",
               "SPL": round(float(d.spl.mean()), 3), "Шагов (медиана)": int(d.num_decision_steps.median())}
        for code, (ru, _) in STAGES.items():
            row[ru] = int(st.get(code, 0))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "01_svodnaya.csv", index=False, encoding="utf-8-sig")

    prs = []
    for a, b, q in PAIRS:
        if a not in data or b not in data:
            continue
        da, db = data[a], data[b]
        oa, ob, p = mcnemar(da.success.astype(bool), db.success.astype(bool))
        prs.append({"Вопрос": q, "A": ARMS[a][0], "B": ARMS[b][0], "Успехов A": int(da.success.sum()),
                    "Успехов B": int(db.success.sum()),
                    "Разница A − B, п.": round(100 * (da.success.mean() - db.success.mean()), 1),
                    "только A": oa, "только B": ob, "p (точный McNemar)": round(p, 3)})
    pairs = pd.DataFrame(prs)
    pairs.to_csv(OUT / "02_sravneniya.csv", index=False, encoding="utf-8-sig")

    cats = []
    for k, d in data.items():
        row = {"Вариант": ARMS[k][0]}
        for c, ru in CATS.items():
            g = d[d.goal_category == c]
            row[f"{ru} (из {len(g)})"] = int(g.success.sum())
        cats.append(row)
    cat_df = pd.DataFrame(cats)
    cat_df.to_csv(OUT / "03_po_kategoriyam.csv", index=False, encoding="utf-8-sig")

    dec = []
    for en, kk, who in (("qolda_en", "qolda_kk", "Qolda"), ("qwen_en", "qwen_kk", "Qwen")):
        if en not in data or kk not in data:
            continue
        a, b = data[en], data[kk]
        gap = 100 * (a.success.mean() - b.success.mean())
        groups = {}
        for code, (_, grp) in STAGES.items():
            groups[grp] = groups.get(grp, 0.0) + 100 * ((b.failure_stage == code).mean() - (a.failure_stage == code).mean())
        dec.append({"Модель": who, "Успех en, %": round(100 * a.success.mean(), 1),
                    "Успех kk, %": round(100 * b.success.mean(), 1), "Разрыв en − kk, п.": round(gap, 1),
                    **{f"из-за: {g}, п.": round(v, 1) for g, v in groups.items()}})
    dec_df = pd.DataFrame(dec)
    dec_df.to_csv(OUT / "04_razryv_kk_en.csv", index=False, encoding="utf-8-sig")

    n = len(ids)
    md = ["# Этап 4 — подтверждение на свежих сценах (dev3, не тест)", "",
          f"{n} одинаковых эпизодов в 10 сценах HM3D train, нигде раньше не использованных; колёсный робот-пылесос",
          "(камера 0.15 м), лимит 800 шагов. «Новые» настройки: промпт v4, английский вопрос про короткое имя цели,",
          "«последний метр», калибровка на 18 сценах (D-070…D-073). Ничего не выбирается.",
          f"95% доверительный интервал при n = {n} — около ±11 п.: разницы меньше ~10 п. не доказаны.",
          *([f"", f"**Предварительно: без плеч {', '.join(missing)}.**"] if missing else []), "",
          "## 1. Сводная", "", summary.to_markdown(index=False), "",
          "## 2. Попарные сравнения на одних и тех же эпизодах", "",
          "«только A» — эпизоды, где нашёл только A; p < 0.05 — значимо (без поправки на множественность).", "",
          pairs.to_markdown(index=False) if len(pairs) else "—", "",
          "## 3. По категориям (успехов)", "", cat_df.to_markdown(index=False), "",
          "## 4. Откуда разрыв «английский − казахский» (H1)", "",
          "Вклад этапа = доля эпизодов, проваленных на этом этапе в kk, минус то же в en; сумма вкладов = разрыв.", "",
          dec_df.to_markdown(index=False) if len(dec_df) else "—", ""]
    (OUT / "tablitsy.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
