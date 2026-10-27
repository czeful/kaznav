"""Tables for the supervisor's review of stage A (D-066, D-068, D-069): every arm of paper/matrices/dev_demo.yaml on
the same 57 held-out dev episodes (8 train scenes used nowhere else). Dev data, not the test: a preview.

    python reports/make_demo_2026-10-17.py            # -> reports/demo_2026-10-17/tables/*.csv + tablitsy.md

CSV files are UTF-8 with BOM, so Excel opens the Cyrillic labels correctly.
"""
from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "runs" / "matrix" / "dev_demo"
OUT = REPO / "reports" / "demo_2026-10-17" / "tables"

# key: (вариант, модель, язык команды, проверка перед остановкой)
ARMS = {
    "vlfm_en": ("Английский робот без VLM (как VLFM)", "—", "en (слово категории)", "нет"),
    "oracle": ("Идеальный робот (потолок навигации)", "—", "en (слово категории)", "нет"),
    "qolda_kk_fix": ("Qolda kk", "Qolda-4B", "kk", "да"),
    "qolda_kkru_fix": ("Qolda kk_ru", "Qolda-4B", "kk_ru", "да"),
    "qolda_en_fix": ("Qolda en", "Qolda-4B", "en", "да"),
    "qwen_kk": ("Qwen kk", "Qwen3-VL-4B", "kk", "да"),
    "qwen_kkru": ("Qwen kk_ru", "Qwen3-VL-4B", "kk_ru", "да"),
    "qwen_en": ("Qwen en", "Qwen3-VL-4B", "en", "да"),
    "mt_qwen_kk": ("Переводчик NLLB → Qwen (kk)", "NLLB + Qwen3-VL-4B", "kk → en", "да"),
    "mt_qwen_kkru": ("Переводчик NLLB → Qwen (kk_ru)", "NLLB + Qwen3-VL-4B", "kk_ru → en", "да"),
    "qolda_kk_nv": ("Qolda kk без проверки", "Qolda-4B", "kk", "нет"),
    "qolda_kkru_nv": ("Qolda kk_ru без проверки", "Qolda-4B", "kk_ru", "нет"),
    "qolda_en_nv": ("Qolda en без проверки", "Qolda-4B", "en", "нет"),
    "qwen_kk_nv": ("Qwen kk без проверки", "Qwen3-VL-4B", "kk", "нет"),
    "qwen_kkru_nv": ("Qwen kk_ru без проверки", "Qwen3-VL-4B", "kk_ru", "нет"),
    "qwen_en_nv": ("Qwen en без проверки", "Qwen3-VL-4B", "en", "нет"),
    "qolda_kk": ("Qolda kk — ДО исправления D-069", "Qolda-4B", "kk", "да (с ошибкой)"),
    "qolda_kkru": ("Qolda kk_ru — ДО исправления D-069", "Qolda-4B", "kk_ru", "да (с ошибкой)"),
    "qolda_en": ("Qolda en — ДО исправления D-069", "Qolda-4B", "en", "да (с ошибкой)"),
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
    ("qolda_kk_fix", "qwen_kk", "H3: казахская модель против мультиязычной на kk"),
    ("qolda_kkru_fix", "qwen_kkru", "H3: то же на смешанной kk_ru"),
    ("qolda_kk_nv", "qwen_kk_nv", "H3 без проверки (только понимание команды)"),
    ("qolda_kkru_nv", "qwen_kkru_nv", "H3 без проверки, kk_ru"),
    ("qolda_en_fix", "qwen_en", "контроль: на английском Qolda не обязана выигрывать"),
    ("qolda_en_fix", "qolda_kk_fix", "H1: сколько Qolda теряет на казахском"),
    ("qwen_en", "qwen_kk", "H1: сколько Qwen теряет на казахском"),
    ("qolda_kk_fix", "mt_qwen_kk", "родная модель против «перевести и отдать Qwen»"),
    ("qolda_kkru_fix", "mt_qwen_kkru", "то же на kk_ru"),
    ("qolda_kk_fix", "qolda_kk_nv", "H4: с проверкой против без (Qolda kk)"),
    ("qwen_kk", "qwen_kk_nv", "H4: с проверкой против без (Qwen kk)"),
    ("qwen_en", "qwen_en_nv", "H4: с проверкой против без (Qwen en)"),
    ("qolda_kk_fix", "qolda_kk", "исправление D-069: после против до (Qolda kk)"),
    ("qolda_kk_fix", "vlfm_en", "казахский робот на Qolda против английского робота без VLM"),
    ("qwen_en", "vlfm_en", "лучший английский VLM-вариант против робота без VLM"),
]


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson interval of ``k`` out of ``n``, in percent."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def mcnemar(a: pd.Series, b: pd.Series):
    """Discordant counts (only A, only B) and the exact two-sided McNemar p-value of two success series."""
    oa, ob = int((a & ~b).sum()), int((~a & b).sum())
    n = oa + ob
    p = min(1.0, 2 * sum(math.comb(n, i) for i in range(min(oa, ob) + 1)) / 2 ** n) if n else 1.0
    return oa, ob, p


def load(key: str) -> pd.DataFrame:
    """An arm's ``episodes.csv`` indexed by episode id."""
    return pd.read_csv(RUNS / key / "seed0" / "episodes.csv").set_index("episode_id").sort_index()


def main() -> None:
    """Build the stage-A tables; all arms must have run the same episodes."""
    OUT.mkdir(parents=True, exist_ok=True)
    data = {k: load(k) for k in ARMS}
    ids = set(data["vlfm_en"].index)
    assert all(set(d.index) == ids for d in data.values()), "arms ran different episodes"

    rows = []
    for k, (name, model, lang, ver) in ARMS.items():
        d = data[k]
        s, n = int(d.success.sum()), len(d)
        lo, hi = wilson(s, n)
        st = d.failure_stage.value_counts()
        row = {"Вариант": name, "Модель": model, "Язык команды": lang, "Проверка": ver, "Эпизодов": n,
               "Успехов": s, "Успех, %": round(100 * s / n, 1), "95% ДИ, %": f"{lo:.0f}–{hi:.0f}",
               "SPL": round(float(d.spl.mean()), 3)}
        for code, (ru, _) in STAGES.items():
            row[ru] = int(st.get(code, 0))
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "01_svodnaya.csv", index=False, encoding="utf-8-sig")

    prs = []
    for a, b, q in PAIRS:
        da, db = data[a], data[b]
        oa, ob, p = mcnemar(da.success.astype(bool), db.success.astype(bool))
        prs.append({"Вопрос": q, "A": ARMS[a][0], "B": ARMS[b][0], "Успехов A": int(da.success.sum()),
                    "Успехов B": int(db.success.sum()),
                    "Разница A − B, п.": round(100 * (da.success.mean() - db.success.mean()), 1),
                    "только A": oa, "только B": ob, "p (точный McNemar)": round(p, 3)})
    pairs = pd.DataFrame(prs)
    pairs.to_csv(OUT / "02_sravneniya.csv", index=False, encoding="utf-8-sig")

    cats = []
    for k, (name, *_rest) in ARMS.items():
        d = data[k]
        row = {"Вариант": name}
        for c, ru in CATS.items():
            g = d[d.goal_category == c]
            row[f"{ru} (из {len(g)})"] = int(g.success.sum())
        cats.append(row)
    cat_df = pd.DataFrame(cats)
    cat_df.to_csv(OUT / "03_po_kategoriyam.csv", index=False, encoding="utf-8-sig")

    dec = []
    for en, kk, who in (("qolda_en_fix", "qolda_kk_fix", "Qolda"), ("qwen_en", "qwen_kk", "Qwen")):
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

    md = ["# Этап А — таблицы для показа (dev, не тест)", "",
          "57 одинаковых эпизодов в 8 квартирах HM3D train, которые нигде раньше не использовались; колёсный робот-пылесос",
          "(камера 0.15 м), лимит 800 шагов; проверка откалибрована (D-067) и исправлена (D-069), если не сказано иное.",
          "На 57 эпизодах 95% доверительный интервал успеха — около ±12 п.: разницы меньше ~10 п. не доказаны.", "",
          "## 1. Сводная", "", summary.to_markdown(index=False), "",
          "## 2. Попарные сравнения на одних и тех же эпизодах", "",
          "«только A» — эпизоды, где нашёл только A; p < 0.05 — значимо.", "", pairs.to_markdown(index=False), "",
          "## 3. По категориям (успехов)", "", cat_df.to_markdown(index=False), "",
          "## 4. Откуда разрыв «английский − казахский» (H1)", "",
          "Вклад этапа = доля эпизодов, проваленных на этом этапе в kk, минус то же в en; сумма вкладов = разрыв.", "",
          dec_df.to_markdown(index=False), ""]
    (OUT / "tablitsy.md").write_text("\n".join(md), encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
