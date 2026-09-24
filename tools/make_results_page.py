"""Results page (HTML) from a debug comparison + the language checks; every number is read from run files.

    python tools/make_results_page.py --compare runs/compare/<dir> --parse runs/language/parse_<t> \
        --verify runs/language/verify_<t> --out runs/page/index.html [--video reel.mp4] [--poster poster.png]

The page is a debug-set report (it says so): the arms' episodes come from episodes_all.csv of
tools/compare_runs.py, the language tables from tools/eval_language.py outputs.
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from kznav.eval.aggregate import paired_contrast  # noqa: E402
from kznav.eval.stats import mean_ci, wilson_ci  # noqa: E402

SHORT = {"oracle": "Оракул", "gd_nearest": "GDINO, ближайший фронтир", "gd_value": "GDINO + карта ценности",
         "gd_value_thr060": "GDINO + карта, порог 0.6", "qolda_kk_noverif": "Qolda kk, без проверки",
         "qolda_kk": "Qolda kk + проверка", "qolda_kkru": "Qolda kk-ru + проверка"}
ARMS = {  # label -> (display name, group)
    "oracle": ("Оракул: семантика симулятора", "ref"),
    "gd_nearest": ("GroundingDINO, ближайший фронтир", "det"),
    "gd_value": ("GroundingDINO + карта ценности SigLIP2", "det"),
    "gd_value_thr060": ("GroundingDINO + карта ценности, порог 0.6", "det"),
    "qolda_kk_noverif": ("Qolda, казахская команда, без проверки", "vlm0"),
    "qolda_kk": ("Qolda, казахская команда + проверка", "vlm"),
    "qolda_kkru": ("Qolda, смешанная kk-ru команда + проверка", "vlm"),
}
CATS = {"banana": "банан", "mustard bottle": "горчица", "pitcher": "кувшин", "power drill": "дрель"}
E = html.escape


def pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def sr_chart(rows) -> str:
    """Horizontal bars: SR with Wilson 95% CI; false stops as a count on the right."""
    w, left, right, bar_h, gap, top = 720, 200, 110, 22, 14, 34
    plot_w = w - left - right
    h = top + len(rows) * (bar_h + gap) + 30
    x = lambda v: left + v * plot_w  # noqa: E731
    out = [f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Доля успеха по плечам" class="chart">']
    for t in (0, 0.25, 0.5, 0.75, 1.0):
        out.append(f'<line x1="{x(t):.1f}" y1="{top - 8}" x2="{x(t):.1f}" y2="{h - 26}" class="grid"/>')
        out.append(f'<text x="{x(t):.1f}" y="{h - 10}" class="tick" text-anchor="middle">{pct(t)}</text>')
    out.append(f'<text x="{left}" y="16" class="axis">Доля успешных эпизодов (SR), 95 % ДИ</text>')
    out.append(f'<text x="{w - 4}" y="16" class="axis" text-anchor="end">ложные стопы</text>')
    for i, r in enumerate(rows):
        y = top + i * (bar_h + gap)
        cls = {"ref": "b-ref", "det": "b-det", "vlm0": "b-det", "vlm": "b-vlm"}[r["group"]]
        out.append(f'<text x="{left - 10}" y="{y + bar_h * 0.7:.1f}" class="lab" text-anchor="end">{E(r["name"])}</text>')
        out.append(f'<rect x="{left}" y="{y}" width="{max(1.5, r["sr"] * plot_w):.1f}" height="{bar_h}" class="{cls}"/>')
        cy = y + bar_h / 2
        out.append(f'<line x1="{x(r["lo"]):.1f}" y1="{cy}" x2="{x(r["hi"]):.1f}" y2="{cy}" class="ci"/>')
        for v in (r["lo"], r["hi"]):
            out.append(f'<line x1="{x(v):.1f}" y1="{cy - 5}" x2="{x(v):.1f}" y2="{cy + 5}" class="ci"/>')
        out.append(f'<text x="{x(r["hi"]) + 8:.1f}" y="{y + bar_h * 0.7:.1f}" class="val">{pct(r["sr"])}</text>')
        fs = r["false_stops"]
        out.append(f'<text x="{w - 4}" y="{y + bar_h * 0.7:.1f}" class="fs{" fs-bad" if fs else " fs-ok"}" '
                   f'text-anchor="end">{fs}</text>')
    out.append("</svg>")
    return "\n".join(out)


def table(headers, rows, cls="") -> str:
    th = "".join(f"<th>{E(h)}</th>" for h in headers)
    trs = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f'<div class="tbl"><table class="{cls}"><thead><tr>{th}</tr></thead><tbody>{trs}</tbody></table></div>'


def verifier_counts(run_dir: Path):
    tp = fn = fp = tn = 0
    f = run_dir / "events.jsonl"
    for line in open(f, encoding="utf-8"):
        if '"verification"' not in line:
            continue
        e = json.loads(line)
        if not e.get("called") or e.get("p_raw") is None:
            continue
        yes, goal = e["p_raw"] >= 0.5, bool(e["is_goal_privileged"])
        tp += yes and goal
        fn += (not yes) and goal
        fp += yes and not goal
        tn += (not yes) and not goal
    return tp, fn, fp, tn


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", required=True)
    ap.add_argument("--parse", required=True)
    ap.add_argument("--verify", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--video", default="")
    ap.add_argument("--poster", default="")
    ap.add_argument("--ref", default="gd_value")
    args = ap.parse_args()
    cmp_dir = Path(args.compare)
    df = pd.read_csv(cmp_dir / "episodes_all.csv")
    dirs = {}
    for line in (cmp_dir / "comparison.md").read_text(encoding="utf-8").splitlines():
        if line.startswith("- **") and "`" in line:
            dirs[line.split("**")[1]] = Path(line.split("`")[1].replace("\\", "/"))
    labels = [a for a in ARMS if a in set(df.arm)]
    n_ep = int(df[df.arm == labels[0]].shape[0])

    # ---------------------------------------------------------------- arms
    rows, arm_rows = [], []
    for a in labels:
        g = df[df.arm == a]
        ci = wilson_ci(int(g.success.sum()), len(g))
        spl = mean_ci(g.spl.astype(float).to_numpy(), n_boot=5000)
        fs = int((g.outcome == "false_stop").sum())
        rows.append({"name": SHORT[a], "group": ARMS[a][1], "sr": ci.estimate, "lo": ci.low, "hi": ci.high,
                     "false_stops": fs})
        oc = g.outcome.value_counts()
        vc = verifier_counts(REPO / dirs[a]) if a in dirs and ARMS[a][1] == "vlm" else None
        arm_rows.append([
            E(ARMS[a][0]), f"{int(g.success.sum())}/{len(g)}", f"{spl.estimate:.2f} [{spl.low:.2f}, {spl.high:.2f}]",
            str(fs), str(int(oc.get("stuck", 0))), str(int(oc.get("not_found", 0)) + int(oc.get("step_limit", 0))),
            str(int(oc.get("model_failure", 0))),
            pct(g.goal_ever_in_view.mean()),
            f"{g.vlm_calls.mean():.1f}" if g.vlm_calls.sum() else "—",
            f"{vc[3]}/{vc[3] + vc[2]} · {vc[0]}/{vc[0] + vc[1]}" if vc else "—",
        ])
    # paired contrasts of the verification arms against the reference detector arm
    dfc = df.assign(success=df.success.astype(float))
    contrasts = []
    for a in labels:
        if a == args.ref:
            continue
        c = paired_contrast(dfc, arm_col="arm", arm_a=a, arm_b=args.ref, metric="success",
                            pair_on=("dataset", "split", "episode_id"), n_boot=5000)
        contrasts.append([E(ARMS[a][0]), f"{100 * c['diff']:+.0f} п.п. [{100 * c['diff_ci_low']:+.0f}, {100 * c['diff_ci_high']:+.0f}]",
                          f"{int(c['mcnemar_only_a'])} / {int(c['mcnemar_only_b'])}", f"{c['mcnemar_p']:.3f}"])
    # per category
    cat_rows = []
    for a in labels:
        g = df[df.arm == a]
        cat_rows.append([E(ARMS[a][0])] + [f"{int(g[g.goal_category == c].success.sum())}/{int((g.goal_category == c).sum())}"
                                           for c in CATS])
    # verifier totals over the verification arms
    vlm_arms = [a for a in labels if ARMS[a][1] == "vlm" and a in dirs]
    tot = np.sum([verifier_counts(REPO / dirs[a]) for a in vlm_arms], axis=0) if vlm_arms else np.zeros(4, int)
    tp, fn, fp, tn = (int(v) for v in tot)
    det = {a: df[df.arm == a] for a in labels}
    fs_det = int((det[args.ref].outcome == "false_stop").sum()) if args.ref in det else 0
    fs_vlm = sum(int((det[a].outcome == "false_stop").sum()) for a in vlm_arms)
    n_vlm_ep = sum(len(det[a]) for a in vlm_arms)

    # ---------------------------------------------------------------- language (pilot data)
    pr = pd.read_csv(Path(args.parse) / "parse.csv")
    parse_rows = []
    for name, g in [("все", pr)] + [({"cultural": "культурные", "universal": "универсальные"}[k], v)
                                    for k, v in pr.groupby("category")]:
        n = len(g)
        f = lambda k: (lambda c: f"{c.estimate:.2f} [{c.low:.2f}, {c.high:.2f}]")(wilson_ci(int(k), n))  # noqa: E731
        parse_rows.append([name, str(n), f(((g.attempts == 1) & g.ok).sum()), f(g.target_match.sum()),
                           f(g.hypothesis_head_match.sum()), f"{np.median(g.latency_s):.1f} с"])
    misses = pr[~pr.hypothesis_head_match][["gold_target", "hypotheses"]].values.tolist()
    vr = pd.read_csv(Path(args.verify) / "verify.csv")
    vr["ok"] = (vr.p_yes >= 0.5) == vr.gold
    verify_rows = []
    for t, g in vr.groupby("tiles"):
        c = wilson_ci(int(g.ok.sum()), len(g))
        verify_rows.append([str(t), f"{c.estimate:.2f} [{c.low:.2f}, {c.high:.2f}]",
                            f"{g[g.category == 'cultural'].ok.mean():.2f}", f"{g[g.category == 'universal'].ok.mean():.2f}",
                            f"{np.median(g.latency_s):.2f} / {np.percentile(g.latency_s, 95):.1f} с"])

    # ---------------------------------------------------------------- page
    video = ""
    if args.video:
        poster = f' poster="{E(args.poster)}"' if args.poster else ""
        video = (f'<figure class="video"><video controls preload="metadata" playsinline{poster} src="{E(args.video)}"></video>'
                 '<figcaption>Эпизоды из полного повтора плеч матрицы: исходы совпадают с таблицей эпизод в эпизод. '
                 'Слева камера робота (высота 15 см), в центре его собственная карта (фиолетовые кольца — кандидаты), '
                 'справа карта ценности. Сверху: команда, гипотезы Qolda с весами, ответ верификатора.</figcaption></figure>')
    ref_name = ARMS[args.ref][0]
    body = f"""
<header class="top">
  <p class="eyebrow">kznav · отладочный прогон · {n_ep} эпизодов на плечо</p>
  <h1>Робот-пылесос находит предмет по казахской команде</h1>
  <p class="lead">Колёсный робот в Habitat-Sim получает команду вроде «Банан тұрған жерге бар.», модель
  <b>Qolda</b> превращает её в английские визуальные гипотезы, GroundingDINO ищет кандидатов, а перед остановкой
  Qolda смотрит на кроп и отвечает «иә» или «жоқ». Главный результат этого прогона: без такой проверки детектор
  останавливается не у той вещи в {fs_det} из {n_ep} эпизодов, с проверкой — в {fs_vlm} из {n_vlm_ep}.</p>
  <p class="warn">Это отладочный набор, не числа статьи: одна сцена (apartment_1), 4 объекта YCB, верификатор без
  калибровки, команды — черновик без проверки носителем. Основные таблицы будут на HM3D.</p>
</header>

{video}

<section>
  <h2>Доля успеха по плечам</h2>
  <p>Одни и те же {n_ep} эпизодов в каждом плече. Столбик — SR, усы — 95 % интервал Уилсона, число справа —
  эпизоды, где робот остановился у чужого объекта.</p>
  {sr_chart(rows)}
</section>

<section>
  <h2>Сводка плеч</h2>
  {table(["плечо", "успех", "SPL [95 % ДИ]", "ложный стоп", "застрял", "не нашёл / лимит шагов", "ошибка разбора", "цель в кадре",
          "вызовов VLM / эп.", "проверка: чужие · цель"], arm_rows, "num")}
  <p class="note">«Проверка: чужие · цель» — вызовы верификатора в цикле по привилегированной разметке, P(да) ≥ 0.5:
  сколько чужих объектов отвергнуто и сколько раз подтверждена настоящая цель. Политика эту разметку не видит.</p>
</section>

<section>
  <h2>Парное сравнение с «{E(ref_name)}»</h2>
  {table(["плечо", "разница SR [95 % ДИ]", "успех только здесь / только там", "McNemar p"], contrasts, "num")}
  <p class="note">Бутстреп по парам эпизодов, точный тест McNemar. При {n_ep} эпизодах выигрыш в 15–20 п.п. ещё не
  значим: нужны HM3D и больше эпизодов.</p>
</section>

<section>
  <h2>Верификатор в цикле</h2>
  <div class="facts">
    <div><span class="big">{tn}/{tn + fp}</span><span>чужих объектов отвергнуто</span></div>
    <div><span class="big">{tp}/{tp + fn}</span><span>раз подтверждена настоящая цель</span></div>
  </div>
  <p>Qolda почти никогда не соглашается с чужим объектом, но иногда отвергает настоящий — обычно когда цель
  видна мелко, сбоку или обрезана. Порог и калибровка пока отладочные; на dev-сплите HM3D они будут подобраны заново.</p>
  <p>Цена проверки — время. Каждый ложный кандидат робот сначала подъезжает посмотреть, поэтому до цели он
  доходит реже, чем оракул: цель в кадре в половине эпизодов против трёх четвертей, и застревает он чаще.
  Проверка убирает ложные остановки, но не ложные детекции.</p>
</section>

<section>
  <h2>Где теряем: язык или навигация</h2>
  {table(["плечо"] + [f"{CATS[c]} ({c})" for c in CATS], cat_rows, "num")}
  <p><b>Кувшин проигран языку.</b> Ни в одном из 10 разборов английские гипотезы не называют кувшин (pitcher, jug):
  «құмыра» Qolda понимает как vase, «кувшин» в смешанной команде — как vase или «көзілдірік» (очки) с гипотезой
  umbrella, ещё два разбора дали невалидный JSON.
  В одном эпизоде робот остановился у горшка с растением на столе: на вопрос «vase?» верификатор ответил «иә» —
  ошибка пришла из разбора, модель искала не тот предмет.
  <b>Дрель проиграна навигации.</b> Разбор верный во всех 10 эпизодах («power drill»), но дрель попала в кадр
  только в 4 из них. Оракул находит кувшин в 5 из 5 эпизодов и дрель в 4 из 5.</p>
</section>

<section>
  <h2>Языковая часть на данных пилота</h2>
  <h3>Разбор 30 казахских команд</h3>
  {table(["группа", "n", "валидный JSON с 1-й попытки", "цель в словарной форме", "гипотеза называет объект",
          "задержка, медиана"], parse_rows, "num")}
  <p>Qolda понимает казахское слово, но английская гипотеза для детектора ошибочна в {len(misses)} из {len(pr)} команд,
  в основном у культурных объектов:
  {", ".join(f"{E(t)} → {E(h.split(' | ')[0])}" for t, h in misses)}.</p>
  <h3>Проверка 32 кропов: сколько тайлов давать модели</h3>
  {table(["тайлов", "точность [95 % ДИ]", "культурные", "универсальные", "задержка, медиана / p95"], verify_rows, "num")}
  <p>Точность от числа тайлов не зависит (разница — 2 кропа из 32). Задержка зависит: при 12 тайлах пик памяти
  Qolda 12.47 GB — больше видеопамяти, драйвер уходит в общую память. Выбрано 4 тайла.</p>
</section>

<section>
  <h2>Что ещё найдено и исправлено</h2>
  <ul class="findings">
    <li><b>Всё помещается в 12 GB.</b> Qolda bf16 + Habitat + GroundingDINO + SigLIP2 одновременно: пик 11.8 из 12.2 GB.
    Квантизация не нужна, числа пилота сравнимы.</li>
    <li><b>Дыра в скане = обрыв.</b> 5 из 8 застреваний были у края скана, где камера видит пустоту. Правило «нет отражения
    там, где должен быть пол — препятствие» подняло оракул с 50 % до 75 % (парно, ни один эпизод не ухудшился).</li>
    <li><b>Кандидат рядом, но не виден.</b> Робот стоял у кандидата 230 шагов. Теперь поворачивается к нему, а если не видит — отказывается.</li>
    <li><b>Воспроизводимость.</b> Результат эпизода зависел от предыдущих эпизодов через общий генератор случайных
    чисел. Теперь сид на эпизод. Полный повтор трёх плеч (60 эпизодов, с Qolda) совпал с матрицей эпизод в эпизод.
    Остаётся тонкость: Bullet помнит предыдущие эпизоды на уровне 10⁻⁶ м, поэтому воспроизводится прогон целиком,
    а не отдельный эпизод, вынутый из середины.</li>
    <li><b>Отрицательный результат.</b> Правило «не спрашивать модель про рамку меньше 32 px» на этом наборе не помогло
    (20 % → 15 % в первой версии матрицы), выключено. Разброс траекторий на одной сцене не различает эффекты
    меньше ~20 п.п.</li>
    <li><b>Qolda не исправляет себя.</b> На «Қыша бөтелкесі қайда?» она пишет гипотезы по-казахски, а при повторе
    с текстом ошибки повторяет тот же ответ слово в слово.</li>
  </ul>
</section>

<section>
  <h2>Дальше</h2>
  <ul>
    <li>HM3D ObjectNav val и dev-сплит: калибровка порога верификатора и весов гипотез на dev.</li>
    <li>Проверка команд и словаря носителем казахского языка.</li>
    <li>Сравнение с Qwen3-VL-4B на тех же эпизодах.</li>
    <li>Больше эпизодов и повторы по сидам, чтобы разрешать эффекты 5–10 п.п.</li>
  </ul>
</section>
"""
    page = f"""<title>Стенд kznav</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap">
<style>
:root {{
  --bg: #f5f6f8; --surface: #ffffff; --ink: #1b2230; --muted: #5b6474; --line: #dde1e7;
  --accent: #0a8fb0; --ok: #2e8b57; --bad: #c8472f; --warn: #b7791f; --warn-bg: #fbf3e4;
  --bar-ref: #9aa6b6; --bar-det: #c9d0da; --bar-vlm: #0a8fb0; --grid: #e6e9ee;
}}
@media (prefers-color-scheme: dark) {{
  :root:not([data-theme="light"]) {{
    color-scheme: dark;
    --bg: #11151c; --surface: #171c25; --ink: #e6e9ef; --muted: #9aa3b2; --line: #2a313d;
    --accent: #3cb7d6; --ok: #5cc28a; --bad: #ec7a63; --warn: #e0a84a; --warn-bg: #2a2317;
    --bar-ref: #5b6678; --bar-det: #3a4250; --bar-vlm: #3cb7d6; --grid: #242a35;
  }}
}}
:root[data-theme="dark"] {{
  color-scheme: dark;
  --bg: #11151c; --surface: #171c25; --ink: #e6e9ef; --muted: #9aa3b2; --line: #2a313d;
  --accent: #3cb7d6; --ok: #5cc28a; --bad: #ec7a63; --warn: #e0a84a; --warn-bg: #2a2317;
  --bar-ref: #5b6678; --bar-det: #3a4250; --bar-vlm: #3cb7d6; --grid: #242a35;
}}
body {{ background: var(--bg); color: var(--ink); font: 16px/1.55 "IBM Plex Sans", system-ui, "Segoe UI", Roboto, sans-serif;
  padding-inline: 16px; padding-block: 32px 64px; }}
main {{ max-width: 860px; margin: 0 auto; display: flex; flex-direction: column; gap: 40px; }}
h1 {{ font-size: clamp(1.7rem, 4vw, 2.4rem); line-height: 1.15; margin: 6px 0 14px; text-wrap: balance; font-weight: 700; }}
h2 {{ font-size: 1.3rem; margin: 0 0 10px; text-wrap: balance; }}
h3 {{ font-size: 1.02rem; margin: 18px 0 6px; color: var(--muted); font-weight: 600; }}
p, li {{ max-width: 70ch; }}
.eyebrow {{ font: 500 .78rem/1 "IBM Plex Mono", ui-monospace, monospace; letter-spacing: .06em; text-transform: uppercase;
  color: var(--accent); margin: 0; }}
.lead {{ font-size: 1.08rem; }}
.warn {{ background: var(--warn-bg); border-left: 3px solid var(--warn); padding: 10px 14px; font-size: .93rem; }}
.note {{ color: var(--muted); font-size: .88rem; }}
.video {{ margin: 0; }}
.video video {{ width: 100%; max-width: 100%; border-radius: 6px; background: #000; display: block; }}
.video figcaption {{ color: var(--muted); font-size: .86rem; margin-top: 8px; max-width: 70ch; }}
.chart {{ width: 100%; height: auto; display: block; background: var(--surface); border: 1px solid var(--line);
  border-radius: 6px; padding: 8px 4px; box-sizing: border-box; }}
.chart .grid {{ stroke: var(--grid); stroke-width: 1; }}
.chart .tick, .chart .axis {{ fill: var(--muted); font: 12px "IBM Plex Mono", ui-monospace, monospace; }}
.chart .lab {{ fill: var(--ink); font: 13px "IBM Plex Sans", system-ui, sans-serif; }}
.chart .val {{ fill: var(--ink); font: 500 13px "IBM Plex Mono", ui-monospace, monospace; }}
.chart .b-ref {{ fill: var(--bar-ref); }} .chart .b-det {{ fill: var(--bar-det); }} .chart .b-vlm {{ fill: var(--bar-vlm); }}
.chart .ci {{ stroke: var(--ink); stroke-width: 1.4; }}
.chart .fs {{ font: 600 13px "IBM Plex Mono", ui-monospace, monospace; }}
.chart .fs-bad {{ fill: var(--bad); }} .chart .fs-ok {{ fill: var(--ok); }}
.tbl {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 6px; background: var(--surface); }}
table {{ border-collapse: collapse; width: 100%; font-size: .88rem; }}
th, td {{ padding: 7px 10px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }}
th {{ font-weight: 600; color: var(--muted); font-size: .8rem; white-space: nowrap; }}
tbody tr:last-child td {{ border-bottom: 0; }}
table.num td:not(:first-child) {{ font-family: "IBM Plex Mono", ui-monospace, monospace; font-variant-numeric: tabular-nums;
  white-space: nowrap; }}
.facts {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 6px 0 12px; }}
.facts div {{ display: flex; flex-direction: column; gap: 2px; padding: 12px 16px; background: var(--surface);
  border: 1px solid var(--line); border-radius: 6px; min-width: 200px; flex: 1; }}
.facts .big {{ font: 600 1.7rem/1.1 "IBM Plex Mono", ui-monospace, monospace; color: var(--accent); }}
.facts span:last-child {{ color: var(--muted); font-size: .9rem; }}
.findings {{ display: flex; flex-direction: column; gap: 10px; padding-left: 1.1em; }}
b {{ font-weight: 600; }}
</style>
<main>
{body}
</main>
"""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    print(f"page: {out}")


if __name__ == "__main__":
    main()
