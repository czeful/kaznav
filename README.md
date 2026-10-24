# kznav — стенд zero-shot ObjectNav для низкого колёсного робота со сменными VLM

Колёсный робот формата робота-пылесоса в Habitat ищет предмет по команде на казахском,
смешанном казахско-русском или английском языке. Языковая часть работает через сменный
VLM-бэкенд (Qolda, Qwen3-VL, ...). На выходе — стандартные метрики ObjectNav (SR, SPL,
SoftSPL, DTS) с доверительными интервалами и парными сравнениями моделей.

- Постановка задачи: [docs/BRIEF.md](docs/BRIEF.md)
- Решения, влияющие на числа: [docs/decisions.md](docs/decisions.md)
- Определения метрик: [docs/measurement_protocol.md](docs/measurement_protocol.md)
- Что чинили при установке: [INSTALL_LOG.md](INSTALL_LOG.md)

## Состояние (2026-10-05)

| Этап | Статус |
|---|---|
| 2. Каркас: интерфейсы, конфиги, Mock/Replay, метрики с тестами | **готов** |
| 2′. Окружение: рендер Habitat в WSL2 | **работает** (Mesa → D3D12 → RTX 5070) |
| 2′. Видеопамять (D-006) | **замерено**: Qolda bf16 + Habitat + GroundingDINO + SigLIP2 — пик 11.8 из 12.2 GB, bf16 остаётся |
| 3. Симуляция: робот, движение, сенсоры, столкновения, застревание, обрывы | **готов на тестовых сценах**; на HM3D не проверено (нет данных) |
| 3. Наклон камеры | ждёт HM3D |
| 4. Карта занятости, фронтиры, планировщик, политика ObjectNav, раннер эпизода | **работает** на отладочной сцене; на HM3D не запускалось |
| 4. Детектор GroundingDINO (HF), карта ценности SigLIP2 (VLFM-стиль) | **работает**; HF-реализация совпадает с оригиналом (89/89 рамок, D-009) |
| 5. Qolda: свой сервер, разбор команды → гипотезы, верификация перед остановкой, байесовские веса | **работает в цикле навигации** (D-028); калибровка на dev — ждёт HM3D |
| 5. Языковая часть на данных пилота (30 команд, 32 кропа) | **замерено** (D-029) |
| 5. Абляции цели: одна гипотеза (D-038), перевод NLLB-200 (D-039) | **замерено** на отладочной сцене и 30 командах пилота: перевод теряет культурные предметы (1/15 против 11/15 у Qolda), польза нескольких гипотез пока не показана |
| 6. Оценка, отчёты | метрики, CSV эпизодов, парные сравнения; раннер HM3D/OVON (D-041), калибровка по dev (D-042), таблицы статьи (D-044) **готовы, проверены без HM3D** — ждут данных |
| 7. Qwen3-VL-4B, матричный прогон | **работает** (свой сервер, D-040): на казахском Qwen 2/20 против 6/20 у Qolda, на английском 9/20 против 6/20; на культурных словах пилота 3/15 против 11/15 |
| 8. К статье: этап провала (D-045), метка остановки по содержимому и сбои эпизодов (D-046), статистика по сценам (D-047), банк естественных команд (D-048), мощность (D-049), матрица плеч (D-050), шумы и одометрия (D-051), [план анализа](docs/analysis_plan.md) | **готово, проверено без HM3D**; план — черновик до заморозки; команды пишут носители языка |
| 9. HM3D: данные v0.1, проверка целей, доступность эпизодов, кинематический робот (D-052, D-053) | **подключено**: val (20 сцен, 2000 эп.), train качается; дымовой прогон на val_mini с оракулом - кинематический робот 10/11 |

Тесты: `python -m pytest` (без видеокарты и без сервера модели; интеграционные с Habitat пропускаются без него).
Проверки физики — [docs/validation/](docs/validation/). Решения и замеры — [docs/decisions.md](docs/decisions.md).

Полный цикл с казахской командой работает на отладочной сцене:

- команда («Банан тұрған жерге бар.») → Qolda → английские гипотезы с весами;
- гипотезы → GroundingDINO, по ним же SigLIP2 строит карту ценности;
- кандидат → кроп → Qolda «иә/жоқ» с логпробами → байесовское обновление → остановка.

Отладочная матрица: 7 плеч × 20 одинаковых эпизодов, `runs/compare/matrix_wheeled_v2`, D-035.

- Без проверки GroundingDINO останавливается у чужого предмета в 19 из 20 эпизодов.
- С проверкой Qolda — 1 ложная остановка на 40 эпизодов; SR 30 % (kk) и 20 % (kk-ru) против 5 %.
- Полный повтор плеч воспроизводит матрицу эпизод в эпизод (D-036).

Числа отладочные, не результаты статьи: одна сцена, верификатор без калибровки,
команды — черновик без проверки носителем.

## Структура

```
kznav/
  eval/        метрики и статистика (без симулятора): metrics, stats, calibration,
               classification, failures, robot (столкновения/застревание), records, aggregate
  language/    VLMBackend + реестр; backends/: openai_compat, qolda, qwen_vl, mock, replay;
               prompts (версионированные файлы), schemas, logprobs, lexicon
  perception/  camera, occupancy (карта log-odds под габариты embodiment), frontiers, objects (локализация
               детекций), oracle (абляция), интерфейсы Detector / Segmenter / ValueEncoder + реестры
  policy/      belief (байес по гипотезам), calibration (калибраторы, порог, защита от утечки dev→test),
               controller (pure pursuit), planner (Дейкстра по своей карте), objectnav (политика)
  run/         episode (исполнитель + раннер эпизода), логирование, манифест, сборка конфига
  report/      video (кадр из панелей + карта сверху, запись MP4)
  sim/         env (обёртка Habitat для обоих embodiment), robot (URDF-робот на моторах),
               urdf (генерация из конфига), diff_drive, frames, arena, noise,
               validation (физические проверки), mesh_drive (езда по сканам, D-003),
               episodes (точки обзора, отладочные сцены), mock_env (аналитический симулятор для CI)
configs/       Hydra: backend/ embodiment/ dataset/ policy/ eval/; validation/ — параметры проверок
prompts/       <набор>/<задача>/v<N>.yaml — хеш каждого промпта пишется в каждый вызов
lexicon/       синонимы категорий по языкам (hm3d_objectnav.yaml — черновик, нужна проверка носителем)
tests/         pytest; tests/data/pilot_verification.jsonl — сырые данные пилота для сверки ECE
tools/         wsl.sh (запуск в habitat_env с настройками рендера), check_render.py,
               validate_robot.py, mesh_drive_test.py, compare_views.py, demo_drive.py
env/           точный состав habitat_env (conda + pip freeze)
qolda check/   пилот (отдельный git-репозиторий, не входит в этот)
```

## Команды

Все команды выполняются в WSL, окружение `habitat_env`. Скрипт `tools/wsl.sh`
активирует его и выставляет переменные рендера. Из Git Bash в Windows:

```bash
MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 --cd /mnt/c/Users/Bieber/Desktop/robot_simulation -- bash tools/wsl.sh <команда>
```

Внутри WSL, из корня проекта:

```bash
# тесты (без видеокарты и без сервера модели, ~10 с)
bash tools/wsl.sh python -m pytest

# проверка рендера Habitat: строка "Renderer:" должна называть NVIDIA
bash tools/wsl.sh python tools/check_render.py --out /tmp/render_check.png

# физика робота на арене из примитивов (9 сценариев, ~30 с) -> runs/validation/<время>/report.md
bash tools/wsl.sh python tools/validate_robot.py

# езда робота по собственному navmesh на сканах (D-003, ~3 мин) -> runs/mesh_drive/<время>/summary.md
# нужны тестовые сцены: python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path ~/kznav_data
bash tools/wsl.sh python tools/mesh_drive_test.py

# демо-видео: робот проезжает маршруты по сцене (камера, глубина, карта сверху) -> runs/demo/<время>/demo.mp4
# это НЕ эпизод ObjectNav: робот едет по своему navmesh, без восприятия и без модели
bash tools/wsl.sh python tools/demo_drive.py --routes 3

# отладочные эпизоды ObjectNav (apartment_1 + 4 объекта YCB, по умолчанию оракульный детектор) -> runs/debug/<время>/
# (episodes.csv, summary.md, events.jsonl, видео первых эпизодов); role: debug — не результаты статьи
bash tools/wsl.sh python tools/run_debug_episodes.py --videos 2
bash tools/wsl.sh python tools/run_debug_episodes.py embodiment=habitat_agent

# кадр робота (0.15 м) рядом с кадром агента Habitat (0.88 м) из тех же точек
bash tools/wsl.sh python tools/compare_views.py --scene ~/kznav_data/scene_datasets/habitat-test-scenes/apartment_1.glb --out views.png

# сервер Qolda (bf16, порт 23333, 4 тайла): servers/qolda_server.py в ~/qolda-env, в фоне
bash tools/start_qolda.sh            # или tools/restart_qolda.sh; лог runs/servers/qolda.log
# сервер Qwen3-VL-4B (bf16, порт 23334, тот же бюджет пикселей, D-040); сервер Qolda перед этим остановить
bash tools/start_qwen.sh             # лог runs/servers/qwen.log; затем backend=qwen3vl_4b в любой команде
curl http://localhost:23333/kznav/stats

# языковая часть на данных пилота: разбор 30 команд / верификация 32 кропов (1, 4, 12 тайлов)
bash tools/wsl.sh python tools/eval_language.py parse
bash tools/wsl.sh python tools/eval_language.py verify --tiles 1,4,12
# те же 30 команд через абляции цели: multi, top1, mt_word, mt_command (D-038, D-039)
bash tools/wsl.sh bash tools/run_language_ablations.sh

# эпизоды с казахской командой: разбор Qolda -> GroundingDINO + SigLIP2 -> верификация Qolda
bash tools/wsl.sh python tools/run_debug_episodes.py --lang kk detector=mm_grounding_dino_base value_encoder=siglip2_base
#   --lang kk_ru (смешанные), --no-verify (без верификации); видео с командой, гипотезами и ответами модели

# то же, с логом памяти GPU (nvidia-smi раз в секунду + пик сервера)
bash tools/wsl.sh bash tools/with_gpu_log.sh runs/servers/gpu.csv python tools/run_debug_episodes.py --lang kk ...

# матрица плеч на одних и тех же эпизодах + парные сравнения -> runs/compare/matrix_<время>/comparison.md
bash tools/wsl.sh bash tools/run_debug_matrix.sh
python tools/compare_runs.py --ref gd_value gd_value=runs/debug/<a> qolda_kk=runs/debug/<b>

# --- HM3D / OVON (D-041, D-052; данные - в $KZNAV_DATA_ROOT, токен Matterport - в ~/.config/kznav/matterport.env) ---
# сцены HM3D v0.1 + эпизоды ObjectNav v1; train - потоком и только 80 сцен с эпизодами (весь архив 26 GB не нужен)
bash tools/wsl.sh bash tools/download_hm3d.sh val
bash tools/wsl.sh bash tools/download_hm3d_subset.sh train
# цели против разметки сцен и какие эпизоды может пройти робот (episodes.csv для make_split --runnable)
bash tools/wsl.sh python tools/check_hm3d.py --dataset hm3d_objectnav_v1_val --per-scene 0 --embodiments wheeled_vacuum_kinematic --out runs/hm3d_check/val_kinematic
bash tools/wsl.sh python tools/diagnose_reachability.py --dataset hm3d_objectnav_v1_valmini --out runs/hm3d_check/reach
# дымовой прогон на val_mini (сцены val, роль smoke: не результат, ничего не настраивать)
bash tools/wsl.sh python tools/run_eval.py --lang none --out runs/smoke/x dataset=hm3d_objectnav_v1_valmini embodiment=wheeled_vacuum_kinematic detector=oracle_semantic
# колёсный робот в основной таблице - кинематика на своём navmesh (D-053): embodiment=wheeled_vacuum_kinematic;
# физический робот (embodiment=wheeled_vacuum) - для исследования устойчивости и валидации физики
# фиксированные выборки: dev из train-сцен, тест - сбалансированная подвыборка val (50 вложена в 100)
python tools/make_split.py dev  --dataset hm3d_objectnav_v1_train --scenes 8 --per-category 10 --out splits/hm3d_v1_dev.txt
python tools/make_split.py test --dataset hm3d_objectnav_v1_val --per-category 100 --out splits/hm3d_v1_test_primary.txt
python tools/make_split.py test --dataset hm3d_objectnav_v1_val --per-category 50 --out splits/hm3d_v1_test_secondary.txt
# таблица (а), стандартный агент: простая случайная выборка из всего val, без стратификации (D-064)
python tools/make_split.py test --dataset hm3d_objectnav_v1_val --random 300 --out splits/hm3d_v1_table_a.txt
# естественные команды (D-048): таблица для авторов -> импорт -> проверка (инструкция: docs/command_authoring.md)
python tools/commands.py sheet --lexicon lexicon/hm3d_objectnav.yaml --authors A01,A02,A03 --out commands/natural/hm3d_objectnav_v1_sheet.csv
python tools/commands.py import commands/natural/hm3d_objectnav_v1_sheet.csv --dataset hm3d_objectnav_v1 --lexicon lexicon/hm3d_objectnav.yaml --out commands/natural/hm3d_objectnav_v1.yaml
python tools/commands.py validate commands/natural/hm3d_objectnav_v1.yaml --min-test 10
# объём выборки до тестовых прогонов (D-049): по отладочным парам плеч -> таблица мощности
python tools/power_analysis.py estimate runs/debug/<плечо a> runs/debug/<плечо b>
python tools/power_analysis.py table --p-a 0.30 --psi 0.30 --m 5 --out paper/power
# прогон: любой робот / модель / язык / режим цели; --resume продолжает папку прогона
# восприятие по умолчанию - стек D-060 (MM-GroundingDINO base 0.50 + SAM 2.1 + слияние 0.45, D-064);
# абляции: detector=none|oracle_semantic|grounding_dino_tiny, segmenter=box, perception.detection.fusion.enabled=false
bash tools/wsl.sh python tools/run_eval.py --lang kk embodiment=wheeled_vacuum_kinematic backend=qolda value_encoder=siglip2_base run.episodes.subset_file=splits/hm3d_v1_dev.txt dataset=hm3d_objectnav_v1_train policy.stop.calibration_file=calib/qolda_dev.json
# калибровка проверки и порог - только по dev-прогонам (D-042)
python tools/fit_calibration.py runs/eval/<dev прогон> --out calib/qolda_dev.json
# наклон камеры (D-004) и бюджет эпизода (D-005) на dev
bash tools/wsl.sh python tools/dev_studies.py pitch --pitches -15,-5,0,10,20 --out runs/dev/pitch dataset=hm3d_objectnav_v1_train run.episodes.subset_file=splits/hm3d_v1_dev.txt
python tools/dev_studies.py budget runs/eval/<dev прогон с большим лимитом> --out runs/dev/budget
# вся матрица статьи (D-050): план, прогон (серверы поднимаются по очереди), описание для таблиц
bash tools/wsl.sh python tools/run_matrix.py plan paper/matrices/hm3d_v1_main.yaml
bash tools/wsl.sh python tools/run_matrix.py run  paper/matrices/hm3d_v1_main.yaml
bash tools/wsl.sh python tools/run_matrix.py spec paper/matrices/hm3d_v1_main.yaml            # + --part secondary
# таблицы и графики статьи по YAML-описанию плеч (D-044, D-047): контрасты по сценам, этапы провала,
# разложение разности SR по этапам (H1); пример на отладочных прогонах
python tools/make_paper_tables.py paper/specs/debug_ycb_example.yaml --out runs/paper_check
# сверка GroundingDINO: HF против оригинала (D-009)
bash tools/wsl.sh python tools/gdino_crosscheck.py frames --out runs/gdino_check

# абляции цели на эпизодах матрицы: одна гипотеза, NLLB-слово (с проверкой и без), NLLB-команда
# -> runs/ablation_arms_<время>.txt; NLLB-200 на CPU, первый запуск скачивает 2.5 GB
bash tools/wsl.sh bash tools/run_ablation_arms.sh

# Qwen3-VL-4B вместо Qolda: языковые проверки пилота + плечи kk / kk-ru / en / kk без проверки
bash tools/wsl.sh bash tools/run_qwen_arms.sh

# видео всех эпизодов плеч (полный повтор = проверка воспроизводимости), ролик и страница результатов
bash tools/wsl.sh bash tools/run_video_arms.sh
bash tools/wsl.sh python tools/make_demo_reel.py --out runs/demo/reel.mp4 <video.mp4>="подпись|вторая строка" ...
python tools/make_results_page.py --compare runs/compare/<dir> --parse runs/language/parse_<t>     --verify runs/language/verify_<t> --out runs/page/index.html --video reel.mp4 --poster poster.png
```

### Выбор компонентов

Всё выбирается конфигом, одной строкой переопределения Hydra:

```bash
backend=qolda | qwen3vl_4b | mock | replay | qolda_think_control
embodiment=wheeled_vacuum | habitat_agent
dataset=hm3d_objectnav_v1_val | hm3d_ovon_val_unseen | custom_room
goal=multi | top1 | mt_word | mt_command      # что разбор цели отдаёт конвейеру (D-038, D-039)
```

Новая модель добавляется так: файл `configs/backend/<name>.yaml` плюс либо подкласс
`OpenAICompatBackend` с `@BACKENDS.register("<name>")`, либо внешний класс через
`class_path: pkg.module:Class`. Навигационный код не меняется.

## Жёсткие правила (из пилота и брифа)

- Qolda — только No-Think. `enable_thinking: false` явно в каждом запросе. Без этого
  или с `true` бэкенд не создаётся, кроме конфига `qolda_think_control`.
- Каждый вызов модели (запрос, сырой ответ, все повторы, входное изображение)
  пишется в `vlm_calls.jsonl`. `backend=replay` пересчитывает всё без модели.
- Пороги и калибровка подбираются только на dev. Артефакт калибровки хранит id
  dev-эпизодов, и пересечение с оцениваемыми эпизодами — ошибка.
- Собственная комната (`dataset=custom_room`, `role: debug`) не входит в основные
  таблицы.
