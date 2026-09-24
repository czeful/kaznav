# kznav — стенд zero-shot ObjectNav для низкого колёсного робота со сменными VLM

Колёсный робот формата робота-пылесоса в Habitat ищет предмет по команде на казахском,
смешанном казахско-русском или английском языке. Языковая часть работает через сменный
VLM-бэкенд (Qolda, Qwen3-VL, ...). На выходе — стандартные метрики ObjectNav (SR, SPL,
SoftSPL, DTS) с доверительными интервалами и парными сравнениями моделей.

- Постановка задачи: [docs/BRIEF.md](docs/BRIEF.md)
- Решения, влияющие на числа: [docs/decisions.md](docs/decisions.md)
- Определения метрик: [docs/measurement_protocol.md](docs/measurement_protocol.md)
- Что чинили при установке: [INSTALL_LOG.md](INSTALL_LOG.md)

## Состояние (2026-09-24)

| Этап | Статус |
|---|---|
| 2. Каркас: интерфейсы, конфиги, Mock/Replay, метрики с тестами | **готов** |
| 2′. Окружение: рендер Habitat в WSL2 | **работает** (Mesa → D3D12 → RTX 5070) |
| 3. Симуляция: робот, движение, сенсоры, столкновения, застревание | **готов на тестовых сценах**; на HM3D не проверено (нет данных) |
| 3. Наклон камеры | ждёт HM3D |
| 4. Карта занятости, фронтиры, планировщик, политика ObjectNav, раннер эпизода | **работает** на отладочной сцене; на HM3D не запускалось |
| 4. Точки обзора под embodiment, отладочные эпизоды, видео эпизода | **готово** |
| 4. Детектор GroundingDINO, энкодер карты ценности | не начато (сейчас только оракульный детектор — абляция) |
| 2′. Замер видеопамяти, точность Qolda (D-006) | не сделано |
| 5. Qolda: разбор цели, верификация, калибровка | интерфейсы готовы, на реальной модели не запускалось |
| 6. Оценка, отчёты | метрики, статистика, CSV эпизодов готовы; сводный отчёт и матрица — нет |
| 7. Qwen3-VL-4B, матричный прогон | конфиг и класс есть, сервер не поднят |

Тесты: 265 (`python -m pytest`), из них 4 интеграционных с Habitat и smoke-тест всего цикла на `MockEnv` без GPU.
Проверки физики — [docs/validation/](docs/validation/).

Цикл ObjectNav (осмотреться → исследовать → найти → подъехать → стоп) работает на
отладочной сцене с оракульным детектором. Команд «оценка на HM3D / сравнение моделей»
пока нет: нужны данные HM3D и детектор; они появятся здесь, когда заработают.

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

# отладочные эпизоды ObjectNav (сцена + 4 стула, оракульный детектор) -> runs/debug/<время>/
# (episodes.csv, summary.md, events.jsonl, видео первых эпизодов); role: debug — не результаты статьи
bash tools/wsl.sh python tools/run_debug_episodes.py --videos 2
bash tools/wsl.sh python tools/run_debug_episodes.py embodiment=habitat_agent

# кадр робота (0.15 м) рядом с кадром агента Habitat (0.88 м) из тех же точек
bash tools/wsl.sh python tools/compare_views.py --scene ~/kznav_data/scene_datasets/habitat-test-scenes/apartment_1.glb --out views.png

# поднять Qolda (сервер пилота, bf16, порт 23333) — в отдельном терминале WSL
source ~/qolda-env/bin/activate
python "qolda check/local_server/qolda_server.py" --model issai/Qolda --served-name qolda-4b --port 23333
curl http://localhost:23333/v1/models
```

### Выбор компонентов

Всё выбирается конфигом, одной строкой переопределения Hydra:

```bash
backend=qolda | qwen3vl_4b | mock | replay | qolda_think_control
embodiment=wheeled_vacuum | habitat_agent
dataset=hm3d_objectnav_v1_val | hm3d_ovon_val_unseen | custom_room
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
