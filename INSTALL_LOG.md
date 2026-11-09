# Журнал установки

Что пришлось чинить при установке, в порядке появления. Предыдущая часть (пилот:
habitat_env, nav_env с GroundingDINO, сервер Qolda) —
`qolda check/third_party/INSTALL_LOG.md`, её выводы здесь не повторяются.

Машина: Windows 11 + WSL2 Ubuntu 22.04, RTX 5070 12 GB (Blackwell, sm_120,
драйвер 591.86 / CUDA 13.1), Ryzen 7 7800X3D, 31 GB RAM (в WSL ограничено до 12 GB
в `.wslconfig`).

## Окружения

| Где | Что | Роль в стенде |
|---|---|---|
| conda `habitat_env` (py 3.9.23) | habitat-sim 0.2.4 (withbullet, headless), habitat-lab 0.2.4, torch 2.8.0+cu128 | основное окружение стенда |
| conda `nav_env` (py 3.10) | оригинальный GroundingDINO, собранный под sm_120 | детектор-бенчмарк пилота; для сверки (D-009) |
| venv `~/qolda-env` (py 3.13) | transformers 4.57.6, torch 2.12, fastapi; lmdeploy 0.17.0 (не работает с Qolda) | сервер Qolda (`qolda check/local_server/qolda_server.py`) |

Точный состав всех окружений (`habitat_env`, `qolda-env`, `avl-env`, `nemo-env`) и закреплённые версии моделей — `env/` (`bash docker/lock_envs.sh`); из него собираются Docker-образы, `docker/README.md` (D-089).

## 2026-09-24

### 1. Habitat не рендерит в WSL2: `unable to find CUDA device 0 among 1 EGL devices`

Причина: headless-сборка habitat-sim ищет EGL-устройство NVIDIA с тем же номером,
что у CUDA-устройства. В WSL2 NVIDIA EGL нет (есть `libcuda` и мост D3D12
`libd3d12.so`, `/dev/dxg`, нет `/dev/dri`). Единственное EGL-устройство — Mesa.

Исправление:

- `SimulatorConfiguration.gpu_device_id = -1` — отключает поиск CUDA-устройства,
  Magnum берёт EGL-устройство по умолчанию;
- `GALLIUM_DRIVER=d3d12` и `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA`.

**Без второй переменной Mesa выбирает первый адаптер — встроенную графику AMD**
(`Renderer: D3D12 (AMD Radeon(TM) Graphics)`). Картинка при этом та же, но глубина
отличается в 6-м знаке. Переменные выставляет `tools/wsl.sh`, строка `Renderer:`
в логе показывает фактический адаптер.

Проверка: `bash tools/wsl.sh python tools/check_render.py`:
`Renderer: D3D12 (NVIDIA GeForce RTX 5070)`, 2.7 мс на кадр RGB+D 640×480,
OpenGL 4.2 (Mesa 23.2.1).

### 2. Пакеты, добавленные в habitat_env

`pip install "pydantic>=2.6,<3" "httpx>=0.27,<1" "pytest>=8,<9" "pandas>=2.2,<2.4" "tabulate>=0.9"`
→ pydantic 2.13.5, httpx 0.28.1, pytest 8.4.2, pandas 2.3.3, tabulate 0.9.0.
Конфликтов с habitat-lab 0.2.4 не возникло. Код стенда совместим с Python 3.9:
без `X | Y` в аннотациях, которые вычисляются во время выполнения (pydantic).

### 3. Вызов WSL из Git Bash

Как в пилоте: Git Bash переписывает POSIX-пути в аргументах `wsl.exe`. Команды
запускаются с `MSYS_NO_PATHCONV=1` и `--cd`:

```bash
MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 --cd /mnt/c/Users/Bieber/Desktop/robot_simulation -- bash tools/wsl.sh python -m pytest
```

Команды с `$VAR` внутри `wsl -- bash -lc '...'` теряют переменные: `wsl.exe`
пересобирает командную строку. Надёжнее класть команды в скрипт.

### 4. Особенности habitat-sim 0.2.4, найденные на этапе 3

Каждая из них молча портит физику; все обойдены в коде и покрыты тестами или проверками.

| # | Симптом | Причина | Где обойдено |
|---|---|---|---|
| a | Физический конфиг не применяется (шаг 0.008 вместо 1/240) | файл распознаётся только по суффиксу `.physics_config.json` | `sim/env.py` |
| b | Боксы-препятствия вдвое больше заданного | примитив `cubeSolid` занимает [-1, 1]³, `scale` = половина размера | `sim/arena.py` |
| c | Статический бокс остаётся в начале координат | позицию надо задать **до** перевода в `STATIC` | `sim/arena.py` |
| d | Робот появляется носом в полу, силы контакта 10⁵ Н | корень сочленённого объекта — инерциальная система базового звена (D-017) | `sim/urdf.py` |
| e | Segfault при `set_link_friction(-1, …)` | базовое звено (id −1) не поддерживается | трение корпуса — тегом `<contact>` в URDF |
| f | Робот стоит на 4 см выше пола сканов | margin статической сцены по умолчанию 0.04 м (D-015) | шаблон сцены до загрузки |
| g | `reconfigure()` не меняет сцену | сравнивает с тем же объектом `SimulatorConfiguration`; нужен новый объект | `sim/env.py` |
| h | Ложные контакты колёс на сканах | цилиндр цепляет внутренние рёбра треугольников (D-016) | сферические коллайдеры колёс |
| i | Два `Simulator` в одном процессе: abort при `close()` | один живой GL-контекст на процесс | симуляторы создаются по очереди |
| j | Нет `Simulator.previous_step_collided` | в 0.2.4 признак возвращает `agent.act()` | `sim/env.py` |

### 5. Тестовые сцены Habitat

`python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path ~/kznav_data`
→ 106 MB (`apartment_1`, `skokloster-castle`, `van-gogh-room` с navmesh).

### 6. MP4 (H.264) для видео

В `habitat_env` стоял conda-пакет `imageio-ffmpeg` без бинарника ffmpeg (ожидает системный).
Исправление: `pip install --force-reinstall --no-deps "imageio-ffmpeg==0.6.0"` — колесо со
встроенным ffmpeg 7.0.2 (static). OpenCV сам пишет только `mp4v`, который стандартный плеер
Windows может не открыть.

### 7. Модели восприятия в habitat_env

`transformers` 4.57.6, safetensors 0.7.0, tokenizers 0.22.2, torch 2.8.0+cu128.
Веса из HF: `IDEA-Research/grounding-dino-tiny` (детектор, D-009) и
`google/siglip2-base-patch16-224` (карта ценности, fp16). Кэш — `~/.cache/huggingface`.
Собственных CUDA-расширений нет, поэтому сборка под sm_120, как в пилоте, не нужна.

### 8. Сервер Qolda (`servers/qolda_server.py`) и отладочные объекты YCB

- Окружение `~/qolda-env`: torch 2.12.1+cu130, torchvision 0.27.1, transformers 4.57.6,
  fastapi 0.141.1, uvicorn 0.53.0, timm 1.0.29, einops 0.8.2. Веса `issai/Qolda` (8.1 GB)
  в кэше HF.
- Запуск: `bash tools/start_qolda.sh`, перезапуск: `bash tools/restart_qolda.sh`.
  Лог — `runs/servers/qolda.log`, готовность — через ~15–20 с.
- Сервер пилота (`qolda check/local_server`) больше не нужен. Этот сервер добавляет
  `max_num_tiles` и `enable_thinking` на запрос, `GET /kznav/stats`
  и `POST /kznav/reset_peak` (замер памяти, D-006).
- Объекты YCB (`ycb_1.2`, через `habitat_sim.utils.datasets_download --uids ycb`):
  конфиги ссылаются на `collision_meshes`, а папка в архиве называется `collison_meshes`.
  Исправление — симлинк `collision_meshes -> collison_meshes` в `~/kznav_data/objects/ycb/`.

### Не сделано / известно

- Docker (2026-11-08, D-089): образы `kznav-sim` и `kznav-speech`, `docker/README.md`.
  Linux-сервер с NVIDIA EGL поддержан (`KZNAV_GPU_DEVICE_ID=0`, без переменных Mesa), но
  на этом ПК проверить его нельзя.
- Датасеты HM3D ещё не скачаны (ждём доступ Matterport). Их место —
  `$KZNAV_DATA_ROOT` (по умолчанию `~/kznav_data` на ext4 внутри WSL: GLB через
  `/mnt/c` читаются медленно).
- На диске C: 71 GB свободно; HM3D val + эпизоды + веса оцениваются в 45–60 GB
  (оценка, не замер).
