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

Точный состав `habitat_env`: `env/habitat_env.conda.yml`, `env/habitat_env.pip-freeze.txt`.

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

### Не сделано / известно

- Docker не установлен (пользователь установит). Dockerfile для Linux-сервера с
  NVIDIA EGL — отдельная задача: там `gpu_device_id` должен быть ≥ 0, а переменные
  Mesa не нужны.
- Датасеты HM3D ещё не скачаны (ждём доступ Matterport). Их место —
  `$KZNAV_DATA_ROOT` (по умолчанию `~/kznav_data` на ext4 внутри WSL: GLB через
  `/mnt/c` читаются медленно).
- На диске C: 71 GB свободно; HM3D val + эпизоды + веса оцениваются в 45–60 GB
  (оценка, не замер).
