# kznav в Docker

Стенд в двух образах. Окружения собраны из точных списков пакетов этого ПК (`env/`); код, данные и веса моделей
в образ не входят, а подключаются с машины по тем же путям. Решение и проверка — `docs/decisions.md`, D-089.

| образ | что внутри | для чего |
|---|---|---|
| `kznav-sim` | `habitat_env` (Habitat-Sim 0.2.4, torch 2.8 cu128) + `/opt/qolda-env` (серверы Qolda, Qwen3-VL, Gemma 3) | все прогоны робота, тесты, `run_matrix` (сервер модели поднимается внутри контейнера) |
| `kznav-speech` | `/opt/avl-env` (Qolda-AVL-5B) + `/opt/nemo-env` (казахская NeMo ASR) | офлайн-шаг речи: расшифровка и разбор записей |

## Что нужно на машине

- NVIDIA GPU, **драйвер ≥ 580** (в окружениях серверов колёса CUDA 13), 12+ ГБ видеопамяти.
- Docker Engine с Compose. На Linux — ещё [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  На Windows — WSL2 и Docker внутри Ubuntu (`docker/install_docker_wsl.sh`), Toolkit не нужен.
- Диск: образы ≈ 30 ГБ, модели ≈ 40 ГБ, HM3D ≈ 16 ГБ (dev + val).
- Токены (в репозиторий не кладутся): Matterport (HM3D), Hugging Face с принятой лицензией Gemma 3.

## Первый запуск на новой машине

```bash
git clone <репозиторий> && cd robot_simulation
bash docker/kz.sh --build                 # образ kznav-sim, ~30–60 мин (интернет)
bash docker/kz.sh --build speech          # образ kznav-speech, если нужен шаг речи

# модели по закреплённым коммитам (env/models.yaml)
HF_TOKEN=hf_... bash docker/kz.sh python tools/fetch_models.py

# данные HM3D (токен Matterport в переменных окружения)
export MATTERPORT_TOKEN_ID=... MATTERPORT_TOKEN_SECRET=...
bash docker/kz.sh bash tools/download_hm3d.sh val
bash docker/kz.sh bash tools/download_hm3d.sh train

bash docker/kz.sh bash docker/selfcheck.sh   # GPU, рендер, модели, pytest -> SELFCHECK OK
```

Данные по умолчанию лежат в `~/kznav_data`, кэш моделей — в `~/.cache/huggingface`. Другие пути задаются переменными
`KZNAV_DATA_ROOT` и `HF_HOME` перед `docker/kz.sh`.

## Проверка воспроизводимости

Те же 14 dev-эпизодов, что на машине разработки. Нужны dev-сцены (`download_hm3d.sh train`). Около 15 минут, вся
видеокарта.

```bash
bash docker/kz.sh bash docker/check/run.sh newmachine
bash docker/kz.sh python docker/check/compare.py runs/docker_check/host runs/docker_check/newmachine
```

`runs/docker_check/host` берётся с машины разработки. На другой видеокарте и с другим путём рендера (EGL вместо D3D12)
совпадение до пикселя не ожидается; отличия записываются в итог D-089. Все плечи теста идут на одной машине.

## Обычная работа

```bash
bash docker/kz.sh python -m pytest -q
bash docker/kz.sh python tools/run_matrix.py run paper/matrices/<m>.yaml [--only arm1,arm2]
bash docker/kz.sh --speech /opt/nemo-env/bin/python tools/transcribe_kk_asr.py --model ... --out ...
bash docker/kz.sh --speech /opt/avl-env/bin/python tools/parse_speech_commands.py ...
bash docker/kz.sh                          # оболочка внутри контейнера
```

Результаты пишутся в `runs/` репозитория, как без Docker. Контейнер работает от имени текущего пользователя.
Остановить всё: `docker ps -q --filter label=com.docker.compose.project=kznav | xargs -r docker kill`.

## Как это устроено

- `docker/kz.sh` выбирает подключение видеокарты: при наличии `/dev/dxg` (WSL2) — `compose.wsl.yaml`, иначе —
  `compose.nvidia.yaml`.
- `docker/entrypoint.sh` выставляет рендер Habitat: в WSL2 — Mesa D3D12 на адаптере NVIDIA и
  `KZNAV_GPU_DEVICE_ID=-1`, на Linux — NVIDIA EGL и `KZNAV_GPU_DEVICE_ID=0`.
- `HF_HUB_OFFLINE=1` по умолчанию: модели берутся только из кэша, закреплённого `tools/fetch_models.py`.
- После изменения окружения на машине разработки: `bash docker/lock_envs.sh` (обновляет `env/`), затем пересобрать
  образ.
