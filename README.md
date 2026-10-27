# kznav — zero-shot object navigation for a household robot commanded in Kazakh

kznav is a research bench for **zero-shot object-goal navigation (ObjectNav)** in which a household robot receives a
natural-language command in **Kazakh**, **mixed Kazakh–Russian** or **English** — e.g. *«Төсек қайда? Шаршап
тұрмын»* ("Where is the bed? I'm tired") — and must find the object in an unseen home and stop next to it. The robot
runs in [Habitat-Sim](https://github.com/facebookresearch/habitat-sim) on
[HM3D](https://aihabitat.org/datasets/hm3d/) scans of real apartments. Its language side is a swappable
vision-language model (VLM), so a native Kazakh model can be compared with general multilingual models of the same
size under identical conditions.

**Research question.** *Where and why does zero-shot object-goal navigation fail when a household robot is commanded
in Kazakh, and does a native Kazakh vision-language model close the gap?*

| | hypothesis |
|---|---|
| **H1** | The loss caused by Kazakh commands (vs English) arises at the language→vision interface — query generation, detection, verification — not in navigation. |
| **H2** | The Kazakh–English gap is larger for culturally specific objects (қазан, сандық, текемет, …) than for matched universal objects, and translate-first pipelines widen it. |
| **H3** | A native Kazakh VLM narrows the Kazakh gap compared with a general VLM of the same size, and is not better than it in English. |
| **H4** | VLM verification before stopping raises success by preventing false stops; it needs calibration. |

The pre-registered analysis plan is [`docs/analysis_plan.md`](docs/analysis_plan.md).

## Status

- **All numbers so far come from development data** (HM3D *train* scenes). The test split (HM3D ObjectNav v1 *val*)
  has not been touched; it will be run once the analysis plan is frozen (git tag `analysis-plan-v1`).
- The analysis plan is a **draft** (v2); its open decisions are listed in §14 of the plan.
- Kazakh test commands are being written and reviewed by native speakers; the Kazakh lines of the prompts and the
  lexicon still await native review.

## How the robot works

One episode, the same for every model and both robots:

```
command (kk / kk-ru / en)
   │
   ▼
[parse]      VLM, once per episode (prompt parse_goal v4):
             target, its English name (target_en), 3–5 weighted English visual hypotheses, likely rooms
   ▼
[look]       360° turn on the spot
   ▼
┌─► [map]        depth → top-down occupancy grid; frontiers between explored free space and the unknown
│   [detect]     MM-GroundingDINO (base, threshold 0.50) on every hypothesis separately
│   [segment]    SAM 2.1 masks → 3-D points → object candidates on the map
│   [fuse]       multi-frame confidence fusion (ApexNav-style); a candidate is approached at ≥ 0.45
│   [value]      SigLIP 2 similarity of the view to the hypotheses, projected on the map → frontier choice
│   [move]       grid planner on the robot's own map → pure-pursuit controller (robot-vacuum) or discrete steps
│        │
│        ▼  at a candidate ("last metre")
│   [verify]     the same VLM: "Is the main object in the image a "<target_en>"?"
│                P(yes) from first-token log-probabilities → Platt calibration (fitted on dev) → threshold
│        ├─ above the threshold → drive to where the object is in view (≤ 0.9 m) → STOP
│        ├─ below 0.25          → the candidate is dropped
└────────┴─ in between          → move closer and ask again (at most 3 times)
```

**Success** follows habitat-lab: the robot itself calls *stop* within 0.1 m (geodesic) of a goal view point.

### Robots (embodiments)

| | robot-vacuum — main table (b) | standard Habitat agent — table (a) |
|---|---|---|
| camera | RGB-D 640×480, **0.15 m** high, 20° up, HFOV 69° | RGB-D 640×480, **0.88 m** high, HFOV 79° |
| motion | wheeled, kinematic on its own navmesh, recovery when blocked | 0.25 m steps, 30° turns |
| step budget | 800 decisions | 500 steps (standard) |
| goal view points | recomputed for its camera | from the dataset |

The standard agent makes the results comparable with the literature (e.g. VLFM); the robot-vacuum is the realistic,
harder case.

### Models

| role | model | notes |
|---|---|---|
| VLM, native Kazakh | [Qolda-4B](https://huggingface.co/issai/Qolda) (ISSAI) | InternViT + Qwen3-4B; Apache 2.0 |
| VLM, general | [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) | Apache 2.0 |
| VLM, general, another family | [Gemma 3 4B](https://huggingface.co/google/gemma-3-4b-it) | Gemma Terms of Use; added in D-083 |
| speech (spoken commands) | [Qolda-AVL-5B](https://huggingface.co/issai/Qolda-AVL-5B) (ISSAI) | audio-vision-language; Apache 2.0 |
| translator (baseline) | [NLLB-200 distilled 600M](https://huggingface.co/facebook/nllb-200-distilled-600M) | CC-BY-NC 4.0 |
| open-vocabulary detector | [MM-GroundingDINO base](https://huggingface.co/openmmlab-community/mm_grounding_dino_base_o365v1_goldg_v3det) | |
| segmenter | [SAM 2.1 tiny](https://huggingface.co/facebook/sam2.1-hiera-tiny) | |
| value map, light verifier | [SigLIP 2 base](https://huggingface.co/google/siglip2-base-patch16-224) | |

All VLMs run locally in bf16 on one 12 GB GPU, one at a time, behind the same OpenAI-compatible server API
(`servers/`). Every model call — request, raw answer, retries and input image — is logged, so any run can be
recomputed without the model (`backend=replay`).

## Repository layout

```
kznav/                 the package
  language/            VLM backend interface and registry; backends: qolda, qwen_vl, openai_compat (Gemma),
                       qolda_avl (speech), mock, replay, parse_replay; prompts, schemas, lexicon, commands, speech
  perception/          occupancy map, frontiers, object candidates, detector / segmenter / value-encoder interfaces
  policy/              ObjectNav policy, belief over hypotheses, verify-before-stop, calibration, planner, controller
  sim/                 Habitat wrapper for both embodiments, robot model, episodes and view points, noise, validation
  run/                 episode runner, benchmark runner, logging, manifests, config composition
  eval/                metrics, failure stages, statistics (paired and scene-clustered), paper tables
  report/              figures and videos
configs/               Hydra config groups: backend, embodiment, dataset, detector, segmenter, policy, goal, eval, ...
prompts/               versioned prompt files; the SHA-256 of each prompt is recorded with every model call
lexicon/               accepted object names per language (draft, awaiting native review)
commands/              natural command bank (native-written; dev and test), recording sheets for spoken commands
calib/                 verifier calibration artifacts fitted on dev runs (with their dev episode ids)
splits/                frozen episode subsets (dev sets; test primary 547 / secondary 297; table (a) 300)
paper/matrices/        experiment matrices (arms × seeds) run by tools/run_matrix.py
servers/               local OpenAI-compatible model servers (Qolda, Qwen3-VL, Gemma)
tools/                 command-line entry points: runs, calibration, splits, analysis, diagnostics, downloads
tests/                 pytest suite (no GPU or model server needed for the core)
docs/                  decisions log, status, analysis plan, measurement protocol, code review, validation reports
reports/               generators of the progress reports and their outputs
env/                   exact composition of the Habitat environment (conda export, pip freeze)
```

## Installation

The bench was developed on Windows 11 + WSL2 (Ubuntu 22.04) with an RTX 5070 (12 GB). Three Python environments
are used:

| environment | contents | used for |
|---|---|---|
| conda `habitat_env` (Python 3.9) | habitat-sim 0.2.4 (with Bullet, headless), habitat-lab 0.2.4, torch, transformers 4.57.6, the perception models | the bench: simulation, perception, runs, tests |
| venv `~/qolda-env` (Python 3.13) | torch, transformers 4.57.6, FastAPI | the model servers (Qolda, Qwen3-VL, Gemma) |
| venv `~/avl-env` (Python 3.13) | torch, transformers 5.2 | Qolda-AVL (its remote code needs transformers 5) |

The exact `habitat_env` is in [`env/`](env/); installation issues and their fixes are in
[`INSTALL_LOG.md`](INSTALL_LOG.md) (in Russian). `tools/wsl.sh` activates `habitat_env` and sets the rendering
variables (under WSL2 Habitat renders through Mesa D3D12; there is no NVIDIA EGL).

**Data.** HM3D v0.1 scenes with semantics and the ObjectNav v1 episodes are downloaded with
`tools/download_hm3d.sh` (`minival`, `val`, `train`) into `$KZNAV_DATA_ROOT` (default `~/kznav_data`). The download
needs a Matterport API token, read from the environment or `~/.config/kznav/matterport.env`; it is never stored in
the repository. Gemma is gated on Hugging Face: accept its license and log in (`huggingface-cli login`) before the
first server start.

## Quick start

All commands run inside WSL from the repository root. From Windows Git Bash, prefix them with
`MSYS_NO_PATHCONV=1 wsl -d Ubuntu-22.04 --cd <repo path in WSL> --`.

```bash
# tests: no GPU and no model server needed (server tests run in ~/qolda-env)
bash tools/wsl.sh python -m pytest -q

# check that Habitat renders on the NVIDIA GPU
bash tools/wsl.sh python tools/check_render.py --out /tmp/render_check.png

# debug episodes on a test scene with placed objects (oracle detector by default; not paper results)
bash tools/wsl.sh python tools/run_debug_episodes.py --videos 2

# a model server (one at a time on a 12 GB GPU); run_matrix starts and stops them by itself
bash tools/start_qolda.sh      # port 23333;  tools/start_qwen.sh: 23334;  tools/start_gemma.sh: 23335

# an experiment matrix: show the cells and their state, then run (resumes unfinished cells)
bash tools/wsl.sh python tools/run_matrix.py plan paper/matrices/dev_confirm.yaml
bash tools/wsl.sh python tools/run_matrix.py run  paper/matrices/dev_confirm.yaml --only qolda_kk
```

Every component is chosen by one Hydra override, e.g. `backend=qolda | qwen3vl_4b | gemma3_4b | mock | replay`,
`embodiment=wheeled_vacuum_kinematic | habitat_agent`, `goal=multi | top1 | mt_word | mt_command`,
`policy.verification.backend=vlm | siglip_zero`. A new model is one file in `configs/backend/` plus, if needed, a
registered backend class; the navigation code does not change.

## Reproducing the experiments

- **Matrices.** Each experiment is a YAML file in `paper/matrices/` (arms × seeds, common overrides, model server per
  arm). `tools/run_matrix.py run` executes the cells with `tools/run_eval.py`, writes
  `runs/matrix/<matrix>/<arm>/seed<k>/{episodes.csv, events.jsonl, vlm_calls.jsonl, summary.md, manifest.json}` and
  resumes from the last finished episode after an interruption.
- **Tables.** `tools/run_matrix.py spec` writes a table spec; `tools/make_paper_tables.py` produces the tables and
  figures (paired contrasts, failure stages, the H1 gap decomposition, sensitivity analyses).
- **Replay.** `backend=replay backend.replay.sources=[<run>/vlm_calls.jsonl]` recomputes navigation and metrics from
  recorded model calls. `++backend.parse_replay` replays only the command parse while verification stays live
  (spoken commands, D-078).
- **Calibration.** `tools/fit_calibration.py` fits the verifier calibration and stop threshold on dev runs; the runner
  refuses an artifact that shares episodes with the evaluated set.
- **Decisions.** Every choice that affects a number — with the rule written *before* the run and the result after it
  — is an entry D-001 … D-084 in [`docs/decisions.md`](docs/decisions.md) (in Russian).

## Measurement

- **Metrics:** success rate (SR), SPL, SoftSPL, distance to success; definitions in
  [`docs/measurement_protocol.md`](docs/measurement_protocol.md).
- **Failure stages:** every failed episode gets exactly one stage — did not understand the command, goal never seen,
  not detected, verifier rejected the goal, false stop, not reached, stuck, stopped too far — grouped into
  language / grounding / navigation (`kznav/eval/failures.py`).
- **Statistics:** all arms run the same episodes; the scene is the unit of inference (scene-cluster bootstrap CIs,
  scene sign-flip tests, Holm correction over the primary contrasts), with exact McNemar tests as a secondary check.

## Results so far (development data only)

Fresh dev scenes not used for any tuning; 95% confidence intervals are about ±8–11 points. These are **not** test
results.

| setting | Qolda, kk | Qwen3-VL, kk | Gemma 3, kk | no VLM (VLFM-style) | ideal detector |
|---|---|---|---|---|---|
| standard agent, 150 episodes | **44.0%** | 40.7% (en) | 38.7% | 30.0% | 71.3% |
| robot-vacuum, 72 episodes (dev3) | **23.6%** | 15.3% | 19.4% | 20.8% | 75.0% |

- On the standard agent the method is significantly better than the no-VLM baseline (Qolda kk vs baseline: 38 vs 17
  discordant episodes, exact McNemar p = 0.006).
- Command understanding (robot-vacuum, 72 fresh episodes): episodes lost because the Kazakh command was not
  understood — Qolda 0, Gemma 8, Qwen 11, translate-first NLLB → Qwen 18.
- On the robot-vacuum the main loss is false stops at similar-looking objects seen from 15 cm; better verification
  precision is the main open problem.
- Spoken Kazakh commands (pilot, one speaker): Qolda-AVL transcription followed by Qolda parsing reached 40.7% on the
  standard agent against 44.0% for typed commands (difference not significant).

The full history, including the changes that did not survive confirmation on fresh scenes, is in
[`docs/decisions.md`](docs/decisions.md) and [`docs/STATUS.md`](docs/STATUS.md).

## Documentation

| file | contents |
|---|---|
| [`docs/analysis_plan.md`](docs/analysis_plan.md) | pre-registered analysis plan (draft v2) |
| [`docs/code_review_2026-10.md`](docs/code_review_2026-10.md) | code review: method, findings, verification |
| [`docs/measurement_protocol.md`](docs/measurement_protocol.md) | metric definitions (in Russian) |
| [`docs/decisions.md`](docs/decisions.md) | decisions log D-001 … D-084 with rules and results (in Russian) |
| [`docs/STATUS.md`](docs/STATUS.md) | current state and next steps (in Russian) |
| [`docs/command_authoring.md`](docs/command_authoring.md), [`docs/cultural_data_collection.md`](docs/cultural_data_collection.md) | guides for native-speaker command authors and for the cultural-object data (in Russian) |
| [`docs/validation/`](docs/validation/) | robot physics and navigation validation reports |

## Limitations

- Simulation only; the robot-vacuum is a model of a real device, not a hardware experiment.
- All current numbers are on development data with 72–150 episodes per arm.
- The Kazakh prompt lines and lexicon are drafts until reviewed by native speakers.
- Spoken-command results come from one speaker and synthetic voices; they are pilots.

## License

No license has been chosen yet; all rights reserved by the author until publication. Third-party models and
datasets keep their own terms (HM3D: Matterport's academic terms; NLLB-200: CC-BY-NC 4.0; Gemma: Gemma Terms of
Use).
