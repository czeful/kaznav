# Analysis plan (pre-registration)

**Status: DRAFT v2 (2026-10-24), not frozen.** It is frozen by tagging the commit `analysis-plan-v1` before the
first test run. After the freeze, every change is a dated deviation in §12 and in the paper.

Draft v2 brings the plan in line with the final system chosen on dev (D-064 … D-078). The changes from draft v1 are
listed in §13. Items still open before the freeze are marked **[to decide before freeze]** and collected in §14.

Implementation references are given as `file` or decision numbers in `docs/decisions.md` (D-xxx).
Numbers marked **[to fix on dev]** are set on dev data before the freeze, never on test data.

---

## 1. Question and hypotheses

*Where and why does zero-shot object-goal navigation fail when a household robot is commanded in Kazakh, and
does a native Kazakh vision-language model close the gap?*

The system is a modular VLFM-style pipeline in Habitat-Sim. The main embodiment is a wheeled robot-vacuum: camera at
0.15 m, its own navmesh, moving kinematically on that navmesh in the main tables (D-053). The pipeline:

- a VLM parses the command once per episode: the target, its plain English name (`target_en`) and 3–5 weighted
  English visual hypotheses (prompt `parse_goal` v4, D-071);
- MM-GroundingDINO detects the hypotheses, SAM 2.1 segments them, and detections are fused over frames into object
  candidates (D-058 … D-060);
- a SigLIP 2 value map of the hypotheses guides frontier exploration;
- before stopping, the same VLM verifies the candidate's crop with an English question about `target_en`; its answer
  is calibrated (D-042, D-069, D-071); the robot stops where the object is in view ("last metre", D-070/D-071).

The VLM is swappable: Qolda-4B (native Kazakh) or Qwen3-VL-4B (general, the same size).

- **H1 — failure locus.** The loss in success caused by Kazakh commands, relative to English commands to the same
  system, arises mainly at the language→vision interface (query generation, detection with the generated
  queries, verification), not in navigation.
- **H2 — cultural words.** The Kazakh–English gap is larger for culturally specific objects (қазан, сандық,
  текемет, …) than for matched universal objects, and translate-first pipelines widen it.
- **H3 — native model.** A native Kazakh VLM narrows the Kazakh gap compared with a general VLM of the same size:
  - its advantage is larger in Kazakh than in English;
  - in English it is not better than the general model.
- **H4 — verification.** VLM verification before stopping raises success by preventing false stops; it needs
  calibration, because a general model's raw confidence is over-confident.

The multi-hypothesis goal representation is an **ablation**, not a hypothesis. It was not supported on debug
data (D-038).

**Exploratory questions** (reported descriptively, not tested, no Holm family):

- **E1 — spoken commands** (D-077, D-078). How much success is lost when the command is spoken, parsed by the
  audio-language model Qolda-AVL either end-to-end or as a cascade (its transcript parsed by Qolda-4B)?
  **[to decide before freeze: in this paper or a separate one]**
- **E2 — camera height.** How the same system behaves on the robot-vacuum (0.15 m) and on the standard Habitat agent
  (0.88 m): stage tables, false-stop rates, the share of false candidates met before the goal (D-073, D-076).
- **E3 — light verifier.** A local SigLIP 2 verifier in place of the VLM verifier (D-074, D-075).
  **[to decide before freeze: include or not]**

## 2. Data

| study | data | what varies | role |
|---|---|---|---|
| main (H1, H3, H4) — table (b) | HM3D ObjectNav v1 **val**, 6 universal categories, 20 scenes; the robot-vacuum | language, VLM, verification | test |
| table (a) | 300 random val episodes (D-064) on the standard Habitat agent | language, VLM | comparability with the literature |
| cultural, navigation (H2) | HM3D scenes with inserted 3D assets: cultural objects **and matched universal objects inserted the same way** (option B) | word type × language | test |
| cultural, language/verification (H2) | photographs of cultural and matched universal objects (option C, D-055) | word type × language | test |
| spoken commands (E1) | recordings of the approved test commands by native speakers | input modality | test, exploratory |
| development | HM3D ObjectNav v1 **train** scenes + the dev commands (below) | everything tunable | dev only |

- **Matched controls (option B).** The cultural/universal comparison uses assets inserted in the same scenes by
  the same pipeline, so that insertion artefacts (lighting, scale, placement) cannot pose as a cultural-word
  effect. Pairs are fixed before the runs:
  - қазан ↔ cooking pot;
  - сандық ↔ chest / trunk;
  - текемет / сырмақ ↔ rug;
  - көрпе ↔ blanket;
  - домбыра ↔ guitar;
  - кесе ↔ cup;
  - further pairs as assets are found.

  Every inserted object gets a semantic id, so all funnel and stop labels work as in HM3D. The main table and
  table (a) run on unmodified HM3D scenes: no inserted object appears in them.
- **Episodes, table (b).** `tools/make_split.py test --runnable` drew a category-stratified subset of the val
  episodes **the robot-vacuum can run** at pitch 20° (seeded hash of episode ids; `tools/check_hm3d.py`, D-052,
  D-056, D-057):
  - primary subset: **547 episodes** (`splits/hm3d_v1_test_primary.txt`, `--per-category 100`; all 47 runnable
    plant episodes);
  - secondary subset: **297 episodes** (`splits/hm3d_v1_test_secondary.txt`, `--per-category 50`), nested in the
    primary one;
  - runnability is decided by geometry only (navmesh, view points, start, reachability). The perception and policy
    changes after D-057 do not change it, so the subsets stay as drawn;
  - why val episodes are not runnable (another floor, steps on the same floor, goal not visible from the robot's
    camera, start off the robot's navmesh) is reported per reason.
- **Episodes, table (a).** A simple random sample of 300 of all 2000 val episodes, without stratification and
  without the robot-vacuum's runnability filter (`tools/make_split.py test --random 300`, seed 0,
  `splits/hm3d_v1_table_a.txt`, D-064), so that the category and scene mix is comparable with published numbers on
  the whole val split. Tables (a) and (b) therefore do not share episodes and are never subtracted from each other.
- **Commands.** Human-written natural commands (D-048, `docs/command_authoring.md`):
  - at least 10 approved **test** commands per category, from at least 3 native authors;
  - each command written in kk, kk-ru and en with the same intent, and reviewed by a second native speaker;
  - an episode's command is chosen from its id, so it is the same command in every language and every arm;
  - dev commands (12, two per category) are used for prompt and threshold development only.

  The developer templates (`commands/hm3d_objectnav_v1.yaml`) are never used for reported runs.
- **Recordings (E1).** Each approved test command, in every language of the spoken arms, read by at least 3
  native speakers (`data/speech/<speaker>/<command id>__<lang>.wav`, any WAV, converted to 16 kHz mono;
  `commands/natural/speech_recording_sheet_*.csv`). Speakers are anonymised ids (S01, …), give consent, and the
  recordings stay outside git. An episode's recording is chosen from its id among the speakers, like its command.
  Synthetic voices (`tts_mms`) are a dev pilot only.
- **Development data** (all HM3D train scenes; none is a val scene):

  | set | scenes | episodes | used for |
  |---|---|---|---|
  | `hm3d_v1_dev` (`dev_pool`) | 10 | 150 (+ `dev_tune85`, 85) | tuning (D-054 … D-063), calibration, stages 1–3 |
  | `dev2` (`dev2_demo`) | 8 | 57 | stage A (D-068), calibration |
  | `dev3` | 10 fresh | 72 | confirmation of the final system (stage 4, D-073) |
  | `dev4` | 10 fresh | 72 | confirmation of the light verifier (D-075) |
  | `hm3d_v1_dev_agent` | (dev pool) | 150 | the standard agent (D-063, D-076, D-078) |
  | verifier data | 40 other train scenes | 16 065 crops | the trained verifier head (D-075, not adopted) |

## 3. Arms

### Shared settings (fixed on dev)

| part | setting | decision |
|---|---|---|
| detector | MM-GroundingDINO base, threshold 0.50, one caption per hypothesis, 640×480 input, boxes ≤ 80% of the frame | D-058, D-059, D-064 |
| segmenter | SAM 2.1-tiny | D-060 |
| fusion | multi-frame candidate fusion, a candidate is approached at fused confidence ≥ 0.45 | D-060 |
| value map | SigLIP 2 base over the weighted hypotheses; rooms not used | D-064, D-072 |
| goal parsing | prompt `parse_goal` **v4** (target, `target_en`, 3–5 hypotheses); all hypotheses, weights as parsed | D-028, D-071 |
| verification question | **English, about the model's own `target_en`** ("Is the main object in the image a …?"); prompt `verify_candidate` v1 | D-071 |
| verification rule | P(yes) from the yes/no logprobs → Platt calibration → max-F1 threshold; the first answer counts in full; below 0.25 the candidate is dropped; in between, a re-ask when the crop has doubled (at most 3 calls) | D-042, D-067, D-069 |
| last metre | commit to the approached candidate; ask from up to 1.5 m when the whole box is in the frame; stop within 0.9 m where ≥ 30% of the object is in the camera's vertical view | D-070, D-071 |
| stop of arms without a verifier | within 0.6 m of a recently detected candidate (the visible-stop rule hurt them on dev) | D-070 (1a) |
| ideal-detector arm | ground-truth semantic detector, box masks, no fusion, stop where the object is in view | D-064, D-070 (1a) |
| seeds, temperature | one seed; temperature 0 | D-036 |

**Embodiments.**

| | robot-vacuum — table (b), main | standard agent — table (a) |
|---|---|---|
| config | `wheeled_vacuum_kinematic` (D-053) | `habitat_agent` (D-063) |
| camera | RGB-D 640×480, 0.15 m, pitch 20° up (D-004), HFOV 69° | RGB-D 640×480, 0.88 m, HFOV 79° |
| motion | kinematic on its navmesh, pure-pursuit controller, recovery when blocked | discrete: 0.25 m forward, 30° turns; collisions marked on the map |
| budget | 800 decisions (D-005) | 500 steps (standard) |
| view points | recomputed for its camera (D-001) | from the dataset |

**Calibration.** One artifact per VLM and embodiment, `tools/fit_calibration.py`: content labels (D-042), Platt
scaling, max-F1 threshold, 5 folds by episode. The runner refuses an artifact fitted for another model, prompt or
verifier, or one that shares episodes with the evaluated set (D-041).

- Current dev artifacts: robot-vacuum `calib/{qolda,qwen3vl_4b}_dev18_en.json` (18 dev scenes); standard agent
  `calib/{qolda,qwen3vl_4b}_dev23_en.json` (dev2 + dev3 scenes, no overlap with `hm3d_v1_dev_agent`), both fitted
  on robot-vacuum crops.
- Final test artifacts: **[to decide before freeze]** — refit on all dev verification calls with the English
  question (dev pool, dev2, dev3, dev4), and whether the standard agent gets an artifact fitted on its own crops
  (on `hm3d_v1_dev_agent` the vacuum-fitted threshold rejects about one real target in four, D-076).

### Table (b): the robot-vacuum, `paper/matrices/hm3d_v1_main.yaml` (D-066)

| key | VLM | command | goal handling | verification | episodes |
|---|---|---|---|---|---|
| `qolda_kk` | Qolda-4B | kk | parse → hypotheses | English question, calibrated | 547 |
| `qolda_kkru` | Qolda-4B | kk-ru | parse → hypotheses | English question, calibrated | 547 |
| `qolda_en` | Qolda-4B | en | parse → hypotheses | English question, calibrated | 547 |
| `qwen_kk` | Qwen3-VL-4B | kk | parse → hypotheses | English question, calibrated | 547 |
| `qwen_kkru` | Qwen3-VL-4B | kk-ru | parse → hypotheses | English question, calibrated | 547 |
| `qwen_en` | Qwen3-VL-4B | en | parse → hypotheses | English question, calibrated | 547 |
| `mt_qwen_kk` | Qwen3-VL-4B | kk → NLLB-200 → en | translate-first, English pipeline | English question, calibrated | 547 |
| `mt_qwen_kkru` | Qwen3-VL-4B | kk-ru → NLLB-200 → en | translate-first, English pipeline | English question, calibrated | 547 |
| `qolda_kk_nv` | Qolda-4B | kk | parse → hypotheses | none (stop at 0.6 m) | 547 |
| `vlfm_en` | — | English category label | label → detector (VLFM-style) | none (stop at 0.6 m) | 547 |
| `oracle` | — | — | ground-truth detector (navigation ceiling) | — | 297 |
| `mt_kk` | Qolda-4B | kk → NLLB-200 → en | translate-first, within one model | English question, calibrated | 297 |
| `qwen_kk_nv` | Qwen3-VL-4B | kk | parse → hypotheses | none (stop at 0.6 m) | 297 |
| `qolda_kk_top1` | Qolda-4B | kk | top hypothesis only (ablation) | English question, calibrated | 297 |
| `vlfm_nllb_kk` | — | kk category name → NLLB-200 | translated label → detector | none (stop at 0.6 m) | 297 |
| `qolda_kk_cmdq` | Qolda-4B | kk | parse → hypotheses (prompt v2) | **question in the command language**, calibrated (`*_dev18.json`) | 297 |
| `qwen_kk_cmdq` | Qwen3-VL-4B | kk | parse → hypotheses (prompt v2) | question in the command language, calibrated | 297 |

- The `*_cmdq` arms are the secondary row "verification in the command language" (D-071: the English question is
  the architecture, the command-language question is reported beside it). They use prompt v2, which keeps the
  `target` field in the command's language.
- The arms without a verifier (`*_nv`) may replay the parse calls of their verified twin (`backend=replay`): the
  parse is deterministic and identical, and the replay frees the GPU for parallel runs.
- **[to decide before freeze]** a third VLM of the same size from another family (candidate: Gemma 3 4B), primary
  arms kk and en, after an offline check of its parses and verification answers on dev.
- **[to decide before freeze]** the light-verifier row (E3): `qolda_kk` with the SigLIP 2 verifier
  (`policy.verification.backend=siglip_zero`, `calib/siglip_zero_v1.json`), 297 episodes.

### Table (a): the standard agent, `paper/matrices/hm3d_v1_table_a.yaml` (D-062, D-064)

| key | what | episodes |
|---|---|---|
| `vlfm_en` | VLFM-style, our perception stack, English category label | 300 |
| `vlfm_closed` | VLFM-style with a closed-set COCO detector (RT-DETR 0.8) + SAM, no fusion — the literature point of reference | 300 |
| `qolda_kk` | the method, kk | 300 |
| `qolda_en` | the method, en | 300 |
| `qwen_kk` | Qwen3-VL-4B, kk | 300 |

**[to decide before freeze]** adding `qwen_en` (run on dev in stage 5, D-076) and `oracle`.

### Spoken commands (E1): `paper/matrices/hm3d_v1_speech.yaml` (to be written)

Everything as `qolda_kk` (the same verification by Qolda-4B, the same calibration); only the parse differs. The
parse comes from calls made before the robot runs (`++backend.parse_replay`, D-078), because Qolda-AVL (10.2 GB)
and the robot's perception do not fit on the GPU together.

| key | parse | episodes |
|---|---|---|
| `avl_kk_e2e` | Qolda-AVL-5B hears the recording and parses it (prompt `speech/parse_goal/v1`, no thinking) | 547 |
| `avl_kk_cascade` | Qolda-AVL-5B transcribes the recording; Qolda-4B parses the transcript (prompt v4) | 547 |

The dev rule of D-078 chooses which of the two also runs on the robot-vacuum on dev. **[to decide before freeze]**
which arm(s) and languages run on test, given the dev results (the robot-vacuum with synthetic voice is a pilot)
and the recordings available.

## 4. Outcomes

- **Primary:** success (habitat-lab definition):
  - the robot calls stop within 0.1 m (geodesic) of a goal view point;
  - for the robot-vacuum, view points are recomputed for its camera (D-001).
- **Secondary:**
  - SPL, SoftSPL, distance to success;
  - false-stop rate, stuck rate, collisions;
  - model calls and latency per episode;
  - verifier decisions on goal and non-goal crops (stop / reject / ask again) by content label.
- **Failure stage** (D-045, `kznav/eval/failures.py`): each failed episode gets exactly one stage, the first rule
  that matches in pipeline order:
  1. infrastructure;
  2. query (the parse fails or no query names the goal; lexicon in `lexicon/hm3d_objectnav.yaml`);
  3. false stop / stop too far / unlabeled stop;
  4. stuck;
  5. not seen;
  6. not detected;
  7. verifier rejected the goal;
  8. not reached.

  Groups:
  - *language*: query;
  - *grounding*: not detected, verifier reject, false stop;
  - *navigation*: not seen, not reached, stuck, stop too far.
- **Stop labels** (D-046): a stop counts as "at a goal object" if at least 30% of the stop candidate's latest
  detection box is goal pixels. The position-based label is recorded as a secondary label.
- **Verifier calibration** (H4): ECE and Brier score of the raw P(yes) on test verification calls, with content
  labels (D-042).
- **Spoken commands** (E1): per command and speaker, whether a hypothesis names the object; the transcript's
  character error rate (a diagnostic call, not part of the parse); parse latency.

## 5. Primary contrasts

The success difference is taken on the same episodes. These seven contrasts form the Holm family (m = 7, D-066).
Directional predictions are stated, but the tests are two-sided.

| # | contrast | hypothesis | prediction |
|---|---|---|---|
| C1 | `qolda_kk` − `qwen_kk` | H3 | > 0 |
| C2 | (`qolda_kk` − `qwen_kk`) − (`qolda_en` − `qwen_en`) | H3 (interaction) | > 0 |
| C3 | `qolda_kk` − `mt_qwen_kk` | native model vs translate-first (universal objects) | > 0 |
| C4 | `qolda_kk` − `qolda_kk_nv` | H4 | > 0 |
| C5 | `qolda_en` − `qolda_kk` | the gap that H1 decomposes | > 0 |
| C6 | `qolda_kkru` − `qwen_kkru` | H3, mixed kk-ru | > 0 |
| C7 | `qolda_kkru` − `mt_qwen_kkru` | native model vs translate-first, mixed kk-ru | > 0 |

The second part of H3 (in English the native model is not better) is checked descriptively with the 95% CI of
`qolda_en` − `qwen_en`. It is not tested for equivalence.

Dev evidence behind the predictions (not a result): on the fresh dev scenes the translator failed to understand
18–20 of 72 kk commands and Qwen 10–11 (mostly "төсек" → door), Qolda none (D-073, D-075); verification raised
Qolda kk from 6 to 17 of 72 (D-073).

**Secondary contrasts** (exploratory, unadjusted): `qolda_kk` − `mt_kk`; `qwen_kk` − `mt_qwen_kk`; `qwen_kk` −
`qwen_kk_nv`; `qolda_kk` − `qolda_kk_cmdq` and `qwen_kk` − `qwen_kk_cmdq` (English vs command-language
question); `qolda_kk` − `qolda_kk_top1`; table (a) `qolda_kk` − `qwen_kk`, `qolda_en` − `qolda_kk`, `qolda_kk` −
`vlfm_closed`; E1 `avl_kk_e2e` − `qolda_kk`, `avl_kk_cascade` − `qolda_kk`, `avl_kk_e2e` − `avl_kk_cascade`.

### H1 test

The C5 gap is decomposed into stage contributions:

- SR(en) − SR(kk) = Σ_stage [rate_kk(stage) − rate_en(stage)], which is exact because the stages partition the
  failures (`paper.gap_decomposition`).

H1 is **supported** if both conditions hold:

1. the *language + grounding* contribution is positive, with its 95% CI above 0;
2. it is more than half of the gap.

Both must hold under the pipeline order **and** under the navigation-first sensitivity order. In the latter,
stuck and goal-never-in-view take precedence over every language stage, so every failure that navigation alone
would explain is charged to navigation.

If only the pipeline order supports H1, the result is reported as **not robust**. The debug runs show that this
can happen: Qwen kk vs en, 71% vs 14% of the gap at the language stage under the two orders.

The same decomposition for `qwen_en` − `qwen_kk` is secondary.

### H2 tests

Cultural data, run after the main study with the same frozen pipeline:

- **Navigation (option B).** Interaction on success:
  [SR(en) − SR(kk)]\_cultural − [SR(en) − SR(kk)]\_universal-control, for Qolda and for Qwen, with scenes as the
  unit. Both word types are inserted in every scene, so the interaction is computed per scene and sign-flipped
  (§6). Translate-first amplification:
  [SR(`qolda_kk`) − SR(`mt_qwen_kk`)]\_cultural − [same]\_universal-control.
  **[to decide before freeze]** the embodiment of option B: the robot-vacuum (as table (b)), the standard agent
  (higher success, so a larger signal) or both.
- **Language and verification (option C).** Per item (photo + command):
  - query names the object (lexicon);
  - verifier accuracy;
  - verifier AUROC.

  Interaction as above with the object category as the unit: category-cluster bootstrap CI and category-level
  sign-flip test.
- The H2 family (2 VLMs × 2 interactions, plus the option-C interaction) gets its own Holm adjustment.

## 6. Statistical methods (D-047, `kznav/eval/stats.py`, `kznav/eval/paper.py`)

- **Unit of inference: the scene.** Episodes of one scene share layout and object instances. Seeds are averaged
  per episode first.
- **p-values:** two-sided sign-flip permutation test of the mean paired difference, flipping whole scenes:
  - exact for ≤ 16 scenes;
  - otherwise 20 000 Monte Carlo flips;
  - with 20 scenes the smallest attainable p is about 2·10⁻⁶.
- **Confidence intervals:** 95% percentile bootstrap resampling scenes (10 000 resamples) for:
  - arm means;
  - contrasts;
  - interactions;
  - stage contributions.
- **Multiplicity:** Holm over C1–C7 at α = 0.05. Everything outside the families of §5 is labelled
  exploratory and reported without adjustment.
- **Secondary tests:**
  - exact McNemar on pooled episode-seed pairs;
  - effect size d_z on episode differences.
- **Sensitivity analyses** (pre-specified):
  - navigation-first stage order (H1);
  - position-based stop labels instead of content labels;
  - command-cluster bootstrap (commands as the unit) for C1–C7, because each command serves many episodes;
  - excluding episodes with any model call that failed with invalid output.

## 7. Sample size (D-049, `tools/power_analysis.py`)

- **Pilot inputs** (debug runs, `tools/power_analysis.py estimate`):
  - reference SR ≈ 0.30;
  - discordance between arms ψ = 0.20–0.45 (median 0.30).
- **Assumptions:**
  - scene SD of SR 0.10;
  - scene SD of the effect 0.05;
  - 20 scenes;
  - α = 0.05/5 (Bonferroni bound on Holm; with m = 7 the bound is 0.05/7 and the power below is slightly
    optimistic).
- **Simulated power** of the primary test (`runs/power_psi030`, `runs/power_psi040`):

| Δ SR | n for 80% power, ψ = 0.30 | ψ = 0.40 | McNemar n, no clustering (ψ = 0.30) |
|---|---|---|---|
| 10 points | 600 | 800 | 348 |
| 15 points | 300 | 300 | 153 |
| 20 points | 150 | 150 | 85 |

- **Achieved sizes.** The subsets are drawn from the episodes the robot can run at pitch 20° (D-004, D-056), and
  val has only 47 runnable plant episodes, so the sizes are below the targets:
  - primary arms: **547 episodes**. Power at ψ = 0.30: 0.81 for 10 points, > 0.99 for 15 points; at ψ = 0.40: 0.69
    and 0.98;
  - secondary arms: **297 episodes**, nested. Power at ψ = 0.30: 0.51 for 10 points, 0.93 for 15 points.

  If compute forces a cut, every arm runs the 297-episode subset, and the paper states that the detectable effect
  is 15 points.
- **Seeds:** one seed for all test runs. Outcomes are almost deterministic per episode (temperature 0, D-036),
  so episodes buy more power than seeds.
- **Reproducibility check:** two arms (`qolda_kk`, `vlfm_en`) re-run on 50 test episodes with seed 1. The
  episode-level agreement is reported.
- **Compute budget**, re-measured on dev (2026-10-24): about 2.5 min per episode with a VLM server on the
  robot-vacuum (2.1–3.0), 2.3 min on the standard agent (2.0–2.6), 0.7 min without a VLM. Arms with a VLM run one
  at a time (one model fits on the 12 GB GPU next to the robot); arms without one run three in parallel.

  | part | arms × episodes | GPU-hours |
  |---|---|---|
  | table (b), primary arms with a VLM | 8 × 547 | ≈ 182 |
  | table (b), primary arms without a server (`qolda_kk_nv` replayed, `vlfm_en`) | 2 × 547 | ≈ 10 (parallel) |
  | table (b), secondary arms with a VLM (`mt_kk`, `qolda_kk_top1`, `*_cmdq`) | 4 × 297 | ≈ 50 |
  | table (b), secondary arms without a server | 3 × 297 | ≈ 4 (parallel) |
  | table (a) | 3 × 300 with a VLM, 2 × 300 without | ≈ 38 |
  | reproducibility check | 2 × 50 | ≈ 3 |
  | **main study** | | **≈ 290 (about 12 days on one RTX 5070)** |
  | spoken commands (E1), per arm | 1 × 547 | ≈ 23 |
  | robustness study (§11) | 5 × 300 | ≈ 60 |
  | each optional arm (third VLM, light verifier) | 1 × 547 or 297 | ≈ 23 or 12 |

  H2 is budgeted when its episodes exist.

## 8. Exclusions, failures, missing data

- **Exclusions** (D-041) are decided before an episode runs, from geometry only:
  - goal not in the semantic annotation;
  - no view points;
  - start off the robot's navmesh;
  - goal unreachable;
  - start at the goal.

  Exclusions are identical in every arm of an embodiment (the same episode list and the same view point cache)
  and are reported per reason.
- **Failures** (D-046) are exceptions during an episode, such as a model server down or a simulator error.
  - They are recorded in `failures.csv` and retried.
  - A model-server outage is never recorded as a model answer.
  - Episodes that still fail after `run.max_episode_attempts` are investigated, the cause is fixed, and the
    episodes are re-run in **all** arms.
  - Any failures left at analysis time drop the episode from every arm of the affected contrast. The count is
    reported in the contrast table (`dropped_failed`).
  - A run interrupted by a GPU driver reset or a power loss resumes from the last finished episode; finished
    episodes are kept (D-071, D-078).
- **Invalid model output** after retries (invalid JSON) is a result of the method and stays in the data. A failed
  parse is a query failure. In replayed parses (`*_nv`, E1), a command without a recorded parse is a query failure
  too, never a call to a live model.
- No other episode is removed. In particular, episodes are never removed for being long, odd, or contradicting a
  hypothesis.

## 9. Order of work and blinding

1. **Dev — done** (train scenes only; details in `docs/decisions.md`):
   1. camera pitch 20° (D-004); runnable dev and test episodes at that pitch (D-056, D-057);
   2. perception stack: detector thresholds, one caption per hypothesis, 640×480, SAM 2.1, fusion (D-054 … D-060,
      D-064); standard-agent executor (D-063); episode budget (D-005);
   3. calibration on dev with content labels (D-042, D-067); the first-answer fix (D-069);
   4. stage A on held-out dev scenes (D-068); plan of improvements D-070, stages 0–4: last metre, English question
      with `target_en` (prompt v4), strict threshold (rejected), search (rejected), confirmation on fresh scenes
      dev3 (D-070 … D-073);
   5. light verifier (D-074, D-075: not adopted); the standard agent with the final system (D-076);
   6. spoken commands with Qolda-AVL (D-077, D-078).
2. **Before the freeze — open:**
   1. the decisions marked **[to decide before freeze]** (§14);
   2. the final calibration artifacts (§3);
   3. bring `paper/matrices/hm3d_v1_main.yaml`, `hm3d_v1_table_a.yaml` and `hm3d_v1_robustness.yaml` in line with
      §3 (prompt v4, English question, last metre, calibration files, `*_cmdq` arms), write
      `hm3d_v1_speech.yaml` if E1 is in, and check every matrix with `tools/run_matrix.py plan`;
   4. approve the test commands (§2), and have a native speaker review the Kazakh lines of the prompts
      (`parse_goal` v4, `speech/parse_goal` v1) and the lexicon;
   5. collect the recordings if E1 is in.
3. Freeze: tag `analysis-plan-v1`; deposit this file (e.g. on OSF) with the tag's commit hash.
4. Run the full matrices (`tools/run_matrix.py run`). Results are not inspected per arm while they run, except
   `plan` and `summary.md` for failures and exclusions.
5. Analysis: `tools/run_matrix.py spec`, then `tools/make_paper_tables.py`, unchanged from the frozen commit.
   Any code change after the freeze is a deviation.

## 10. What is reported regardless of outcome

- Every primary contrast with its CI and its Holm-adjusted p.
- The H1 decomposition under both orders.
- The stage table of every arm.
- Exclusions and failures.
- Table (a) next to published VLFM numbers, with the differences listed:
  - natural-language commands parsed by a VLM instead of a category label, and a verification step;
  - SigLIP 2 instead of BLIP-2 for the value map;
  - MM-GroundingDINO base + SAM 2.1 + multi-frame fusion instead of VLFM's detectors (the `vlfm_closed` row is the
    closest to VLFM's);
  - a map-based planner and our executor instead of VLFM's learned PointNav policy.
- A summary of the dev history, including the changes that did not survive the fresh-scene confirmation (D-073:
  the improvements on the tuning scenes did not carry over) and the rejected ones (D-070 … D-075).
- Negative and null results stay in the paper.

## 11. Robustness study (exploratory; D-051, `paper/matrices/hm3d_v1_robustness.yaml`)

This study shows how much of the method's success survives conditions a real robot-vacuum has. It runs the
method arm (`qolda_kk`, settings of §3) on the 297-episode subset with:

- the articulated robot in Bullet instead of the kinematic base (contacts, wheel slip, steps and stairs of the
  scanned meshes - D-052 found every failure of the physical robot with a perfect detector to be mechanical);
- Redwood depth noise;
- actuation noise;
- wheel odometry instead of the ground-truth pose: dead reckoning from the known start, with Bullet's wheel slip
  plus calibration drift;
- all three together.

The noise parameters are fixed in the matrix file before the runs. Results are reported as SR differences from
the clean arm, with scene-cluster CIs. They are not tested, and they are not in a Holm family.

## 12. Deviations

None yet (the plan is not frozen).

## 13. Changes in draft v2 (2026-10-24) relative to draft v1

- §1, §3: the final system chosen on dev: MM-GroundingDINO base + SAM 2.1 + fusion (D-064), prompt `parse_goal` v4
  with `target_en`, the English verification question, the first-answer fix, the last metre (D-069 … D-071); the
  stop rule of arms without a verifier; embodiment table.
- §3: table (b) arms per D-066 (kk-ru and the translate-first arms with Qwen are primary; `mt_kk` secondary); the
  command-language verification row `*_cmdq` (D-071); table (a) arms and episodes per D-062 and D-064.
- §1, §2, §3, §4: exploratory questions E1–E3; spoken commands with Qolda-AVL (D-077, D-078).
- §2: development data sets dev2, dev3, dev4, the standard agent's dev set, the verifier data.
- §5, §6: the Holm family has seven contrasts (D-066); the predictions of C3 and C7 are directional because of the
  dev evidence.
- §7: compute budget re-measured on dev.
- §8: resumption after GPU resets and power loss; replayed parses.
- §9: dev work done; the steps left before the freeze.
- §10: the differences from VLFM updated; the dev history is reported.

## 14. Open decisions before the freeze

| # | decision | options | where it is discussed |
|---|---|---|---|
| 1 | "Ask from afar" on the robot-vacuum | keep it (2–3× fewer steps) or ask only within 0.6 m (precision of "stop" fell 0.45 → 0.28 on fresh scenes) | D-073 |
| 2 | Final calibration artifacts | refit on all dev scenes; a separate artifact for the standard agent fitted on its own crops | §3, D-076 |
| 3 | Spoken commands (E1) | in this paper or a separate one; which arm(s), languages and how many speakers | D-077, D-078 |
| 4 | Third VLM of the same size | Gemma 3 4B or none | §3 |
| 5 | Light-verifier row (E3) | include on 297 episodes or not | D-075 |
| 6 | Table (a) arms | add `qwen_en` and `oracle` or not | D-076 |
| 7 | Embodiment of H2 option B | robot-vacuum, standard agent or both | §5 |
