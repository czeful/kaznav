# Code review — October 2026

Review of the whole repository before sharing it for external code review. **No finding below has been applied to
the code**: changing logic could change recorded results, so each fix is left to the project author. The review
was followed by a documentation-only pass (README, docstrings, comments); §5 shows that the pass left the program
logic unchanged.

## 1. Scope and method

| what | how |
|---|---|
| code | 162 tracked Python files, ~26,000 lines after the documentation pass: `kznav/` (package, 84 files), `tools/` (33), `servers/` (3), `reports/` (5), `tests/` (37) |
| static analysis | `ruff 0.17.3`, rule sets `E9` (syntax), `F` (pyflakes: undefined names, unused imports/variables), `B` (bugbear: likely bugs) |
| tests | full suite in `habitat_env`: 486 passed, 2 skipped (the two server tests need FastAPI and are run in the model-server environment: 7 passed) |
| manual review | the code that produces the paper's numbers: episode metrics, failure-stage attribution, the H1 gap decomposition, paired contrasts and statistics, verifier calibration and the stop rule, the verify-before-stop decision, replay of recorded model calls, the matrix runner's resume logic, episode subsets and command assignment |
| data integrity | all 120 finished run cells under `runs/matrix/`: no episode is recorded twice |
| repository hygiene | secrets, large files, hard-coded paths, broken links in Markdown |

## 2. Summary

- **No defect that affects reported numbers was found.** The metrics follow habitat-lab 0.2.4; the exact McNemar
  test, Wilson interval, Holm adjustment, scene-cluster bootstrap and scene sign-flip test are implemented
  correctly; calibration is fitted on dev data only and the runner refuses an artifact that shares episodes with
  the evaluated set; replayed model calls are keyed by the prompt's hash, so a changed prompt cannot reuse an old
  answer.
- Static analysis reports 37 items: 31 are cosmetic (unused imports, unused loop variables, an f-string without
  placeholders), 5 are false positives, and 1 is a type-annotation reference with no runtime effect.
- One statement in the analysis plan does not match the implementation (R1).

## 3. Findings

Severity: **major** — could change a reported number; **minor** — correctness of documentation, typing or dead code,
no effect on results; **cosmetic** — style only; **note** — design remark, no action required. Line numbers refer to
the code after the documentation pass.

| ID | severity | location | finding | recommendation |
|---|---|---|---|---|
| R1 | minor | `docs/analysis_plan.md` §6; `kznav/eval/stats.py` `cluster_sign_flip_test` | The plan states that with 20 scenes the smallest attainable p is about 2·10⁻⁶ (exact test, 2/2²⁰). The implementation is exact only up to `exact_max_clusters = 16` scenes and uses 20,000 Monte Carlo sign flips above that, whose smallest p is 1/20,001 ≈ 5·10⁻⁵. Conclusions at α = 0.05 / 7 are unaffected. | Either set `exact_max_clusters = 20` for the test runs (2²⁰ ≈ 10⁶ sign vectors, feasible) or correct the sentence in the plan before it is frozen. |
| R2 | minor | `kznav/sim/env.py:433` | `semantic_objects()` is annotated `Dict[int, "SemanticObject"]`, but `SemanticObject` is imported only inside the function body (ruff F821). With `from __future__ import annotations` the annotation is never evaluated, so there is no runtime effect; `typing.get_type_hints` and type checkers fail on it. | Import `SemanticObject` under `if TYPE_CHECKING:` at module level. |
| R3 | minor | `kznav/sim/validation.py:311–312` | `pushed` and `blocked` are computed and never used; the two checks below recompute the same conditions with force thresholds. Dead code. | Remove the two lines. |
| R4 | cosmetic | 25 places (list in §4) | Unused imports (F401), one f-string without placeholders (F541, `reports/make_confirm_2026-10.py:170`), three unused loop variables (B007). | `ruff check --select F401,F541 --fix`; rename unused loop variables to `_`. |
| R5 | note | `kznav/eval/paper.py:284`, `tools/diag_hypothesis_detection.py:293–294`, `tools/make_results_page.py:212` | ruff B023 (closure over a loop variable). In all three places the closure is called within the same iteration, so it sees the intended value. False positives. | None; optionally pass the value as a default argument to silence the warning. |
| R6 | note | `kznav/policy/calibration.py:138` | ruff F841: `lo2` is unused in the pool-adjacent-violators loop. Correct by design: a merged block keeps the left edge of the earlier block. | None; optionally `_lo2`. |
| R7 | note | `kznav/policy/calibration.py` `assert_disjoint_from` | Disjointness between calibration and evaluation data is checked per episode, not per scene. Test episodes (HM3D val) never share scenes with dev (HM3D train), so the test is safe by construction. For dev-to-dev evaluations the scene-disjoint calibration sets were chosen by hand (D-076, D-084). | Optionally also store and check the calibration scenes. |
| R8 | note | `reports/make_report_2026-10-10.py:25`; `reports/*.py` fonts | One report generator writes to an absolute Windows path; the PDF generators load fonts from `C:/Windows/Fonts`. They are one-off report scripts that run on the Windows host. | Make the output path relative to the repository if the script is reused. |
| R9 | cosmetic | `docs/decisions.md`, D-055 | The formula `[en − L](культурный)` is rendered by Markdown as a link with a non-existent target. | Wrap the formula in backticks. |
| R10 | note | environment | Three Python environments are used (Habitat: Python 3.9 + transformers 4.57.6; model servers: Python 3.13 + transformers 4.57.6; Qolda-AVL: transformers 5.2). `pyproject.toml` lists only the light core dependencies; the heavy ones are installed by hand (`INSTALL_LOG.md`). The Habitat environment is exported in `env/` (`habitat_env.conda.yml`, `habitat_env.pip-freeze.txt`); the two model-server environments are not. | Export the model-server environments (`pip freeze`) to `env/` as well before the paper's code release. |

## 4. Static analysis, full list

```
kznav/eval/paper.py:284:77: B023 Function definition does not bind loop variable `g`              (R5)
kznav/eval/photo_bench.py:16:41: F401 `typing.Optional` imported but unused
kznav/language/backends/openai_compat.py:10:46: F401 `typing.Optional` imported but unused
kznav/policy/calibration.py:138:25: F841 Local variable `lo2` is assigned to but never used       (R6)
kznav/policy/objectnav.py:21:25: F401 `typing.Dict` imported but unused
kznav/report/figures.py:8:20: F401 `typing.Dict` imported but unused
kznav/report/video.py:10:25: F401 `typing.List` imported but unused
kznav/run/benchmark.py:37:86: F401 `typing.Tuple` imported but unused
kznav/sim/arena.py:11:25: F401 `typing.List` imported but unused
kznav/sim/arena.py:11:31: F401 `typing.Optional` imported but unused
kznav/sim/env.py:433:46: F821 Undefined name `SemanticObject`                                     (R2)
kznav/sim/episodes.py:19:47: F401 `typing.Mapping` imported but unused
kznav/sim/episodes.py:19:56: F401 `typing.Optional` imported but unused
kznav/sim/episodes.py:19:76: F401 `typing.Tuple` imported but unused
kznav/sim/mock_env.py:14:46: F401 `typing.Optional` imported but unused
kznav/sim/validation.py:311:9: F841 Local variable `pushed` is assigned to but never used         (R3)
kznav/sim/validation.py:312:9: F841 Local variable `blocked` is assigned to but never used        (R3)
reports/make_confirm_2026-10.py:170:14: F541 f-string without any placeholders
reports/make_report_2026-10-10.py:7:44: F401 `reportlab.lib.enums.TA_LEFT` imported but unused
reports/make_report_2026-10-10.py:146:13: B007 Loop control variable `k` not used within loop body
reports/make_report_pdf_2026-10-17.py:181:13: B007 Loop control variable `k` not used within loop body
tests/test_attribution.py:3:8: F401 `csv` imported but unused
tests/test_attribution.py:19:31: F401 `kznav.run.logging.RunLogger` imported but unused
tests/test_attribution.py:22:28: F401 `test_benchmark.CHAIR` imported but unused
tests/test_backends.py:10:51: F401 `kznav.language.backends.openai_compat.OpenAICompatBackend` imported but unused
tests/test_failures_and_robot.py:2:8: F401 `math` imported but unused
tests/test_failures_and_robot.py:4:17: F401 `numpy` imported but unused
tests/test_last_meter.py:3:8: F401 `math` imported but unused
tests/test_stage3.py:11:29: F401 `test_last_meter.FakeBackend` imported but unused
tools/compare_runs.py:16:17: F401 `numpy` imported but unused
tools/diag_hypothesis_detection.py:25:8: F401 `math` imported but unused
tools/diag_hypothesis_detection.py:66:9: B007 Loop control variable `eid` not used within loop body
tools/diag_hypothesis_detection.py:293:54: B023 Function definition does not bind loop variable `res`(R5)
tools/diag_hypothesis_detection.py:294:54: B023 Function definition does not bind loop variable `res`(R5)
tools/diagnose_reachability.py:19:17: F401 `numpy` imported but unused
tools/make_results_page.py:212:101: B023 Function definition does not bind loop variable `n`      (R5)
tools/parse_speech_commands.py:24:8: F401 `re` imported but unused
```

Command: `ruff check --isolated --target-version py39 --select E9,F,B $(git ls-files '*.py')`.

## 5. Documentation pass: proof that the logic is unchanged

The pass added module, class and function docstrings (Google style, PEP 257) and one comment; it rewrote
`README.md` in English. No configuration, prompt, calibration, split, command or lexicon file was touched (their
hashes are part of each run's identity).

| check | before the pass | after the pass |
|---|---|---|
| syntax-tree fingerprint of each of the 162 files (SHA-256 of `ast.dump` with docstrings removed) | reference | identical in all 162 files |
| diff of the Python files | — | 1,535 lines added, 0 removed |
| documented modules / classes and functions | 153 / 478 of 1,825 | 162 / 1,053 of 1,825 (the rest are mostly `test_*` functions and private helpers) |
| ruff (`E9,F,B`) | 37 findings | the same 37 findings (line numbers shifted) |
| test suite, `habitat_env` | 486 passed, 2 skipped | 486 passed, 2 skipped |
| server tests, model-server environment | 7 passed | 7 passed |

The fingerprint compares the program as Python parses it, so any change to code (statements, expressions,
default values, decorators, imports) would have shown up; docstrings and comments are the only parts it ignores.
