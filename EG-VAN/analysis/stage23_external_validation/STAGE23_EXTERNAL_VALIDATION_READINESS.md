# Stage 23 external evaluation readiness

**Overall decision: BLOCKED for the requested claim of *independent* PH2 validation.** PH2 is locally complete and the Stage 23 runner is technically **READY for a qualified external follow-up**, but prior PH2 project use means this cohort is not untouched. No Stage 23 PH2 inference or result has been generated. If the goal is a genuinely independent external-validation claim, obtain a new cohort before executing; the prepared PH2 command is for follow-up only.

| Gate | Result |
|---|---|
| Frozen Stage 23 checkpoint | PASS: SHA256 `42b018305694392ea874c1e9a4b28457adef780f7be9e740a16506404c9fc5ab`; epoch 14 registry; class order verified. |
| Frozen HAM split | PASS: SHA256 `f1c04c5ef5b54372c667daf0aebc29e6f24743c6c2cc094f16e2c4e679149883`. |
| PH2 metadata, mapping and source images | PASS: 200 unique metadata/manifest IDs; 80 common NV, 80 atypical excluded, 40 MEL; all 200 flat/original BMP pairs present, byte-matched and decodable. |
| Exact Stage 23 numerical path | PREPARED: Tesla T4 CUDA FP16, physical batch 16, selective FP32 q@k, frozen preprocessing, eval mode. No PH2 Stage 23 prediction exists for empirical exact-replay comparison. |
| Stage 23 training use of PH2 | No direct use evidenced: Stage 23 training manifest `ph2_accessed=false` and no PH2 loader in training source. Historical project-level PH2 influence cannot be disproved. |
| Patient independence | UNVERIFIED: only `IMD` case/image IDs are available; no patient linkage or cross-dataset patient map. |
| Untouched external cohort | FAIL: PH2 was previously used for diagnostics and older-model evaluations. |
| Local verification | `python -m py_compile` passed; four focused CPU tests passed; full read-only `--preflight` returned `PASS_QUALIFIED_EXTERNAL_FOLLOWUP`. |

The exact Colab commands and expected outputs are in [COLAB_STAGE23_PH2_INSTRUCTIONS.md](COLAB_STAGE23_PH2_INSTRUCTIONS.md). The `--run` command requires an explicit acknowledgment of prior PH2 use and refuses any existing output directory. It is **not** executed by this preparation task.

Next decision: either register a genuinely unused external cohort for independent validation, or explicitly proceed with the already frozen PH2 **external follow-up** while retaining the independence limitation. Neither route involves model development or test-driven threshold tuning.

## Pre-execution recheck after the follow-up decision

The user chose the **qualified PH2 follow-up**. A fresh local, read-only preflight again returned `PASS_QUALIFIED_EXTERNAL_FOLLOWUP`; protocol SHA256 remained `3e6e31165e6afcb6b270e940f3abfe94dc99e2c805d84e08db2a156a91754ed8`. The fixed 120 included / 80 atypical excluded manifest was therefore in place before any Stage 23 PH2 output. No `run/` or `.run_inprogress_*` directory existed locally. The Colab instructions now check the T4, Python dependencies, required input paths and empty output location before the single run. Syntax checks and all four focused CPU tests passed. The result summary also computes **mapped balanced accuracy** from the prespecified NV and MEL exact-label recalls; this is a reporting-only calculation and changes neither inference nor the frozen protocol. **Status for the authorized qualified follow-up: READY; independent-validation claim: BLOCKED.**
