# Stage 15 downloaded epoch-16 recovery audit

**Disposition: archival recovery complete.** The bounded recovery's epoch-14 through epoch-16 observable replay is exact, and every artifact listed in its manifest is now present locally with the expected SHA256. This is an audit of the bounded recovery, not a new experiment. No training or inference was run for this audit.

The downloaded Colab folder is nested at `experiments/stage15_single_candidate_exp1/stage15_single_candidate_exp1/`. Its `recovery_epoch16/` directory was audited in place. The original canonical `run/` artifacts were not copied, replaced, or modified.

## Provenance and replay

- Recovery manifest SHA256: `ca37278b89523f69b8713615a59b244fd35d737fea6675a2980941d10e9de661`.
- The canonical and downloaded source epoch-13 checkpoints both hash to `caec0fc13117b31e34b5760a8c7b618be1a39fcb634690eda21cbaf9e7a2c001`; the canonical and downloaded original history CSVs both hash to `2df7edd11a23658e56dc723ee57638476579cb3620d9c5d10039eefa07941fd0`.
- The archived original runner and bounded recovery runner match the manifest source hashes. The recovered checkpoint records the frozen configuration, selection-rule, and numerical-protocol hashes `3ad8c205e06cde8203a869be0a6cc9643ed0ca5b6beb553b57d53d4dce9d25f2`, `daf07a1bac3538b5a35ed59121bd5561aba4797a6889e852fd0ca07ad88be735`, and `f59010b7e2509c0bf1633b7126ff1d8d629afab39b515f57e95012fb78eb0196`.
- The manifest and standalone comparison log agree. For epochs 14, 15, and 16, they record exact history-row matches and byte-identical validation prediction CSV and metric JSON files against the persisted original run. The downloaded validation file hashes match the manifest. These comparisons include classwise metrics, confusion matrices, and eligibility. The recovered epoch-16 checkpoint's history is continuous from 1 through 16 and its rows equal the original CSV prefix.
- The recovered epoch-16 checkpoint and `best_checkpoint.pt` both record epoch 16, best epoch 16, and best validation loss `0.08387391282241738`. Their model tensors and optimizer states are equal. Both have finite model and optimizer tensors, a valid scheduler state consistent with the optimizer LR, GradScaler scale 131072, and saved RNG fields. Both embed validation metrics and predictions equal to the original epoch-16 artifacts.
- Original and recovered epoch-16 validation: accuracy `0.8194726166328601`, macro F1 `0.6611314670741572`, MEL recall `0.5887850467289719` (63/107), MEL F1 `0.6028708133971291`, NV recall `0.9396681749622926`, validation loss `0.08387391282241738`; eligible with no failed gate.

The recovered `epoch_016_checkpoint.pt` hashes to `1e37c8e8f6c75537a64aae1dab3bf715ff63bce840221d1457d02d5f98810f0f`. The recovered `best_checkpoint.pt` hashes to `85fcad4b184da16dd741f2d539a05f8a1f0c8d1d015298f4ce5aadf7ef4d16d5`. Both match the manifest. The missing original epoch-16 weight file means identity to the original in-memory epoch-16 weights is unprovable; exact observable replay is the supported conclusion.

## Download completeness

The initial local download omitted `epoch_014_checkpoint.pt` and `epoch_015_checkpoint.pt`. They were added later. The complete manifest artifact set is now present, and every listed artifact matches its manifest SHA256. The epoch-14 checkpoint hashes to `d6fb655f8a08f41e4e7579ab0a61d7d4540db9f122a831becb9ac975e7c121d9`; the epoch-15 checkpoint hashes to `8f95f6f67189b04a8c5bd6da0398932f77f9f5d7589e3a755ebbe662aa228828`.

Read-only checkpoint inspection confirms embedded epochs 14 and 15 and continuous histories 1–14 and 1–15, respectively. Both histories match the original persisted CSV exactly over their prefixes. All model parameters and buffers and all floating optimizer-state tensors are finite. Each scheduler records its corresponding epoch, a finite best value, and an LR matching its optimizer; both saved GradScaler states have valid positive scale 65536 and valid growth/backoff fields. Their embedded experiment, configuration, architecture, class order, rule, numerical protocol, and source/runner provenance fields equal those in the verified epoch-16 checkpoint. Their configuration, rule, and protocol hashes also match the manifest and the downloaded JSON files. Their latest saved validation metrics and predictions match the corresponding original epoch-14 and epoch-15 evidence.

No HAM test or PH2 data was accessed. The audit used the original validation evidence only. No scientific configuration, numerical policy, selection threshold, checkpoint, or training artifact was modified.

## Recheck after reported addition of epoch-14 and epoch-15 checkpoints

On 2026-10-05, the recovery directory was inventoried again; the epoch-14 and epoch-15 checkpoint files were still absent. On 2026-10-06, after the two files were placed in the same directory, the audit was rerun. Their hashes and embedded states passed the checks above, and the manifest is complete. This resolves the prior local-download gap. The original missing epoch-16 checkpoint remains unavailable, so byte-for-byte identity to its original weights remains unprovable; the exact observable replay and recovered selected checkpoint are verified.
