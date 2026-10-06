# Stage 16 descriptive error analysis

Counts below were recomputed from `final_ham_test_predictions.csv` and `final_ph2_predictions.csv`; no threshold was altered and no image was re-evaluated.

## HAM10000 test

| Transition | Count | Fraction of true class | Fraction of errors in true class |
|---|---:|---:|---:|
| MEL → NV | 41 | 41/107 = 38.32% | 41/51 = 80.39% |
| MEL → BKL | 5 | 5/107 = 4.67% | 5/51 = 9.80% |
| BKL → MEL | 13 | 13/104 = 12.50% | 13/46 = 28.26% |
| NV → MEL | 18 | 18/676 = 2.66% | 18/40 = 45.00% |
| AKIEC → BKL | 7 | 7/40 = 17.50% | 7/16 = 43.75% |
| BCC → NV | 8 | 8/58 = 13.79% | 8/18 = 44.44% |

Of 107 true melanomas, 56 were identified as MEL and 51 were missed. The dominant melanoma error was MEL→NV: 41 cases, 80.39% of melanoma false negatives and 38.32% of all test melanomas. The remaining misses were BKL 5, AKIEC 2, VASC 2, and BCC 1. This confusion limits the clinical interpretation of the otherwise high overall accuracy.

## PH² external follow-up

Among 40 true melanomas, 11 (27.5%) were predicted MEL, 18 (45.0%) NV, and 11 (27.5%) OTHER. All 11 MEL→OTHER predictions were BKL in the original seven-class output. For 80 true common nevi, 69 were predicted NV, five MEL, and six OTHER (all BKL). These observations describe limited transfer under the frozen mapped PH² protocol; the data do not establish a specific causal mechanism for the shift. PH² had prior project use, and atypical nevi were excluded by protocol.
