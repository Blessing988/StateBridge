# Results

This file records the paper-safe synthetic-only results used in the StateBridge
manuscript.

| Method | S2 | BLEU-4 | METEOR | ROUGE-L | CIDEr | VQA Acc. |
|---|---:|---:|---:|---:|---:|---:|
| Direct VLM baseline | 55.3330 | 0.2470 | 0.4199 | 0.4331 | 0.7502 | 81.2930 |
| Fact-locked caption fusion | 55.4679 | 0.2438 | 0.4248 | 0.4446 | 0.7252 | 81.2930 |
| Role-phase gated fusion | 55.4659 | 0.2438 | 0.4247 | 0.4446 | 0.7244 | 81.2930 |

The paper reports `55.4679` as the final StateBridge score because it has the
best verified synthetic-only provenance among the final submissions.

Do not report exploratory submissions that used real train/validation labels,
pseudo-labels, or target-domain calibration as part of the paper method.
