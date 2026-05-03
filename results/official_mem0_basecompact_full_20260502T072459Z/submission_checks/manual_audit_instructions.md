# Manual Audit Instructions

Grade each prediction using the question, gold answer, and visible retrieved evidence.
Use `1` for correct, `0` for incorrect, and `?` only when the gold answer is ambiguous.
Do not give credit for unsupported answers unless the prediction matches the gold answer.
Fill `manual_official_mem0_correct`, `manual_odv2_correct`, and `manual_notes` in `manual_audit_sample.csv`.

Report two numbers in the paper if time permits:

- Manual agreement with the automatic span scorer.
- Any corrected accuracy difference between official Mem0 and ODV2 on this 50-case audit.
