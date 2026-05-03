# Manual Audit Summary

Reviewed audit file: `manual_audit_sample_reviewed.csv`.

The reviewed 50-case audit marks official Mem0 correct on 14/50 cases and ODV2 compact correct on 11/50 cases, for a manual ODV2-minus-Mem0 delta of -3 cases. The automatic scorer on the same rows marked official Mem0 correct on 13/50 and ODV2 compact correct on 9/50, for an automatic delta of -4 cases. Manual and automatic judgments agree on 91/100 policy-case decisions (91.0%); agreements are 45/50 for official Mem0 and 46/50 for ODV2 compact.

Case `2318644b` (Hawaii versus Tokyo accommodations) is a retrieval/evidence-availability failure: the original dataset contains the Maui resort and Tokyo hostel costs needed for the `$270` gold answer, but the retrieved reader evidence shown in the audit omits the Maui cost, so both policies are manually marked incorrect.
