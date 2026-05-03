# Oracle Answer-Session Context Replay

This sanity check gives the reader only LongMemEval sessions marked as containing the answer.
It is not a deployable memory policy and should be reported only as a reader/scorer sanity check.

Wall-clock replay time: 16767.16 ms

| category | cases | correct | accuracy | total tokens | tokens/correct |
|---|---:|---:|---:|---:|---:|
| all | 64 | 27 | 0.421875 | 54172 | 2006.37 |
| knowledge-update | 13 | 6 | 0.461538 | 16185 | 2697.50 |
| multi-session | 31 | 9 | 0.290323 | 30198 | 3355.33 |
| single-session-preference | 8 | 2 | 0.250000 | 2012 | 1006.00 |
| single-session-user | 12 | 10 | 0.833333 | 5777 | 577.70 |

## Correct Examples

- `08e075c7`: gold='9 months', prediction='9 months'
- `10e09553`: gold='7', prediction='7'
- `2698e78f`: gold='every week', prediction='every week'
