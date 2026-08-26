# FinHRM QP3-CORE generalization coverage implementation

- Overall gate: **PASS**
- Answers retained: `187/187`; new safe answers: `72`.
- Gold execution: `363/363`; permutation: `363/363`.
- Counterfactuals: `141`; false accepts: `0`.
- ADD grounding UNKNOWN: `67 -> 3`; frozen replay improvement `64` across `44` companies / `1` real layout.
- End-to-end ADD confirmation: `6` positive answers; `2` additional layout strata each satisfy the locked 3-company/3-PDF minimum; confirmed real layouts including frozen replay: `3`.
- Earlier candidate-level source-backed confirmations remain supporting-only and are not used to close the gate.
- Query metric UNKNOWN: `33 -> 25`; improvement `8` across `3` companies / `3` real layouts.
- Period UNKNOWN: `74 -> 74`; no inference from absent lineage.
- Failed gates: `[]`.
- No ABW confirmation, protected validation/blind/test access, fine-tuning, threshold/checkpoint change, synthetic confirmation, or QP4/Phase-D unlock.

The admitted implementation is safety-clean and the independently reviewed real-PDF replay exercises document extraction, retrieval, frozen neural proposal, symbolic binding, verification and execution end to end. Each of the two additional layout strata has three positive companies and three distinct PDF sources; residual failures abstain and are not counted as positive. Together with the frozen ADD improvement layout this closes the locked three-real-layout development coverage gate without an unseen or universal generalization claim.
