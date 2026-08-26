# FinHRM QP3-CORE post-implementation replay

- Implementation: **COMPLETE**.
- QP3 core runtime oracle: **122/122**, false accepts **0**, gold retained **26/26**.
- Frozen QP3A–QP3F plus CORE tests: **66/66**.
- Compiler/proof/generalization/company-disjoint tests: **145/145**.
- Document runtime tests: **181/181**.
- New-company development direction audit: **19/19** direction-supported across **19** company/source keys; all verifier states remain `UNKNOWN`, so none is counted as proof-complete confirmation or answered coverage.
- Original frozen S4: **NOT RUN** because its audit opens `validation_candidates.jsonl`, conflicting with the explicit no-validation constraint.
- Validation-free S4 stress replacement: **PASS** on 363 SELECT/ADD examples, 121 company groups and 6 layout families; permutation `363/363`, 123 explicit counterfactuals and false accepts `0`.
- Proof-complete gold coverage is only `187/363`; company/layout worst-group answer rate is `0`. This limitation is explicit and prevents an unseen or universal-generalization claim.
- QP4 remains locked. No ABW confirmation, protected evaluation access, fine-tuning, threshold/checkpoint change, or Phase-D unlock occurred.

The synthetic QP3 contract and development/train S4 stress replay are not unseen evidence or a claim of universal PDF coverage.
