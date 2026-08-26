# QP4 source/retrieval roadmap-review precommit v1

- Gate: `PASS`; runtime unchanged and whitelist empty.
- Generic evidence: `8` REFERENCE_BASE observations, `4` companies, `4` PDFs, `3` real layouts; first failure is source load/extraction.
- Decision: authorize development OCR-cache materialization with the frozen existing pipeline, followed by artifact-only replay. This does not authorize an OCR/layout/retrieval algorithm change.
- Replay must use no gold engine input, retain source failures in the denominator, keep accepted-wrong and false accepts at zero, and pass all frozen replay gates.
- QP4 remains incomplete; QP5 and Phase D remain locked.
