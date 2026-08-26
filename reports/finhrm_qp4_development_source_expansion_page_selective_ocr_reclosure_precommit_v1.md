# QP4 development source expansion page-selective OCR reclosure precommit v1

- Gate: `PASS`; this freezes an executable OCR-acquisition contract only. No OCR, runtime, model, checkpoint, threshold, test or protected-split action was performed.
- Frozen scope: `25` image-dominant development PDFs / `25` companies / `721` physical pages from the unchanged parent sample; replacements and additional sources are `0`.
- Exact routing: `179` unique pages (`24.83%` of frozen physical pages), at most `8` per PDF, from a uniform raster-morphology transform plus four deterministic physical quantiles. Whole-document OCR sources: `0`.
- Replay identity: `FINHRM_QP4_DEV_IMAGE_DOMINANT_PAGE_SELECTIVE_OCR_V1`; selected manifest SHA-256 `ab463443df08a2d8a41dd064cc0a1f8281429b8ace0355b61fb54feb18199bf9`. Raster scores route OCR only and never establish roles, period, provenance, correctness or admission.
- Build admission remains strict: every admitted hierarchical/fragmented anchor needs physical-page visual review and explicit same-entity, same-period, two-role, operand, unit and cell provenance. Missing proof is `UNKNOWN -> ABSTAIN`; conflict is `REJECT`.
- Required coverage remains separate for hierarchical and fragmented: each must reach `3/3/3/3` companies/sources/real-layouts/instances. Flattened stays frozen and cannot be pooled. No result-driven page replacement or extra OCR is allowed in the same build.
- Precommit verification observed `363/363` execution, `363/363` permutation, `141` counterfactuals with `0` false accepts, and `360/360` current affected regression; the build must retain all of them.
- QP4 is not complete; QP5 and Phase D remain locked. The only authorized next action is `QP4-DEVELOPMENT-SOURCE-EXPANSION-PAGE-SELECTIVE-OCR-COHORT-BUILD`.
