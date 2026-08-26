# FinHRM QP3-CORE ADD diversity confirmation

- Diversity gate: **NOT CLOSED**.
- Candidate-level positives: `10` across `5` companies / `2` layouts; role: `SUPPORTING_ONLY_NOT_GATE_CLOSING`.
- Independently reviewed positives: `0/10`.
- Pipeline stages: `{'real_pdf_source': True, 'extraction_or_table_reconstruction': False, 'typed_evidence_graph': True, 'retrieval_or_intent_proposal': False, 'symbolic_binding': False, 'deterministic_verification_and_execution': True}`.
- Failed gates: `['minimum_three_companies_in_every_additional_layout', 'minimum_three_source_pdfs_in_every_additional_layout', 'independent_review_complete_in_every_company_layout', 'all_required_end_to_end_stages_exercised', 'company_layout_disjoint_confirmation_available']`.

| Layout | Positives | Companies | PDFs | Reviewed company/PDF pairs |
|---|---:|---:|---:|---:|
| DEVELOPMENT_NATIVE_MULTI_TABLE | 6 | 3 | 3 | 0/3 |
| DEVELOPMENT_OCR_OR_HYBRID_TABLE | 4 | 2 | 2 | 0/2 |

The 10 source-backed positives remain valid supporting evidence that the frozen ADD proof contract behaves safely on materialized candidates. They do not close diverse system operation: OCR/hybrid has only two companies/sources, all positives remain pending independent review, and extraction, retrieval/intent proposal and symbolic binding were not replayed end to end.
