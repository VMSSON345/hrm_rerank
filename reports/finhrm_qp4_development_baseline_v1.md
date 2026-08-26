# QP4 Development Cohort and Baseline v1

- Status: **BASELINE_COMPLETE_RUNTIME_UNCHANGED**
- Runtime changed: **no**
- Compiler canonical: `13/13`
- Ambiguity controls: `3/3`
- Compiler surface invariance: `73/91`
- Compiler counterfactual detection: `44/57`
- Source-backed anchors: `5` companies `5` / sources `5` / real layouts `3`
- Source-backed surface invariance: `20/35`
- Source-backed old-winner counterfactual false accepts: `4/15`
- Failed baseline gates: `['compiler_counterfactual_detection_rate', 'compiler_surface_semantic_invariance_rate', 'source_counterfactual_old_winner_false_accepts', 'source_surface_winner_invariance_rate']`
- Source dimensions not confirmed: `['COLUMN_ROLE', 'DIRECTION', 'REFERENCE_BASE']`
- Safe abstention is not counted as answered; synthetic questions are not layout confirmation.
- QP4 runtime, QP5 and Phase D remain locked.

## Generic diagnostic groups

- `compiler_surface:BILINGUAL_PARENTHETICAL`: failures `3`, layer `question_planner`, generic evidence `True`.
- `compiler_surface:DISCOURSE_PREFIX_SUFFIX`: failures `12`, layer `question_planner`, generic evidence `True`.
- `compiler_surface:IRRELEVANT_DISTRACTOR_CLAUSE`: failures `3`, layer `question_planner`, generic evidence `True`.
- `compiler_counterfactual:UNIT_SCALE`: failures `13`, layer `question_planner`, generic evidence `True`.
- `source_surface:CLAUSE_PERMUTATION`: failures `5`, layer `retrieval_or_binding_isolation_required`, generic evidence `True`.
- `source_surface:DISCOURSE_PREFIX_SUFFIX`: failures `5`, layer `question_planner_then_end_to_end_replay`, generic evidence `True`.
- `source_surface:SHORT_LONG_EQUIVALENT`: failures `5`, layer `retrieval_or_binding_isolation_required`, generic evidence `True`.
- `source_counterfactual:UNIT_SCALE`: failures `4`, layer `question_planner`, generic evidence `True`.

Next: Review the frozen QP4 baseline diagnosis. Build independently reviewed source-backed ACTUAL/PLAN ratio/direction anchors for COLUMN_ROLE, DIRECTION and REFERENCE_BASE; then pre-authorize only generic failure groups that meet the exact QP4 whitelist. No runtime change, QP5 or Phase D is authorized by this baseline.
