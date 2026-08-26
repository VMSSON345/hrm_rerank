# QP4 residual-coverage source-replay precommit v1

- Gate: `PASS`; đây là contract development-only, không sửa runtime và chưa chạy source replay/OCR.
- Cohort đã khóa: `6` anchor / `6` company / `6` PDF, `3` real document layout, `6` real table layout, `2` extraction family và `4` metric identity; overlap với diagnosis bằng `0`.
- Denominator bất biến: `90` primary observation = `60` target (`6 × 2 program × 5 surface`) + `30` semantic control; mỗi source có đúng `15` observation. Source error, timeout và abstention đều ở lại denominator; abstention không được tính ANSWER.
- Target gồm `30` ACTUAL/PLAN ratio và `30` ACTUAL/prior-ACTUAL reference-base mutation. Có `12` nhóm surface-equivalence, `30` paired reference-base comparison và `90` candidate-permutation observation.
- OCR policy là native-first và question-triggered: page rank dùng chính câu hỏi với `top_k=6`, mở rộng `±1` trang; chỉ các trang được chọn mới có quyền materialize. Cấm dùng gold physical page/provenance, cấm force OCR, cấm OCR toàn PDF/corpus và cấm tạo OCR cho trang ngoài selection.
- Artifact cache được phép chỉ là fingerprinted MinerU `content_list` và page-index state; không lưu bản OCR `.md`. Mọi observation phải replay lại với auto-generation tắt để chứng minh cache tái sử dụng được.
- Mỗi observation nhận đúng một earliest layer: `QUESTION_PLAN → SOURCE_LOAD_OR_EXTRACTION → CANDIDATE_POPULATION_OR_RETRIEVAL → SEMANTIC_BINDING_OR_PROGRAM_SYNTHESIS → SYMBOLIC_PROOF_OR_ACTION → PASS`.
- Residual chỉ đủ evidence generic khi đạt ít nhất `3 company + 3 source + 3 real document layout`; source/extraction residual còn phải phủ `2 extraction family`, layout residual phải phủ `3 real table layout`. Kết quả replay không tự động cho phép sửa runtime.
- Gate an toàn giữ `accepted_wrong=0`, old-winner/reference-base false accept `0`, semantic-control false accept `0`, hard-negative ANSWER `0`, permutation `90/90`, cùng frozen compiler `242/242`, regression `360/360`, execution/permutation `363/363` và counterfactual false accept `0/141`.
- QP4 vẫn chưa hoàn tất; QP5 và Phase D tiếp tục khóa.

**Bước duy nhất được phép tiếp theo:** `QP4-POST-QUESTION-SURFACE-POST-CACHE-RESIDUAL-COVERAGE-SOURCE-REPLAY` theo đúng contract này. Không được sửa runtime trong bước replay.
