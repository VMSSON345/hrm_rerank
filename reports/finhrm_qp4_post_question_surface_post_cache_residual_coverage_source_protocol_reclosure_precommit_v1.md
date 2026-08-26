# QP4 residual-coverage source-protocol reclosure precommit v1

- Gate: `PASS`; contract-only precommit. Không OCR/source replay và không sửa runtime trong bước này.
- Generic evidence: `37` earliest source failures trên `4` company / `4` PDF / `3` real document layout / `4` table layout / `2` extraction family; thêm `4` source errors đứng sau question-plan failure và không được relabel.
- Code-path root: page-index load/build hiện xảy ra trước question-specific page ranking và có thể gọi MinerU cho mọi native-unusable page; selected-page fallback còn có thể tự đổi `mineru_auto_generate=false` thành `true`.
- Đây là generic protocol behavior dựa trên page-index state, selected mask và caller config; không dùng company, filename, page, year, value, anchor hay question literal.
- Exact runtime whitelist cho bước sau chỉ gồm `document_page_index.py`: `build_page_indexed_chunks` và `_materialize_selected_pages`. `document_reasoning_engine.py`, `document_sources.py` cùng toàn bộ semantic runtime tiếp tục freeze.
- Required behavior: native-only index trước ranking; MinerU/cache chỉ sau ranking trên đúng ranked-plus-neighbor mask; caller `false` không bao giờ bị promotion; không request/generate trang unselected; không whole-document/corpus OCR.
- Reclosure sẽ materialize on-demand cho đúng 90 frozen development questions khi caller bật generation, sau đó replay `90` primary + `90` permutation với generation tắt. Cache availability không phải semantic proof.
- Source-protocol gate yêu cầu source failures `0`, cache reusable `90/90`, pre-ranking/unselected OCR request `0`, generated unselected `0`; semantic answer metrics chỉ report, chưa phải gate của bước protocol.
- Frozen controls phải giữ compiler `242/242`, regression `360/360`, execution/permutation `363/363`, `141` counterfactual với false accept `0`, safe replay `PASS`.
- Runtime hiện chưa đổi, whitelist chưa được dùng. QP4 chưa hoàn tất; QP5 và Phase D vẫn khóa.
- Next allowed action: `QP4-POST-QUESTION-SURFACE-POST-CACHE-RESIDUAL-COVERAGE-SOURCE-PROTOCOL-RECLOSURE`.
