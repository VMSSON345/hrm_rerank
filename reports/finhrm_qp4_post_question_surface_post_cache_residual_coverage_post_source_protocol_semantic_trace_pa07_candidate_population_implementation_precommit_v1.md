# QP4 PA07 candidate-population implementation precommit

## Kết luận

Precommit **FAIL_CLOSED** và không mở runtime. Lý do không phải model nhỏ hay cap `64`: trên đúng ba row thuộc root đã khóa, candidate ACTUAL/PLAN đúng kỳ đã thiếu ngay cả khi diagnostic bỏ cả quota cục bộ lẫn quota cuối.

Không có thay đổi runtime, model, checkpoint, threshold hay OCR artifact. Không dùng ABW, protected split, gold candidate/program hoặc metric beautification.

## Bằng chứng theo tầng

Sáu PA07 development control được replay qua public PDF+question boundary; `6/6` merge output khớp chính xác input của predictor, mỗi query gọi predictor đúng một lần và source error bằng `0`.

Ba row exact root giữ đủ diversity ở mức triệu chứng: `3` company, `3` PDF, `2` document layout, `3` table layout, `2` extraction family. Candidate count qua bốn tầng lần lượt là:

- hierarchical Markdown header: `709 → 296 → 248 → 64`;
- flattened native compact header: `565 → 395 → 231 → 64`;
- fragmented native row label: `560 → 444 → 274 → 64`.

Ở cả bốn tầng của cả ba row, support ACTUAL+requested-period và PLAN+requested-period đều là `0`. Vì vậy local quota, dedup, final cap, rank và relevance không được coi là causal root; tăng `max_candidates` hoặc reserve theo score không phải fix hợp lệ.

## Vì sao chưa được sửa runtime

Broad candidate-population root phân rã thành ba structural subroot khác nhau:

1. multirow Markdown header giữ role và period ở hai level nhưng cell binding hiện ghép sai/thiếu;
2. flattened native header có explicit `TH/KH + year` nhưng structured plaintext-row path không materialize các monetary cell thành typed columns;
3. native text làm vỡ row label qua nhiều dòng, nên anchor page không tạo được complete metric+role+period cell.

Mỗi subroot hiện chỉ có `1` company, `1` source và `1` real table layout. Không có một minimum symbol whitelist chung được chứng minh. Gộp ba subroot rồi sửa nhiều symbol sẽ là tối ưu theo vài báo cáo, trái invariant đa dạng mà dự án đã khóa.

## Bước duy nhất tiếp theo

`QP4-POST-QUESTION-SURFACE-POST-CACHE-RESIDUAL-COVERAGE-POST-SOURCE-PROTOCOL-SEMANTIC-TRACE-PA07-CANDIDATE-POPULATION-ROOT-DECOMPOSITION-COHORT-PRECOMMIT`.

Bước này chỉ được xây cohort development-safe theo structural signature. Chỉ một subroot đạt độc lập tối thiểu `3` company, `3` PDF và `3` real table-layout instance mới được dùng để khóa exact runtime symbol và executable replay contract. QP5 và Phase D tiếp tục khóa.
