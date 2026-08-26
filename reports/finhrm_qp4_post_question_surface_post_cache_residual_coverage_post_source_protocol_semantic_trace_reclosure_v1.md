# QP4 semantic-trace reclosure — evaluator/oracle defect

## Kết quả

Bước `QP4-POST-QUESTION-SURFACE-POST-CACHE-RESIDUAL-COVERAGE-POST-SOURCE-PROTOCOL-SEMANTIC-TRACE-RECLOSURE` **không pass**. Outcome đúng contract là `EVALUATOR_OR_ORACLE_DEFECT`; không có runtime edit, model edit, threshold edit, checkpoint edit hay OCR mới.

Đây không phải cách diễn giải một failure runtime thành abstention pass. Trace execution dừng ở preflight vì oracle bắt buộc để chứng minh observer không can thiệp không tồn tại đầy đủ trong frozen parent.

## Lỗi contract được xác nhận

Precommit yêu cầu từng primary/permutation output phải khớp frozen parent trên program DAG, proof obligations và verification reasons. Nhưng trong đúng `42` row đã khóa:

- `42/42` primary và `42/42` permutation không có `program_dag`;
- `42/42` primary và `42/42` permutation không có `proof_obligations`;
- `42/42` primary và `42/42` permutation không có `verification_reasons`;
- `42/42` primary và `42/42` permutation không có `acceptance_basis`.

Parent chỉ đủ để đối chiếu source outcome, action, verification state, observed operation, candidate count và selected physical evidence. Theo missing-field policy đã khóa, không được suy hoặc điền lại các trường thiếu.

Candidate-permutation baseline cũng chỉ lưu output và con số `20/42`; không lưu exact transform/transform ID, deterministic order hoặc pointer-remapping semantics. Vì vậy không thể chứng minh một replay mới tái hiện đúng permutation đã khóa.

Do hai lỗi trên, chạy `42 + 42` trace vẫn không thể thỏa no-interference gate. Bước này ghi nhận `0` trace row đã chạy, thay vì tạo ra metric không kiểm chứng rồi báo pass.

## Có phải HRM quá yếu vì chỉ khoảng 800 nghìn tham số?

Chưa có evidence để kết luận như vậy. Checkpoint hiện tại có chính xác **843.097 tham số**. Tuy nhiên parameter count không chỉ ra layer gây lỗi trong pipeline neuro-symbolic.

Breakdown frozen của `42` row là:

- `18/42` lỗi ở candidate population/retrieval — trước khi HRM có thể chọn program;
- `11/42` lỗi ở symbolic proof/action — sau neural proposal;
- `5/42` lỗi ở semantic binding/program synthesis — vùng duy nhất hiện có thể chứa đóng góp của HRM, nhưng artifact chưa tách được neural proposal khỏi symbolic synthesis;
- `8/42` pass.

Vì vậy không thể quy `34` failure còn lại, hoặc toàn bộ `0 ANSWER`, cho kích thước model. Muốn xác nhận HRM là causal root phải chứng minh candidate cần thiết đã có, typed program đúng đã được tạo, symbolic proof đủ, rồi failure còn lại nhất quán ở neural operation/pointer proposal trên ít nhất ba company, ba source và nhiều real layout. Semantic trace được thiết kế để trả lời việc đó, nhưng oracle hiện thiếu nên chưa thể kết luận.

## Bước được phép tiếp theo

Chỉ mở `QP4-POST-QUESTION-SURFACE-POST-CACHE-RESIDUAL-COVERAGE-POST-SOURCE-PROTOCOL-SEMANTIC-TRACE-EVALUATOR-CONTRACT-RECLOSURE-PRECOMMIT` để:

1. khóa semantic envelope đầy đủ của baseline unobserved gồm DAG, obligations, reasons và acceptance basis;
2. khóa exact candidate-permutation transform cùng pointer remapping;
3. yêu cầu unobserved replay và observed identity-preserving replay khớp toàn bộ envelope trước khi attribution.

QP4 chưa hoàn tất. QP5 và Phase D vẫn khóa.
