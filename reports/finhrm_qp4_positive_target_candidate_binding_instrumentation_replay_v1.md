# QP4 positive-target Candidate/Binding instrumentation replay

Kết luận: **FAIL / HOLD**. Không mở runtime repair.

## Execution

- Đã chạy đúng `5/5` baseline + `5/5` instrumented; không thêm, thay hoặc fallback case.
- Output semantic envelope không đổi `5/5`; decision không đổi `5/5`; baseline khớp frozen compact `5/5`.
- Runtime, checkpoint và MinerU manifest giữ nguyên.

## Page-index hashes

- `original_before`: `df8be07147d790b7ef18a917aca8f3d3c28ca7923b959a78c2dc721a11284fd7`
- `original_after`:  `df8be07147d790b7ef18a917aca8f3d3c28ca7923b959a78c2dc721a11284fd7`
- `copy_before`:     `df8be07147d790b7ef18a917aca8f3d3c28ca7923b959a78c2dc721a11284fd7`
- `copy_after`:      `ea821b094c6fd52ef0195d460863732f38ee74ab905dbcc77c97b3632f9e1798`

Original byte-identical. Copy đổi physical SQLite header nhưng logical digest giữ `a3be831c…d7b5` trước/sau.

## Vì sao HOLD

Harness đã emit đủ 5 trace nhưng terminal transport cắt payload dài sau khi inference hoàn tất; process đã đóng nên buffer không thể phục hồi. Không chạy lại vì sẽ vượt denominator đã khóa `5+5`. Do đó Candidate không có complete trace để audit cùng module/symbol + branch/predicate + causal signature.

Binding còn có blocker độc lập: governing invariant yêu cầu `3 company + 3 source + 3 real layouts`, trong khi cohort chỉ `2/2/2`. Vì vậy Binding luôn là `INSUFFICIENT_EVIDENCE` trong bước này.
