> **Audit source_period_v1 phát hiện lỗi cần xử lý:** ba ca CCM/DPC vẫn đúng, nhưng phản ví dụ mới cho thấy bỏ lọt mâu thuẫn cùng ngày, gán sai năm hồ sơ và chọn sai thời điểm trong năm. Grader cũ có thể PASS dữ liệu thiếu ca/sai đơn vị/ERROR. Runtime và gold chưa sửa. [Báo cáo audit](data/test/suites/source_period_review_v1/analysis.md).

> **Đã bàn giao source_period_v1:** binding kỳ nguồn và endpoint đã phục hồi 3 ca CCM/DPC; bộ development/replay đạt 4/4 câu có đáp án. 142 kiểm thử đạt, 230 chữ ký hồi quy giữ nguyên; 17 ca HTTP đạt tiêu chí riêng, gồm đáp án và guard. Không đổi transfer lần đầu, HRM/graph/encoder/checkpoint hoặc tolerance. [Báo cáo và giới hạn](data/test/suites/source_period_v1/analysis.md).

> **Bàn giao lookup_context_v1:** đã sửa tra cứu không nêu năm/nhãn–giá trị và kiểm backend bản cuối. 129 kiểm thử đạt; 230 chữ ký hồi quy giữ nguyên; câu vốn TIE và bốn phản ví dụ KH/TH đạt. **CCM/DPC mới chỉ 1/4 câu có đáp án trả đúng**, còn lỗi kỳ nguồn/đầu–cuối kỳ. [Báo cáo và hướng tiếp](data/test/suites/lookup_context_v1/analysis.md).

> **Đã kiểm lại bằng 33 câu HTTP thực tiễn:** 21/21 replay khớp hành vi cũ; 4/8 biến thể trả đúng số theo chính sách hiện hành; 3 ca MGG/ACG vẫn trả đúng, VIE chưa phục hồi. Đã xác định lỗi bảng nhãn–giá trị, role SELECT không nêu năm và mất ngữ cảnh bảng nối trang. Bản sửa đơn vị 9/9 kiểm thử chạy lại đạt; chưa sửa runtime trong lượt audit. [Báo cáo đúng/sai và giới hạn](data/test/suites/practical_review_v1/analysis.md).

> **Đã hoàn tất đánh giá độc lập 21 câu TIE/VEC/TAB: chưa đạt.** 3/10 câu có đáp án theo gold khóa trước chạy trả đúng đầy đủ; 1 sai KH/TH, 6 chưa trả đủ. Tách riêng 8 ca thiếu, 2 ca phạm vi/công bố và 1 ca dữ liệu một phần. Gold TAB ưu tiên số chi tiết có giới hạn cần đọc trong báo cáo; không đổi điểm/gold sau chạy. Runtime giữ nguyên. [Phân tích](data/test/suites/independent_reports_v2/analysis.md).

> **Bản sửa đơn vị đã kiểm chứng:** 121 kiểm thử + 3 gateway fixture đạt; 230/230 chữ ký giữ nguyên so với fact_plan_recovery_v1; backend 18/18 khớp và 3/3 ca cơ chế đơn vị đạt. VIE chưa phục hồi; UI thật chưa kiểm được. Bộ TIE/VEC/TAB 21 câu đang chuẩn bị nguồn dưới mã đã khóa, chưa có kết quả độc lập. [Bàn giao](data/test/suites/unit_evidence_v1/delivery.md).

> **Đang nghiệm thu unit_evidence_v1.** Đã sửa ưu tiên đơn vị cụ thể hàng/cột trước khai báo chung; 121 kiểm thử đạt, nhóm trọng tâm giữ kết quả. VIE vẫn chưa phục hồi; UI bị lỗi khởi tạo công cụ. Đã khóa metadata TIE/VEC/TAB và 21 câu; chỉ đọc nguồn sau khóa mã. [Tiến độ](data/test/suites/unit_evidence_v1/analysis.md).

> **Đã nạp fact_plan_recovery_v1 lên demo theo tongquan.md.** Phục hồi 2 MGG + 1 ACG: nhóm gốc 9/9 đúng; tính cả vốn VIE là 9/10, VIE vẫn chưa phục hồi. Chuyển PSD/NBW 4/4 số đúng + 2 ca thiếu năm giữ chặn; 112 post-HRM + 3 gateway thử riêng đạt; backend 18/18 khớp. Hồi quy 228/230 giữ chữ ký, hai MGG chuyển sang số chi tiết có căn cứ. [Bàn giao](data/test/suites/fact_plan_recovery_v1/delivery.md) · [Phân tích](data/test/suites/fact_plan_recovery_v1/analysis.md).

> **Đã cập nhật phản hồi theo từng chặng vào [tongquan.md](tongquan.md).** Bổ sung theo dõi span → candidate → fact trước sửa; sửa mô tả ngữ cảnh số, slot planner và cách dùng head HRM; phân biệt log triệu chứng với nguyên nhân và tiêu chí coverage/công bố sai. Các đề xuất chưa triển khai được đánh dấu; runtime và kết quả nghiệm thu giữ nguyên.

> **Tài liệu tổng hợp:** [tongquan.md](tongquan.md) — quá trình sửa đổi, kiến trúc hiện tại, kết quả và giới hạn kiểm chứng, cùng thứ tự xử lý tiếp. Chỉ tổng hợp tài liệu; không thay runtime hoặc chạy benchmark mới.

> **Đã đối chiếu phân tích kiến trúc tham khảo.** Tiếp nhận đánh giá từng chặng, ưu tiên plan ACG và đường số → binding của MGG/VIE; làm rõ các giới hạn tự chấm theo planner và vai trò HRM. Chưa thay runtime hoặc số liệu bàn giao. [Đối chiếu và hướng xử lý](data/test/suites/architecture_integration_v1/reference_review.md).

> **Đã nạp bổ sung kiến trúc lên demo.** Resolver trước planner, truy hồi thích ứng dùng graph hiện có, guard định dạng số và kiểm tra kế toán có điều kiện. 97 kiểm thử post-HRM + 3 gateway thử riêng đạt; giữ 230/230 đầu ra cũ; 12/12 lượt backend khớp bản kiểm chứng. Chỉ 6/9 câu có đáp án trong nhóm trọng tâm trả đúng, chưa có bằng chứng graph tăng số trả đúng; MGG/ACG còn thiếu, VIE mới chặn số sai. [Bàn giao](data/test/suites/architecture_integration_v1/delivery.md) · [Phân tích](data/test/suites/architecture_integration_v1/analysis.md).

> **Đã hoàn tất lượt chuyển công ty và nạp demo; chưa đạt tổng quát hóa ổn định.** MST/PSD/HMR: 12/25 → 25/25 câu có đáp án. Nhóm độc lập VIE/NBW/UDC: 15/24 theo quy ước, trong đó một ca chỉ dựa vào mặc định VND; tách ca đó còn **14/23**. Giữ 230/230 đầu ra cũ; 83 kiểm thử. Còn sai thang đọc số VIE, bỏ qua mâu thuẫn NBW, thiếu nhân sự/ngành nghề, nhãn bảng/câu văn và tỷ suất in sẵn. Mã khóa trước đọc nhóm độc lập và không sửa theo kết quả. [Phân tích và bước tiếp](data/test/suites/company_holdout_v1/analysis.md) · [Bàn giao](data/test/suites/company_holdout_v1/delivery.md). Các mốc dưới là lịch sử.

> **Đã mở rộng thông tin báo cáo và nạp demo.** 27/27 ca mới đạt tiêu chí; 203 ca trước giữ trạng thái/số/đơn vị; 71 unit tests; 27/27 lượt backend demo. Tra cứu tên/địa chỉ/liên hệ/mã/ngành nghề/nhân sự và giải thích mâu thuẫn AAH bằng từng công bố. Gateway đăng nhập chưa chạy lại vì auto-review chặn đọc credential. [Bàn giao và giới hạn](data/test/suites/report_information_v1/delivery.md).

> **Đã áp sửa cơ chế ngữ nghĩa/ngữ cảnh lên demo.** 15/15 ca trọng tâm; 188/188 giữ trạng thái/số/đơn vị cũ; 58 unit tests, 15/15 lượt demo, 12/12 phép phá công bố/nguồn được phát hiện. Hai AGP giữ đối chiếu riêng, bốn AAH vẫn chưa trả; chưa chứng minh mọi báo cáo. [Bàn giao và giới hạn](data/test/suites/semantic_transfer_v1/delivery.md). Các mốc dưới là lịch sử.

> **Kiểm toán mới: chưa đạt yêu cầu đa báo cáo.** Tái hiện lỗi ABW 2024 trên runtime bốn tầng mới. 15 câu/4 báo cáo/3 công ty: 8/10 câu có số trả đúng, 3/4 làm rõ đúng, 1 ca thiếu năm không công bố; còn lỗi trình bày. Bộ chấm cũ bỏ lọt 4 biến thể sửa sai nội dung/đơn vị; không dùng 182 khớp tự động để khẳng định 182 phản hồi đầy đủ đúng. Chưa sửa runtime trong lượt này. [Nguyên nhân, bằng chứng và thứ tự sửa](data/test/suites/cross_report_audit_v1/analysis.md).

> **Đã áp tầng sau HRM mới lên demo 8765**: không còn nhánh/fallback symbolic cũ. 42 kiểm thử đạt; 188 HTTP gồm 182 khớp tự động, 2 AGP đối chiếu nguồn riêng, 4 AAH chưa trả vì nguồn mâu thuẫn. Chuyển BHH/BHK/SSG 9/9; live 11/11; hiển thị 4/4. Runtime khớp hash trên đĩa, MinerU healthy. [Bàn giao](data/test/suites/post_hrm_contract_v1/delivery.md). Các mốc phía dưới là lịch sử.

> Luồng trả lời làm rõ — **đã áp lên demo 8765**. ABI: câu hỏi phần trăm→“Lợi nhuận sau thuế” trả giảm khoảng 16,9%. 48 kiểm thử đạt, 9/9 luồng hội thoại, 8/9 đối chiếu tham chiếu chi tiết và 2/2 lượt trực tiếp demo được xác nhận; hội thoại thử đã dọn. Gateway giữ câu hỏi chờ theo tài liệu/hash, hoàn thành đúng lựa chọn và giữ nguyên đường kiểm chứng. HRM/verifier không đổi; SSG và mâu thuẫn AAH vẫn chưa xử lý. [Báo cáo](data/test/suites/clarification_turn_v1/delivery.md).

> Phần trăm và chuyển báo cáo — **350 kiểm thử, 155/155 hồi quy đạt**. Bộ24 có 18 đáp án đúng, 2 hành vi làm rõ/thiếu năm đúng, 4 AAH chưa phục hồi; lặp 24/24 cùng phản hồi. Bộ chuyển BHH/BHK/SSG có 4/6 ca xác định chỉ tiêu trả đúng, 2 SSG còn bị chặn; 3 ca hỏi độc lập làm rõ loại lợi nhuận. Demo đã nạp bản kiểm tra, gateway 7/7 qua đối chiếu nguồn trên SQLite sao chép. Tiếp theo: truy vết SSG về định dạng/kỳ/thang đo, đối chiếu mâu thuẫn AAH, thu gọn số lần. [Mục lục](data/test/README.md) · [Phân tích](data/test/suites/profit_transfer_v1/analysis.md). Các mốc dưới là lịch sử; chưa chứng minh mọi báo cáo.

> AAS field transfer — đã phục hồi vốn điều lệ AAS và kiểm tra nhiều loại câu hỏi: **22/22 hành vi mục tiêu trên 6 báo cáo** (19 đầy đủ, 1 từng phần, 2 từ chối); lặp đảo thứ tự 22/22 cùng đầu ra. Giữ 54/54 câu và 62/62 đầu ra hồi quy, doanh thu 7/7; **335 kiểm thử đạt**. Demo 8765 đã nạp bản cuối, gateway 8/8 và đổi tài liệu 4/4 đạt. [Báo cáo](reports/finhrm_B_AAS_field_transfer_delivery_v1.md). Chưa chứng minh mọi báo cáo/câu hỏi; các mốc dưới giữ làm lịch sử.

> Consistency V3 — triển khai `plan_1.md`, bảo toàn **62/62 đầu ra cũ, 54/54 câu**, doanh thu7/core72; suite320. Development: **28/32 cặp cùng đúng**, còn sáu câu CMS/PDB. Nguồn mới: **8/24 câu có dữ liệu, 6/6 ca âm**; lặp **30/30** cùng chữ ký. Chưa chứng minh mọi báo cáo/câu hỏi. [Báo cáo](reports/finhrm_B_consistency_v3_delivery.md). Các mốc dưới giữ làm lịch sử.

> Verify-flex V2 — đã triển khai và chạy nghiệm thu, **chưa đạt toàn bộ plan**. AAH vốn điều lệ: 6/6 lần hỏi đầu đúng. Bộ độc lập sau/lặp: 29/80 và 29/80; core giữ 72/72; hồi quy chỉ giữ 59/62 đầu ra cũ. Giữ nguyên failure của test/oracle, không sửa theo bộ độc lập. [Báo cáo đầy đủ](reports/finhrm_B_verify_flex_v2_delivery.md). Các mốc dưới là lịch sử.

> Giao diện — đã thu gọn câu trả lời so sánh đầy đủ thành kết luận + bảng hai kỳ; nguồn/ghi chú một lần, công thức trong mục mở rộng. Giữ nguyên phản hồi và logic HRM; áp dụng cả lịch sử. 7 kiểm thử JS và build đạt, đã xem desktop/mobile. [Báo cáo](reports/finhrm_comparison_presentation_v1.md). Ctrl+F5 để nhận bundle mới.

> B — đã phục hồi câu AAH “so sánh doanh thu năm 2023 và năm 2024”: hỏi độc lập lần đầu, hỏi lặp, đảo năm và hỏi tiếp đều trả đúng hai năm/chênh lệch. Resolver xác định cùng chỉ tiêu trước khi chọn số; web giữ ngữ cảnh theo hội thoại/tài liệu; sửa thông báo 0 mục xác minh. **7 HTTP/3 nguồn đạt kỳ vọng (6 đầy đủ, 1 từng phần), 49 kiểm thử liên quan đạt; giao diện đã build.** [Báo cáo và giới hạn](reports/finhrm_B_revenue_context_implementation_v1.md). Đóng phạm vi lỗi doanh thu này; không tự mở rộng sang năm tồn tại khác hoặc training. Cần khởi động lại demo để nạp mã mới. Các mốc dưới giữ làm lịch sử.

> B — sửa hai lỗi đã tái lập: nhận diện tiền tệ “thu nhập” và gán năm từ câu hỏi. Đã giải quyết hai lỗi trong phạm vi đã giao và hoàn tất kiểm tra bảo toàn. Test 583; cùng 9 câu thực tế: 3/8 câu có dữ liệu được xác nhận (2 tự động + 1 đối chiếu nguồn riêng), 5/8 còn giữ; câu 2030 kiểm tra kỳ UNKNOWN. Hồi quy giữ 62/62 đầu ra đúng cũ; core 72/72. [Báo cáo đủ câu hỏi và phản hồi trước/sau](reports/finhrm_B_unit_period_implementation_v1.md). Không OCR/train/đổi tham chiếu; các lỗi ngoài phạm vi còn được ghi riêng. Các mốc dưới là lịch sử.

> Kiểm tra thực tế mới nhất — **9 câu mới/3 báo cáo MinerU, toàn luồng HTTP B**: 9/9 request hoàn tất, **2/8 câu có đáp án trả lời đúng, 6/8 chưa công bố**; 2/2 đầu ra công bố khớp tham chiếu. Ca âm 2030 không công bố nhưng lời giải thích chưa đúng trọng tâm và `period_coverage` vẫn VALID, nên không tính là đạt hoàn toàn. **Chưa chốt bản demo.** [Báo cáo có đủ 9 câu và nguyên văn phản hồi](reports/finhrm_B_practical9_results_v1.md).
>
> Runtime/model giữ nguyên bản column_mapping, không train/OCR/sửa code/đổi câu hỏi hoặc tham chiếu. Hai ca “khác” thiếu operand đúng trong pool cuối; bốn ca chọn đúng số/chuỗi số nhưng xác nhận còn thiếu. Ưu tiên tiếp theo phải truy vết mất bằng chứng và kỳ nguồn trước khi tăng khả năng công bố; chưa quy lỗi cho weights HRM. Đây là tập mới riêng, không thay điểm core 72/72 hoặc gộp vào HTTP 72 câu cũ; các mốc dưới là lịch sử.

> B hiện hành — công bố ô theo tên cột và phân biệt chỉ tiêu “khác”: **đã phục hồi 5/5 fact, 10/10 lượt gốc/paraphrase được model thật trả ANSWER; giá trị đúng, certificate/trình bày VALID**. Giữ core **72/72** và HTTP **62/62 đầu ra đúng** (72 câu/15 nguồn: 55 đầy đủ, 1 từng phần, 16 ca âm không công bố). **577 test đạt**; chỉ sửa ba file runtime, không thêm module/train/OCR/đổi gold. [Báo cáo và phản hồi thật](reports/finhrm_B_column_mapping_implementation_v1.md).
>
> Runtime khóa: `config/finhrm_B_column_mapping_protocol_v1.json`. 10 lượt trọng tâm dùng pool/context MinerU cũ, không phải truy hồi lại toàn báo cáo; không cộng vào mẫu số HTTP 72. Bộ đối chiếu đã phân biệt “Tài sản ngắn hạn” với “Tài sản ngắn hạn khác”; ánh xạ cột và hiển thị không ép thêm năm khi câu hỏi không yêu cầu. Đóng hai vấn đề đã giao trong phạm vi này; các kết quả là development, chưa chứng minh tổng quát trên mọi báo cáo. Các mốc dưới là lịch sử.

> B hiện hành — sửa đọc cột thay đổi và năm dính đơn vị (13/09/2026): **5/5 ca core đã chọn đúng**, giữ 67/67 ca đúng cũ; cùng dataset v2/gold/model: **execution/pair/program và operation macro-F1 100% (core 72/72, một nguồn/ba phép toán)**. Sáu file runtime hiện có, không thêm module/train/OCR; **571 test đạt**. HTTP 72 câu/15 nguồn giữ **55 đầy đủ đúng, 1 từng phần đúng, 16 ca âm không công bố, 62/62 đầu ra công bố đúng**. [Báo cáo và giới hạn](reports/finhrm_B_reported_column_implementation_v1.md).
>
> **Chưa phục hồi công bố cho chính 5 ca core:** 10 lượt B trên pool cố định (gốc + paraphrase) đều chọn đúng nhưng vẫn ABSTAIN. Bốn ca hỏi cột chưa có mapping khi không nêu năm tuyệt đối; ca `2022VND` còn nhiều ánh xạ/thiếu định danh công bố. Không đánh đồng điểm core với end-to-end. Runtime khóa: `config/finhrm_B_reported_column_final_protocol_v1.json`; không đổi gate, dữ liệu hay tham chiếu để ép công bố. Nếu tiếp tục, xử lý đúng ánh xạ công bố của nhóm này; không mở rộng sang training/OCR/parser. Các mốc dưới là lịch sử.

> B hiện hành — sửa chọn phạm vi operand (13/09/2026): **đã sửa đúng 4/4 ca Công ty mẹ/Hợp nhất**, giữ 63/63 ca core đúng cũ. Trên cùng dataset v2 và gold: **execution/pair/program exact 93,06% (67/72), operation macro-F1 94,54%**. Chỉ sửa `document_hrm.py` và `symbolic_verifier.py`, không thêm module runtime; **566 test đạt**. HTTP 72 câu/15 nguồn giữ **55 đầy đủ đúng, 1 từng phần đúng, 16 ca âm không công bố; 62/62 đầu ra công bố đúng và giữ đủ 62/62 đầu ra cũ**. [Báo cáo đầy đủ](reports/finhrm_B_operand_scope_implementation_v1.md), [delivery trước/sau](reports/finhrm_B_operand_scope_delivery_v1.json).
>
> Ràng buộc đọc nhãn phạm vi của cột chứa số và dùng chung cho decoder/search/verifier; không lấy nhãn từ header chung, không tạo proof VALID từ nhãn trùng câu hỏi. Runtime khóa: `config/finhrm_B_operand_scope_protocol_v1.json`. Dataset v2, câu hỏi, gold, model/weights giữ nguyên; không train/OCR/nới ngưỡng. Đóng lượt sửa bốn ca phạm vi. Còn 5 mẫu core ngoài phạm vi: bốn SELECT ở cột thay đổi bị chọn SUBTRACT và một câu chứa `2022VND` chọn sai năm; chưa xử lý trong lượt này. Core và HTTP là hai tập development khác nhau, chưa chứng minh tổng quát trên báo cáo mới. Các mốc dưới là lịch sử.

> B hiện hành — sửa nguyên số trong ô và dữ liệu benchmark (13/09/2026): **đã phục hồi hai cách hỏi HFX**, trả đúng 22.006.696.697 đồng từ nguyên ô `22006696 697`, kèm ghi chú mặc định VND. Chỉ sửa 3 file runtime; **560 test đạt**. HTTP 72 câu/15 nguồn: **55 đầy đủ đúng, 1 từng phần đúng, 16 ca âm giữ không công bố; 62/62 đầu ra công bố đúng, giữ 60/60 đầu ra đúng cũ**. Tỷ lệ request có ít nhất một phần trả lời **56/72 = 77,78%**; đầy đủ **55/72 = 76,39%**. [Báo cáo đầy đủ](reports/finhrm_B_cell_tokens_implementation_v1.md), [delivery trước/sau](reports/finhrm_B_cell_tokens_delivery_v1.json).
>
> Core dataset v2 giữ 72 mẫu, không đổi số/gold chương trình; bổ sung năm cho 6 câu và header cho candidate của 24 mẫu, hết 3 cặp gold mâu thuẫn. Sáu câu sửa đúng 6/6, nhưng 4 câu so sánh chọn nhầm Hợp nhất khi hỏi Công ty mẹ. **Execution/pair/program = 87,50% (63/72); operation macro-F1 = 94,54%; còn 9 mẫu lệch gold.** Đây là version dữ liệu mới, không phải kết quả train hay cải thiện mọi mặt. BTH chỉ sửa ưu tiên thông báo chỉ tiêu; câu hỏi số dư theo ngày vẫn từ chối, chưa sửa lựa chọn HRM ở ca đó. Runtime khóa: `config/finhrm_B_cell_tokens_protocol_v1.json`. Không train/OCR/nới gate; tiếp theo chỉ truy vết việc chọn phạm vi trên pool đủ năm, không mở rộng parser hoặc sửa riêng công ty/số đáp án. Các mốc dưới là lịch sử.

> Benchmark hiện hành (13/09/2026): **đã chạy mới core HRM và đủ 72 HTTP B trên 15 nguồn MinerU**. Core 72 mẫu từ một báo cáo: execution/pair/program exact **88,89%**, operation macro-F1 **94,54%**; có 3 cặp đầu vào giống nhau nhưng gold khác nhau, chưa sửa điểm. HTTP: **53/53 câu có đáp án trả lời đầy đủ đúng, 1 từng phần đúng, 60/60 đầu ra công bố đúng; 16 ca âm giữ không công bố**. Hai câu HFX có tham chiếu từ chối vì đơn vị đã lỗi thời báo riêng, vẫn chưa trả lời; không gọi 72/72 hoàn thiện. [Báo cáo đầy đủ và giới hạn](reports/finhrm_B_current_benchmark_v1.md), [delivery/hashes](reports/finhrm_B_current_benchmark_delivery_v1.json).
>
> Lượt này chỉ đánh giá, không sửa runtime/train/OCR. Protocol: `config/finhrm_B_current_benchmark_protocol_v1.json`; runtime giữ bản statement-context. Hai tập 72 khác nhau; core không so trực tiếp với benchmark 462 mẫu cũ. Tiếp theo tập trung nhận diện nguyên số trong ô (`22006696 697` đang bị lấy thành `22006696`) và sửa cách tạo câu hỏi core thiếu năm bằng version dữ liệu riêng. Không nới gate, không sửa theo công ty/số đáp án, không mở rộng parser hoặc train từ gold mâu thuẫn. Chất lượng giải thích từ chối và generalization chưa được chứng minh. Các mốc dưới là lịch sử.

> B hiện hành: **đã phục hồi 2 cách hỏi AGP**, trả 65,903 tỷ đồng từ cùng ô 65.903 triệu đồng đã chọn trước đó. Sửa ranh giới kỳ trong thuyết minh, liên kết phạm vi bằng tên pháp nhân/lời dẫn nguồn và đồng bộ alias chỉ tiêu đã có trong danh mục. **551 test đạt; HTTP cùng 11 câu: 9 đầy đủ đúng, 1 từng phần đúng, 1 từ chối đúng; 16/16 đầu ra công bố khớp tham chiếu.** Giữ 14/14 đầu ra HTTP cũ; replay 63 pool cũ giữ 44/44 công bố. [Báo cáo và giới hạn](reports/finhrm_B_statement_context_implementation_v1.md), [đối chiếu trước/sau](reports/finhrm_B_statement_context_delivery_v1.json).
>
> Runtime hiện hành: `config/finhrm_B_statement_context_protocol_v1.json`. Không train/OCR mới, không đổi câu hỏi/tham chiếu, không có ngoại lệ runtime theo công ty/trang/số đáp án. Đóng lượt sửa AGP; không tiếp tục mở rộng parser khi chưa có lỗi mới được xác nhận. Tập 11 câu vẫn là development; nếu đánh giá tiếp cần câu hỏi/nguồn tách khỏi các ca đã dùng sửa. Các mốc dưới là lịch sử.

> B hiện hành: dùng lại hai số đã xác minh để tính chênh lệch; sửa kiểm tra thứ tự KH/TH và định danh năm tương đối. **544 test; replay giữ 44/44 kết quả đã công bố trên 63 pool cũ.** HTTP 11 câu/4 báo cáo: 7 đầy đủ đúng, 1 từng phần đúng, 1 từ chối đúng, 2 AGP còn giữ vì phạm vi; 14/14 đầu ra công bố khớp tham chiếu. [Báo cáo và giới hạn](reports/finhrm_B_verified_operands_implementation_v1.md).
>
> BTH đã trả đúng chênh lệch 103.109.210.439 đồng. AGP vẫn chọn 65.903 triệu đồng nhưng chưa xác nhận được liên kết phạm vi; không đổi số hay tự gán scope. Tiếp theo chỉ xử lý liên kết nguồn đó khi có căn cứ, không mở rộng parser/training hoặc sửa riêng từng báo cáo. Các tập đã dùng đều là development. Inference: `config/finhrm_B_verified_operands_http_protocol_v1.json`; sau đó sửa câu chữ TH/KH, giữ nguyên kết quả/certificate và kiểm tra riêng 52 test, fingerprint cuối tại `reports/finhrm_B_verified_operands_presentation_v1.json`. Các mốc dưới là lịch sử.

> B hiện hành: ý nghĩa cột/kỳ, xác nhận định dạng bằng công bố cùng fact, và so sánh nhiều phần có kiểm chứng. **518 test; giữ 40/40 đáp án đúng trước sửa** trên các nhóm39/9/15. Nhóm thực tế9 request/17 đầu ra: 6/7 yêu cầu đủ dữ liệu trả lời đầy đủ, 1 từng phần đúng, 1 từ chối đúng. [Báo cáo](reports/finhrm_B_column_context_implementation_v1.md), [nghiệm thu phạm vi hẹp: HOLD](reports/finhrm_B_column_context_acceptance_v1.json). Runtime: `config/finhrm_B_column_context_protocol_v1.json`. Không OCR/train; chưa tuyên bố hoàn thiện mọi báo cáo. Các mốc dưới là lịch sử.

> Câu so sánh mở có hai năm hoặc KH/TH cùng năm nay trả hai số liệu và chênh lệch; phần thiếu không che phần đã xác minh. Tiếp theo ưu tiên các giới hạn phạm vi/công bố còn ghi trong báo cáo và chi phí ba lượt suy luận; không sửa riêng từng báo cáo hay gán năm cho mọi “Năm nay”. Nhóm thực tế mới nếu dùng sửa tiếp phải chuyển sang development.

> B hiện hành: chuẩn hóa tên chỉ tiêu viết tắt và tách yêu cầu đơn vị đầu ra. **464 test; giữ 28/28 đáp án đúng cũ.** Nhóm development 9 câu: 4/6 câu có dữ liệu trả lời được; AGP đã chọn đúng72 tỷ nhưng còn mapping “BQ5 năm”/“Năm nay”. Nhóm15 mới: 8/12 câu có dữ liệu; nghiệm thu **HOLD**. Đã so HRM/baseline cùng pool. [Báo cáo đầy đủ](reports/finhrm_B_metric_intent_implementation_v1.md), [nghiệm thu](reports/finhrm_B_metric_intent_acceptance_v1.json). Runtime: `config/finhrm_B_metric_intent_protocol_v1.json`. Không OCR/train; chưa hoàn thiện. Các mốc dưới là lịch sử.

> Việc tiếp theo phải dựa trên chẩn đoán sau vòng cố định này: xử lý ngữ nghĩa kỳ tổng hợp/năm tương đối và các nguyên nhân đã ghi ở nhóm15; không thêm ngoại lệ theo báo cáo, không nới mọi UNKNOWN, không train HRM khi chưa chứng minh lỗi chọn. Nhóm15 nếu dùng phát triển tiếp phải đổi nhãn development. Không coi 464 test hoặc tập nguồn quen thuộc là chứng nhận tổng quát hóa.

> B hiện hành: sửa phạm vi/doanh nghiệp, công thức mã dòng và công bố lặp của trường báo cáo. 456 test; giữ 25/25 đáp án đúng cũ. [Kết quả và giới hạn](reports/finhrm_B_context_identity_implementation_v1.md). Runtime: config/finhrm_B_context_identity_protocol_v1.json. CMS dùng quy ước VND mặc định, không khẳng định đơn vị có in trong PDF. Không OCR/train. Các mốc bên dưới là lịch sử.

> Phần cần làm tiếp, đã được nhóm câu mới chỉ ra: đồng nhất tên chỉ tiêu viết tắt giữa nguồn/planner/selector và tách yêu cầu đơn vị đầu ra khỏi metric. Nhóm mới 2/6 câu có dữ liệu trả lời được, 3/3 năm thiếu giữ đúng; còn4 câu bị giữ. Chưa sửa/tune theo nhóm này. Không coi456 test và3 ca mục tiêu đạt là đã chứng minh tổng quát hóa.

> B hiện hành: thiếu đơn vị tiền tệ thì mặc định đồng (VND), ghi rõ giả định trong citation/certificate/phản hồi theo yêu cầu người dùng. 445 test B/dùng chung đạt; giữ 25/25 đáp án đúng cũ. [Kết quả và phần còn lại](reports/finhrm_B_default_unit_v2_implementation_v1.md). Runtime: config/finhrm_B_default_unit_v2_protocol_v1.json. Không OCR/train. Các mốc bên dưới là lịch sử.

> B hiện hành: đã sửa gắn số với doanh nghiệp theo mục nguồn; HAV trả lời đúng 32.978.600.000 đồng của công ty báo cáo và 2.200.000.000.000 đồng khi hỏi công ty mẹ. Giữ 25 đáp án đúng/11 ca cần từ chối/2 CMS chưa chấm trong bộ 38; nhóm 4 cũ tăng 2 → 3 đáp án đúng. 432 test đạt. [Kết quả và giới hạn](reports/finhrm_B_entity_owner_implementation_v1.md). Runtime: config/finhrm_B_entity_owner_v2_protocol_v1.json. Không OCR/train, chỉ B. Các mốc bên dưới là lịch sử.

> B hiện hành: sửa mapping trường thông tin theo năm báo cáo. Cùng 38 câu: 24 → 25 đáp án xác nhận, giữ cả 24 đáp án cũ; 11 ca cần từ chối và 2 CMS chưa đủ tham chiếu. Đã đối chiếu GGG trên hai bản trước sửa; lỗi đã có trước source-context. [Kết quả, phản hồi thật và giới hạn](reports/finhrm_B_report_fields_implementation_v1.md). Runtime: config/finhrm_B_report_fields_protocol_v1.json. Không OCR/train, chỉ B. Các mốc bên dưới là lịch sử.

> B hiện hành: bản sửa ngữ cảnh nguồn. Nhóm 34 câu development có 23 đáp án xác nhận (trước 19); giữ baseline19: PASS. Đã thử 8 câu trọng tâm rồi chạy 38 HTTP (34 hồi quy + 4 câu bổ sung chấm riêng). 407/407 test đạt. SSG chỉ OCR lại trang 6 bằng MinerU; artifact mới bổ sung đơn vị sau khi khớp nguyên bảng. BSD đối chiếu số theo cùng fact/precision; PDB dùng phạm vi hỏi khi chọn; thông báo từ chối rõ hơn. Không train, chỉ B. [Báo cáo và giới hạn](reports/finhrm_B_source_context_implementation_v1.md). Runtime: config/finhrm_B_source_context_final_protocol_v1.json. Các mốc bên dưới là lịch sử.

> Quyết định hiện hành (2026-09-10, theo yêu cầu người dùng): **chỉ triển khai và đánh giá B; dừng so A/B, dọn code A để sau**. So B trước/sau, không để calibration hoặc test riêng A chặn công việc B. [Kế hoạch chi tiết](ke_hoach_B_cong_bo_lap_va_do_chinh_xac_v1.md): cấu hình rõ B cho runner/demo trong bước code tới; xử lý quan hệ phạm vi của công bố lặp HKT trước, rồi precision/đơn vị/nhãn chỉ tiêu AGP; hồi quy 22 câu và thử 6–10 câu trên ít nhất 3 báo cáo OCR khác. Lượt này chỉ cập nhật tài liệu, chưa sửa default/runtime hoặc chạy đánh giá mới. Các yêu cầu so A/B và giữ A làm mặc định ở các mốc dưới là lịch sử, không còn là kế hoạch áp dụng.

> Công bố lặp/report-year/phản hồi ngắn (2026-09-10): đã sửa HFX kế hoạch, MGG tổng doanh thu, BTH vốn điều lệ. B **5 → 8/10 câu gốc**, hai ca thiếu dữ liệu giữ nguyên; 4 paraphrase: 3 đúng/1 giữ; 2 ca report-year/fact-date giữ đúng. Sáu câu thêm trên ACG/AGP/HKT có OCR sẵn: 3 công bố đúng, 2 còn AMBIGUOUS, 1 thiếu kỳ giữ. 44/44 HTTP, 363 test chạy/362 methods đạt, còn một test lịch sử hai subcase. Có review riêng 4 response evaluator chưa hỗ trợ; không đánh đồng tự động và review nguồn. Không OCR/train, B vẫn thử nghiệm. Runtime: config/finhrm_disclosure_repair_protocol_v1.json. [Báo cáo, câu trả lời thật và bằng chứng](reports/finhrm_disclosure_repair_implementation_v1.md). Ưu tiên tiếp: truy vết mapping AGP chênh lệch và HKT LNST trên artifact sẵn có. Các mốc dưới là lịch sử.

> Sửa theo kế hoạch (2026-09-10): phục hồi cả ba ca mục tiêu. B công bố đúng **2 → 5/10 câu gốc** (4 tự động xác minh, 1 câu văn BTH review PDF riêng), A 0/10; 2/2 ca âm giữ nguyên. Bốn cách hỏi lại báo riêng: 3 đúng, 1 thiếu đơn vị giữ lại. Sửa hợp đồng triệt tiêu đơn vị cho tỷ lệ, quota giữ operand chưa rõ role, và binding/provenance mệnh đề có ngày. 357 test chạy, 356 methods đạt, một test lịch sử còn hai subcase lỗi. Không OCR/train; B vẫn thử nghiệm. Runtime: config/finhrm_policy_candidate_repair_protocol_v1.json. Xem [báo cáo sửa, ví dụ và bằng chứng](reports/finhrm_policy_repair_implementation_v1.md). Còn hai mapping AMBIGUOUS và lỗi phân biệt năm báo cáo/ngày ký ở vốn điều lệ; đây là ưu tiên tiếp. Các mốc dưới là lịch sử.

> Đánh giá nguồn mới (2026-09-10): đã OCR đủ HFX/BTH/MGG, 23 trang, rồi chạy 10 cặp A/B với core/weights giữ nguyên. A công bố 0/10, B công bố đúng 2/10; hai câu là paraphrase của cùng fact MGG, tương đương 1/7 câu có đáp án riêng biệt. Hai ca âm được giữ; 6/8 câu có đáp án chưa công bố. Không chuyển B mặc định. Xem [báo cáo nguồn mới và nguyên nhân](reports/finhrm_policy_new_eval_report_v1.md). Ưu tiên tiếp: tách năm báo cáo/ngày ký/kỳ fact, binding câu văn và giữ operand khi hỏi chênh lệch; không train HRM hoặc nới proof từ kết quả này. Nguồn tách biệt trong phạm vi audit tên/mã công ty, không chứng nhận mọi lịch sử/alias. Các mốc phía dưới là lịch sử trước lượt này.

> Cập nhật evaluator (2026-09-10): đã kiểm tra identity kế thừa theo dòng/hash và ranh giới nguồn, tách reporting_company khỏi riêng/hợp nhất. Chấm lại 8 cặp đã lưu: B công bố vẫn 4/8, cả 4 qua kiểm tra tự động v4; 3 AMBIGUOUS và 1 MISSING giữ nguyên. 24 test PASS, hash runtime/nguồn/oracle/raw HTTP/proposal không đổi. Không OCR hoặc chạy HRM. Xem [triển khai evaluator v4](reports/finhrm_evidence_policy_evaluator_implementation_v4.md). Tiếp theo khóa tập tách biệt để so A/B; B vẫn thử nghiệm, A vẫn mặc định. Các mục nói evaluator còn đọc raw metadata phía dưới là mốc lịch sử.

> Cập nhật triển khai (2026-09-09): đã thêm policy B thử nghiệm, mặc định vẫn A. So A/B cùng pool/prediction trên 8 câu / 3 báo cáo MinerU: A công bố 0/8, B công bố 4/8; review nguồn xác nhận 4 câu, còn 3 AMBIGUOUS và 1 MISSING. Hai cờ raw role/scope của evaluator v2 được giải thích riêng bằng nguồn ACG; chưa có đánh giá độc lập để đổi mặc định. Runtime: `config/finhrm_evidence_policy_protocol_v1.json`. Xem [báo cáo triển khai và giới hạn](reports/finhrm_evidence_policy_implementation_v1.md). Tiếp theo hoàn thiện evaluator về identity kế thừa rồi khóa tập tách biệt để đánh giá A/B. Các trạng thái “chưa triển khai policy” bên dưới là mốc lịch sử.

> Chiến lược kế tiếp (2026-09-09): [thử chính sách công bố dựa trên bằng chứng](chien_luoc_cong_bo_dua_tren_bang_chung_v1.md). So A/B trên cùng proposal, giữ kiểm chứng và executor; calibration không là điều kiện bắt buộc ở nhánh B thử nghiệm. Chưa đổi mặc định, chưa triển khai hoặc chạy đánh giá policy mới. Các mốc dưới đây là kết quả runtime hiện tại.

> Trạng thái mới nhất (2026-09-09): đã sửa mất ngữ cảnh caption liền nhau và phần trăm cạnh tranh sai với chỉ tiêu tiền tệ. Chạy lại 5 câu HTTP trên nguyên Markdown MinerU: khớp tham chiếu cơ bản 5/5, đủ proof 0 → 4/5, công bố 0/5 vì calibration DISABLED; câu chênh lệch AGP còn AMBIGUOUS. Xem [triển khai và giới hạn](reports/finhrm_v14_2_mineru_verify_fix_implementation_v1.md). Runtime hiện tại: `config/finhrm_v14_2_mineru_verify_fix_protocol_v1.json`. Không OCR/train mới; ALT còn thiếu. Tiếp theo ưu tiên kiểm tra chính sách công bố/calibration trên dữ liệu tách biệt và phép so cùng evidence theo rule.md; không nới gate từ 5 câu này.

# Kế hoạch tiếp theo cho FinHRM — chu kỳ V14.2 development

Ngày lập: 2026-09-05  
Trạng thái hiện tại (2026-09-08): Phase 0–2 đã triển khai; Phase 3 đã có replay Markdown development và audit lợi ích gate. Useful coverage/promotion và generalization HOLD; Phase 4–5 chưa thực hiện. Certification: HOLD.

Kết quả Phase 0: [baseline closure và action audit](reports/finhrm_v14_2_baseline_closure_v1.md).
Kết quả Phase 1: [gate diagnosis](reports/finhrm_v14_2_gate_diagnosis_v1.md), đủ 462 câu; observer không đổi prediction/gate/metric (PASS 462/462). Regression: 41 test và 31 subtest PASS. Bước tiếp theo là review regression Phase 2 theo nguồn, rồi đánh giá Markdown end-to-end ở Phase 3.

Kết quả Phase 2 (2026-09-07): [triển khai và đánh giá v2](reports/finhrm_v14_2_phase2_implementation_v2.md), [kiểm tra độc lập](reports/finhrm_v14_2_phase2_final_audit_v2.json). Đã đồng bộ quyết định policy-calibrated, chuẩn hóa tọa độ nguồn có version, sửa logical uniqueness theo role thực sự áp dụng và yêu cầu mapping UNIQUE ở search cho evidence mới. Regression mã cuối: **244 test, 43 subtest PASS**.

Đánh giá lại đủ 462 mẫu: program exact **81.39%**, execution **82.47%**, proof-eligible **68/462**, policy accepted **0/462**. Đây vẫn là chẩn đoán SyntheticDocument; chưa phải coverage phục vụ thực tế. Kết quả v1 và lần sửa search v2 đều được giữ để đối chiếu. HRM V14.1 scratch seed31 và threshold giữ nguyên; certification HOLD. **Program exact giảm 382 → 376/462, execution giảm 387 → 381/462; chưa chốt bản này thay baseline.** Sau inference, requirements.txt đã loại dependency của đường đọc PDF cũ; audit lưu bản gốc khớp SHA trong snapshot riêng, không sửa file hiện tại. Replay phải dùng snapshot đã khóa. 68 ca proof-eligible đều đúng và giữ đủ 28 ca cũ, nhưng đó không bù trừ cho regression toàn tập.

Cập nhật sau review ảnh nguồn: [source binding và annotation review v2](reports/finhrm_v14_2_source_review_implementation_v2.md). Ca CNG đã xác nhận candidate gán sai vùng và gold chưa thống nhất ý nghĩa số tiền/tỷ lệ; quarantine annotation cũ trong manifest riêng. Không gọi ca này là bằng chứng search làm mất đáp án đúng về ngữ nghĩa.

Guard mới chặn ô trộn phần trăm/số không nhãn ở search và public gate, kể cả khi mapping UNIQUE. Audit 462 mẫu: **1 quarantine, 45 cần review nguồn, 416 chưa có cờ tự động**; không coi 416 là đã sạch. Regression **253 test, 45 subtest PASS**. Gate replay đủ 462 prediction (185 DAG) giữ **68 → 68 proof-eligible**, không nâng ANSWER; chưa có accuracy neural/search mới. Dataset, gold và mẫu số lịch sử giữ nguyên.

Đã hoàn tất [review PDF 45/45 ca](reports/finhrm_v14_2_pdf_source_review_v1.md) trên **7 PDF, 8 trang vật lý**. Kết luận: **22 ca sửa metadata/candidate để giữ ý hỏi và số theo đơn vị trình bày; 23 ca quarantine để dựng lại câu hỏi/gold**. Ca CNG đã quarantine trước đó cũng được xác minh lại bằng PDF có hash. Overlay mới `reports/finhrm_v14_2_source_review_overlay_v3.jsonl` có 22 cần sửa metadata, 24 quarantine (gồm CNG cũ), 416 chưa review nguồn. Không còn 45 ca chờ xem PDF; còn công việc sửa dữ liệu theo kết luận review.

Theo yêu cầu loại các ca sai, đã tạo [working set 438 mẫu](data/tmp/v14_2_pdf_review_working_v1/validation_working_set.jsonl), loại 23 ca mới và 1 CNG đã quarantine trước đó. [Biên bản loại mẫu](reports/finhrm_v14_2_quarantine_materialization_v1.md) lưu hash/ID. **22 ca metadata vẫn chưa sửa**, 416 ca chưa review nguồn; working set chưa sẵn sàng training/calibration. Dataset và mẫu số lịch sử giữ nguyên.

Cập nhật mới nhất: **đã sửa xong dữ liệu 22/22 ca**, xem [biên bản sửa metadata v2](reports/finhrm_v14_2_metadata_repair_v2.md). Working set sử dụng tiếp là `data/tmp/v14_2_pdf_review_repaired_v2/validation_repaired.jsonl`: 22 sửa, 416 nguyên byte, 24 quarantine vẫn bị loại. Overlay v4 ghi trạng thái SOURCE_METADATA_REPAIRED. 32 test/4 subtest PASS; source replay và exact certificate 22/22 PASS. Gold public gate hiện 15 ANSWER/7 ABSTAIN; chưa chạy HRM inference/training hay công bố accuracy mới. Các đoạn “22 chưa sửa” phía trên là mốc lịch sử trước bản cập nhật này.

Cập nhật mapping: **đã xử lý 7/7 ca**, gold public proof trên nhóm đã review nguồn đạt **22/22** (trước 15/22). [Báo cáo mapping và so sánh 438 mẫu](reports/finhrm_v14_2_mapping_implementation_v1.md), [audit cuối](reports/finhrm_v14_2_mapping_final_audit_v1.json). Regression **351 test, 77 subtest PASS**. Inference runtime mới cùng dataset 438 mẫu: program exact **82.19%**, execution **83.33%**, proof eligible **97/438**, policy accepted **0/438**. Trạng thái: PASS_MAPPING_DEVELOPMENT; certification HOLD. Các đoạn nói còn 7 ca mapping ở trên là mốc trước bản sửa này.

Cập nhật Phase 3: đã hoàn tất [Markdown end-to-end development v1](reports/finhrm_v14_2_markdown_e2e_v1.md): **64 request / 73 output / 15 nguồn**, gọi HTTP API thật, đọc nguyên Markdown. **64/64 HTTP 200**, notice **14/14** từ chối đúng vì thiếu dữ liệu; **4/73 proof-eligible**, **0/73 accepted**, accepted-wrong **0**, precision **N/A**. 73 output vẫn policy DISABLED. Điều kiện coverage tối thiểu không đạt; generalization/certification **HOLD**, vì nguồn đều là development đã biết. 19 test và 9 subtest PASS cho evaluator/certificate. Evaluator revision v2 sửa đối chiếu tên Markdown với provenance PDF; không đổi raw inference, gold/action hoặc gate, aggregate metrics không đổi.

Các lỗi có ví dụ nguồn: GGG chọn tỷ số thay doanh thu; phép chênh lệch GGG chọn đúng số nhưng chưa có unit; HAV chọn diện tích m² nhưng metadata unit lại là tỷ đồng. Ca mới mơ hồ giữa hai năm bị phân loại MISSING thay vì AMBIGUOUS. Đây là cơ sở cho lượt sửa binding/metadata tiếp theo, chưa phải bằng chứng cần train HRM lại.

Cập nhật binding v2: đã sửa unit cục bộ, nhận diện khai báo currency, ưu tiên metric cho Markdown, phân biệt số tiền/tỷ số, lựa chọn năm mơ hồ và giữ ô trống cuối bảng. [Báo cáo vì sao sửa và lợi ích](reports/finhrm_v14_2_binding_implementation_v2.md). Trial v1 mất proof HAV do một ô kế hoạch 2024 bị lệch sang Ghi chú; đã truy nguyên bằng PDF, sửa tokenizer và chạy lại đầy đủ. Bản cuối: **64/64 HTTP 200; notice 14/14; proof 6/73; không mất ca proof baseline; accepted 0/73; đúng lý do abstention 17/17**. 181 test, 80 subtest PASS. Runtime mới khóa ở `config/finhrm_v14_2_binding_protocol_v2.json`; không dùng protocol mapping cũ làm current runtime.

Review PDF xác nhận bảng GGG không khai báo unit và số lỗ khác giữa trang 4/8 có sẵn trên bản gốc. Overlay review giữ 6 ca GGG HOLD nguồn; không thay gold/mẫu số, không tự gán VND. HAV SELECT doanh thu vẫn là vấn đề selection/authority riêng dù lỗi unit diện tích đã được sửa. Calibration, useful coverage và generalization/certification chưa đóng.

Cập nhật selection v1: đã xử lý nguyên nhân HAV SELECT chọn diện tích dù pool có doanh thu. Metric mask cũ ưu tiên từ công ty/currency trong đoạn dài, loại dòng đúng rồi bỏ period khi hai mask không giao nhau. Bản mới dùng metric VALID của verifier và loại dimension trái yêu cầu trước lexical scoring. [Báo cáo trace và lợi ích](reports/finhrm_v14_2_selection_implementation_v1.md). **HAV chọn đúng 2.341 triệu đồng trang 20**; full replay **64/64 HTTP 200, notice 14/14, proof 6 → 10/73, không mất proof baseline; accepted 0/73**. Regression 193 test/93 subtest PASS. Current runtime: `config/finhrm_v14_2_selection_protocol_v1.json`; chỉ sửa document_hrm.py, không đổi weights/calibration/gold.

Các đoạn trước nói HAV vẫn chọn diện tích là lịch sử trước bản này. Chọn đúng bằng chứng không tự giải quyết authority ở tất cả các bảng; source/scope/unique mapping và calibration vẫn bắt buộc. Useful coverage/generalization/certification chưa đóng.

Cập nhật proof parser v1: đã audit đủ 73 output, phân loại 46 output có gold ANSWER nhưng chưa đủ proof; observer khớp baseline **64/64** ngoài trường thời gian. Đã sửa parser proof dùng chung quy tắc giữ ô trống/căn header với extractor, xác nhận sửa 4 lần xuất hiện operand thuộc 3 output GGG. [Báo cáo nguyên nhân và kết quả](reports/finhrm_v14_2_proof_parser_implementation_v1.md). Regression **191 test/100 subtest PASS**; bộ QP3B cũ có 9 test chưa kiểm chứng được do thiếu fixture, được ghi riêng. Replay cuối **64/64 HTTP 200, proof 10 → 10/73, không mất proof cũ, notice 14/14, từ chối đúng lý do 17/17, accepted 0/73**. Header đúng hơn nhưng coverage chưa tăng vì nhóm GGG vẫn thiếu unit/nguồn chưa thống nhất. Current runtime: `config/finhrm_v14_2_proof_parser_protocol_v1.json`; chỉ evidence_schema.py thay so với selection v1. Không đổi weights/calibration/gold.

Cập nhật audit lợi ích gate (2026-09-08): đã đối chiếu 73 proposal trước public gate với nguồn, review trực quan 28 trang/8 PDF và tái sử dụng review notice đã khóa. [Kết quả và giới hạn](reports/finhrm_v14_2_gate_utility_v1.md): **10 proposal đúng qua proof; 16 proposal được nguồn hỗ trợ nhưng bị chặn; 17 proposal sai bị chặn; 7 giới hạn nguồn; 6 chưa kết luận; 17 negative từ chối đúng**. Không gọi tất cả 16 là gate vô ích: 4 mất unit footer, 6 chưa ràng buộc section hỏi, các nhóm còn lại liên quan role/precision/fact không nêu năm. Calibration vẫn DISABLED; runtime/weights/gold không đổi. Hướng tiếp theo ưu tiên nhóm cấu trúc có bằng chứng, không sửa lần lượt theo tên công ty.

Cập nhật triển khai scope/unit (2026-09-08): đã sửa section intent/binding, giữ `table_footnote` của MinerU, gắn unit đúng bảng/nối trang và bảo toàn dòng trống để một ô không bị thành hai tọa độ qua chunk overlap. [Báo cáo cuối](reports/finhrm_v14_2_scope_unit_implementation_v3.md). **Proof 10 → 18/73, gain 8/loss 0; 64/64 HTTP; notice 14/14; negative đúng lý do 17/17**. 6/6 output mục tiêu section phục hồi; 4/4 output mục tiêu unit có VND đúng, 1 đã qua proof và 3 còn role/mapping. Trong 17 proposal từng sai: 16 vẫn bị chặn, 1 đã sửa đúng trước khi qua proof. Regression **237 test/104 subtest PASS**, observer 64/64 khớp ngoài thời gian, 4.582 lượt dòng chunk/15 nguồn đúng tọa độ. Runtime current: `config/finhrm_v14_2_scope_unit_protocol_v3.json`; nguồn đã bổ sung footnote từ cache ở manifest `config/finhrm_v14_2_scope_unit_v2_requests.json`. Bản nguồn mới thay đổi 7 tài liệu/35 trang; PDF, Markdown cũ, câu hỏi/gold và weights giữ nguyên. Calibration vẫn DISABLED; accepted 0/73.

Cập nhật role/fact (2026-09-08): [implementation v2](reports/finhrm_v14_2_role_fact_implementation_v2.md). Đã nhận diện cách hỏi số đã ghi nhận/dự kiến, xác minh công bố lặp có metadata đầy đủ và hiện yêu cầu làm rõ KH/TH ở câu đơn/nhiều phần. **294 test/150 subtest PASS; HTTP 64/64; proof giữ 18/73, gain 0/loss 0; 73 proposal giữ nguyên; negative đúng lý do 17/17**. 10 output có yêu cầu làm rõ công khai. Ba output HKT mục tiêu còn thiếu role nguồn/phạm vi tương đương; chưa coi là đã sửa xong. Current runtime `config/finhrm_v14_2_role_fact_protocol_v2.json`; nguồn/gold/weights không đổi, calibration DISABLED.

Cập nhật ngữ cảnh bảng (2026-09-08): [implementation v2](reports/finhrm_v14_2_table_context_implementation_v2.md). Đã gắn role/scope/entity từ tiêu đề và chú thích có provenance, ưu tiên header, sửa ranh giới caption và ghép theo dòng nguồn. **313 test/158 subtest PASS; replay 64/64 HTTP, proof giữ 18/73, gain 0/loss 0, negative 17/17**. Nhóm bổ sung riêng 3 câu hỏi HKT đã làm rõ có **3/4 output đúng nguồn và đủ proof**; output LNST còn chọn nhầm KH và bị chặn. Không cộng nhóm bổ sung vào 73. Current runtime `config/finhrm_v14_2_table_context_protocol_v2.json`; calibration DISABLED, nguồn/gold/weights giữ nguyên.

Cập nhật ưu tiên đọc/trả lời (2026-09-08): [triển khai reading v2](reports/finhrm_v14_2_reading_implementation_v2.md). Đã phục hồi LNST thực hiện HKT đúng mục 4.1: 4,8 tỷ KH → 461.013.161 đồng TH; mặc định TH ở luồng public và sửa thông báo policy. Bộ thử riêng 10 câu: 6 proof/0 accepted; cùng pool HRM khớp tham chiếu 8/9, bộ đơn giản 7/9 (9 câu có số liệu), chưa đủ kết luận tổng quát. Full 64 request giữ 18/73 proof, gain/loss 0, negative 17/17. 298 test/250 subtest PASS. Current runtime `config/finhrm_v14_2_reading_protocol_v2.json`. Còn lỗi “Cho biết…” làm đổi metric/financial identity và proof trên cùng pool; ưu tiên chuẩn hóa ý hỏi, không training mới. Calibration/coverage/certification HOLD.

## 1. Quyết định ưu tiên

Chốt checkpoint V14.1 scratch seed31 làm baseline cho tầng đề xuất HRM và giữ
nguyên weights/kiến trúc trong chu kỳ hoàn thiện tầng sau. Kết quả development
hiện tại đủ tốt để chuyển bước; chưa phải chứng minh đã hoàn thiện trên mọi báo cáo.
Không yêu cầu train lại HRM để nâng program exact hoặc accepted coverage trước
khi triển khai tầng kiểm chứng.

Phân công trách nhiệm:

- HRM cùng pipeline đề xuất cung cấp operation, evidence candidates và top-10 programs.
- Tầng symbolic kiểm tra metric/kỳ/entity/unit/physical mapping và lựa chọn
  phương án đủ chứng cứ trong danh sách, không chỉ tin phương án đứng đầu.
- Executor thực hiện phép tính chính xác trên operand đã xác minh.
- Tầng trả lời áp dụng proof, certificate và policy để quyết định ANSWER/PARTIAL/ABSTAIN.

Trước mắt tập trung vào evidence binding, kiểm chứng/chọn phương án, đồng bộ
policy và đánh giá end-to-end. Cải tiến HRM là nhánh dự phòng, không phải bước
bắt buộc của chu kỳ này.

Mục tiêu là tăng số câu trả lời có bằng chứng đầy đủ trên nhiều công ty/layout,
đồng thời giữ sai chấp nhận bằng 0 trên tập đánh giá. Không đặt mục tiêu trả lời
mọi câu: dữ liệu thiếu hoặc mơ hồ phải được từ chối đúng.

Không ưu tiên OCR thêm, đổi embedding, tăng top-k hay cài LEAN ở giai đoạn đầu.

## 2. Baseline và cách đọc kết quả

Nguồn chuẩn cho checkpoint scratch hiện tại:

- [Báo cáo triển khai](reports/finhrm_v14_1_exact_symbolic_execution_v1.md).
- [Development evaluation V2](reports/finhrm_v14_1_bkai_scratch_seed31_development_evaluation_v2.json).
- Checkpoint: `models/document_hrm_V14_1_bkai_scratch_seed31_v1.pt`.
- Checkpoint SHA-256: `e6641d13f169dc674728c32aafb508946495027985ce1fde6e2b1095a8d55413`.
- Dataset: `data/tmp/v14_development_v1/validation.jsonl`, 462 mẫu.
- Dataset SHA-256: `3ddcb5e0a88e431787133b36edacd28a4bbb8b0fd2d1ac29494c0e9ffeec0758`.

| Chỉ số | Kết quả | Ý nghĩa cho bước tiếp theo |
|---|---:|---|
| Operation accuracy | 98,48% | Chưa cần ưu tiên sửa operation head |
| Logical program exact | 82,68% — 382/462 | Ưu tiên kiểm chứng/chọn lại trong top-10 ở tầng sau, chưa suy ra cần train HRM |
| Execution accuracy | 83,77% — 387/462 | Đúng số chưa đủ chứng minh đúng evidence |
| Physical program recall@10 | 97,19% — 449/462 | Phần lớn gold program đã xuất hiện trong beam |
| Operand candidate recall@20 | 99,57% | Truy hồi candidate chưa phải ưu tiên số một trên tập này |
| Proof-eligible coverage | 6,06% — 28/462 | Rất ít winner đủ điều kiện proof công khai |
| Proof-eligible program/execution precision | 100% — 28/28 | Kết quả tốt nhưng support nhỏ |
| Policy-accepted coverage | 0% | Xử lý ở tầng proof/policy/calibration; không phải thước đo riêng HRM |
| Accepted precision | N/A | Không có mẫu accepted để tính precision |

Hai chỉ số program exact và accepted coverage tiếp tục được đo ở tầng sau.
Recall@10 = 97,19% là tín hiệu tốt cho hướng kiểm chứng/chọn lại, không phải
cam kết exact cuối sẽ đạt 97,19%: verifier còn phải phân biệt được phương án đúng
bằng chứng cứ. Tầng sau không thể chọn gold program nếu nó không có trong danh
sách, và không được tạo bằng chứng cho thông tin thiếu trong nguồn.

Lưu ý bắt buộc:

- Các accuracy trên là kết quả đường decode/search hiện có, đã có hỗ trợ
  planner/symbolic; không được gọi là accuracy của HRM neural thuần.
- 462 mẫu đều có nhãn `SyntheticDocument`, dùng candidate có nguồn gốc báo cáo.
  Đây là chẩn đoán học chương trình, chưa phải benchmark đọc Markdown end-to-end.
- Tập này đã được dùng nhiều lần để phát triển/chọn epoch; không còn là test mù.
- p3 = 72/80 và p4 = 36/40 không so trực tiếp với p1 = 391/462, p2 = 286/351.
  p5–p8 không có support và phải ghi N/A.
- `proof_eligible` chưa bao gồm toàn bộ certificate/presentation/API.
  Không dùng 6,06% làm coverage production.
- 28/28 có cận dưới Wilson hiện tại khoảng 87,94%, chưa đạt 90%.
  Với cùng công thức, 35/35 mới vượt 90%; đó chỉ là minh họa support cần thiết,
  không phải chỉ tiêu để tạo thêm các câu dễ hoặc câu lặp.
- `current_res.md` chứa lịch sử nhánh warm-start. Khi bổ sung kết quả scratch,
  phải phân biệt checkpoint và protocol; không chép đè số liệu lịch sử.

## 3. Phase 0 — khóa baseline và protocol mới

Thực hiện trước mọi sửa runtime tiếp theo:

1. Tạo manifest V14.2 chứa hash checkpoint, toàn bộ runtime/evaluator, dependency,
   encoder revision, template/cache fingerprint, seed, chế độ ACT và dataset.
2. Ghi riêng kết quả fixed-max-segments và adaptive serving. Hiện evaluator batch
   và API có thể dùng số bước suy luận khác nhau; không so ngang khi chưa khóa chế độ.
3. Giữ nguyên evaluator/report V2, V13 và các artifact V14.1 đã có.
4. Tách regression chức năng hiện tại khỏi kiểm tra tái lập snapshot lịch sử.
   Kiểm tra lịch sử đang lỗi vì runtime hash, đường dẫn catalog cũ và NumPy khác.
   Chỉ phục hồi snapshot lịch sử trong môi trường riêng khi có đủ artifact;
   không sửa hash kỳ vọng để chạy runtime mới dưới tên kết quả cũ.
5. Khai báo tập development đã mở, tập calibration chỉ dùng hiệu chỉnh và tập
   protected chưa dùng phát triển. Tận dụng nguồn hiện có; không gộp cả data
   thành train và không chuyển file tùy tiện.

Đầu ra: `config/finhrm_v14_2_protocol_v1.json` và baseline manifest có version.

Điều kiện đạt: xác định được đầy đủ baseline, nguồn dữ liệu và chế độ chạy.
Thiếu nguồn/lineage để chứng minh độc lập thì ghi HOLD cho certification.

## 4. Phase 1 — chẩn đoán nguyên nhân theo từng gate

Đây là bước triển khai đầu tiên sau khi khóa baseline.

Tạo telemetry development-only cho từng câu và từng output:

- operation/program trước và sau search, top-k rank, candidate pool;
- mapping resolution, số physical mapping của từng operand;
- từng required proof: VALID / UNKNOWN / INVALID;
- lý do bị loại tại search, source-action gate, calibration, executor,
  presentation và API;
- raw unit/scale, period, PLAN/ACTUAL, entity, section, bảng và nguồn;
- action cuối, certificate hợp lệ hay không; không ghi giá trị tạm thành answer.

Báo cáo hai thống kê riêng: gate đầu tiên gây thất bại và toàn bộ proof chưa đạt.
Tránh đếm một câu nhiều lần khi lập tổng failure buckets.

Baseline search có các nhóm độc quyền:

| Nhóm | Số mẫu |
|---|---:|
| Gold operation không được xét | 4 |
| Gold operand ngoài pool@20 | 0 |
| Gold program không được tạo | 4 |
| Gold program nằm dưới top-10 | 5 |
| Gold trong top-10 bị verifier INVALID | 14 |
| Gold trong top-10 bị gate khác loại | 274 |
| Gold trong top-10 eligible | 161 |
| Tổng | 462 |

161 là số câu có gold program eligible trong beam; 28 là số winner qua public
proof gate. Hai con số đo khác nhau. Phải truy vết từng mẫu, không suy ra rằng
133 câu chỉ cần bỏ một gate là sẽ đúng.

Review nguyên nhân ưu tiên:

1. Gold/annotation có thật sự đủ bằng chứng không? Câu tổng hợp có đang hỏi năm,
   đơn vị hoặc phạm vi không có trong nguồn?
2. Bằng chứng có trong Markdown nhưng mất khi extraction/serialization?
3. Binding sai scope bảng, entity, scale, role hoặc physical uniqueness?
4. Verifier yêu cầu proof không áp dụng, hoặc diễn giải khác planner?
5. Program đúng đã có nhưng winner/reranker chọn sai?
6. Search sai operation/arity hoặc không tạo đủ tổ hợp?

Không coi mọi gold bị từ chối là lỗi verifier. UNKNOWN do nguồn thiếu phải
giữ abstain; annotation sai cần quarantine/correction ở dataset version mới.

Đầu ra: `reports/finhrm_v14_2_gate_diagnosis_v1.jsonl` và tổng hợp Markdown.

Điều kiện đạt: mọi mẫu có trạng thái gate giải thích được, thống kê đủ tổng,
có ví dụ cấu trúc tối giản và phản ví dụ cho các nhóm lỗi ưu tiên.

## 5. Phase 2 — đồng bộ quyết định và sửa lỗi cấu trúc tổng quát

Chỉ triển khai các sửa đã có nguyên nhân từ Phase 1.

### 5.1. Một hợp đồng action xuyên suốt

Audit khác biệt hiện có giữa evaluator/calibration và runtime:
evaluator tính accepted theo policy AND public proof gate, trong khi runtime
có nhánh chấp nhận nhờ full unique proof. Cần định nghĩa và test rõ sự khác biệt
này trước khi gọi chung là accepted coverage.

Thiết kế một decision object dùng chung, gồm:

- `proof_eligible`;
- `calibration_status/calibration_accepted`;
- `execution_status/certificate_status`;
- `final_action`, `acceptance_basis` và reason codes.

Giữ đường policy-calibrated hiện tại làm baseline. Nếu chọn chế độ phát hành
dựa hoàn toàn vào proof, đó phải là policy riêng có tên/version, được đánh giá
độc lập; không thay đổi ý nghĩa accepted cũ hoặc tự bật khi calibration tắt.

Scalar, multi-output, CLI, API và evaluator phải dùng cùng hợp đồng. Các chỉ số
diagnostic trước calibration vẫn được giữ riêng.

### 5.2. Binding và execution

Sửa theo tần suất/nguyên nhân thực tế, ưu tiên:

- Unit gắn đúng bảng/caption/header; hỗ trợ scale rõ ràng, không lan sang bảng khác.
- PLAN/ACTUAL và kỳ gắn theo cột vật lý, kể cả header nhiều tầng/OCR tách dòng.
- Khử extraction duplicate; giữ ambiguity khi còn hai ô vật lý hợp lệ.
- Entity/section scope dựa vào nguồn; không lấy filename/mã công ty làm shortcut.
- SUBTRACT/PERCENT_CHANGE chứng minh thứ tự kỳ và không dùng một ô cho hai kỳ.
- SUM_N chứng minh đủ thành phần, không double-count subtotal/component.
- DAG giữ giá trị hữu tỷ qua các bước; rounding được khai báo ở biên hiển thị.
- Certificate ràng buộc đầy đủ program, operand, unit, source và output.
- Mỗi ý trong multi giữ output ID/proof/citation; PARTIAL không làm mất ý đã xác minh.

Tệp trọng tâm: `symbolic_verifier.py`, `evidence_schema.py`,
`program_data.py`, `source_action_policy.py`, `document_reasoning_engine.py`,
`answer_presentation.py`, `request_answer.py`, `document_service.py`.

Test mỗi sửa bằng biến đổi company/year/value, hoán vị candidate/row/column,
thay layout và câu diễn đạt tương đương. Bổ sung phản ví dụ thiếu unit, sai entity,
hai mapping hợp lệ, zero denominator và proof/certificate bị sửa.

Điều kiện đạt: regression liên quan PASS; không phát sinh accepted-wrong;
chỉ tăng coverage ở nhóm được xác nhận có đủ bằng chứng. Nguồn thiếu vẫn abstain.

## 6. Phase 3 — đánh giá end-to-end trên báo cáo thực

Tận dụng Markdown đã chuẩn bị. Kiểm kê nguồn trước, không OCR toàn bộ corpus.

Chuẩn bị tập development đọc nguyên Markdown, không cấp gold candidate/operand
cho runtime. Review câu hỏi và expected action từ nguồn trước inference.
Có thể gồm direct fact, hai kỳ, PLAN/ACTUAL, mixed scale, parent/consolidated,
duplicate table, notice thiếu kỳ, multi-output và SUM_N arity 2–8.

Chọn cohort tách khỏi nhóm dùng sửa lỗi theo company/source và phân tầng năm/layout.
Với thử nghiệm tuyên bố unseen-year hoặc unseen-layout, phải có holdout tương ứng
thật sự; không chỉ dựa vào tên file khác nhau. Nếu nguồn hiện có không đủ độc lập,
báo giới hạn/HOLD thay vì lấy protected cũ làm validation mới.

Tách hai kiểm tra:

- Regression trên báo cáo development cũ, gồm notice 14 case.
- Generalization trên cohort mới đã khóa; không sửa code giữa lượt chạy.

Đo: end-to-end exact toàn câu, exact từng output, accepted wrong, accepted coverage,
source errors, abstention đúng theo loại, PARTIAL, certificate validity, latency
và peak RAM. Báo cáo theo operation/arity/layout và company, kèm support.

Điều kiện đạt: accepted_wrong = 0, source errors = 0, notice regression = 14/14,
mọi ANSWER có proof/certificate/citation hợp lệ; coverage được đo trên tập có
cả answerable và unanswerable. Chốt coverage target trước inference từ baseline
và mục tiêu sử dụng, không đặt sau khi đã thấy kết quả.

Nếu dùng lỗi holdout để phát triển, chuyển cohort đó thành development và
ghi lại việc chuyển vai trò; không tiếp tục gọi nó là đánh giá mù.

## 7. Phase 4 — calibration độc lập

Chỉ bắt đầu khi policy/action contract và runtime đã khóa.

- Dùng calibration cohort độc lập với tập dùng sửa code và protected test.
- Tối ưu coverage dưới ràng buộc precision đã đăng ký; giữ cận dưới Wilson và
  target 90% hiện tại nếu chưa có protocol thay thế được công bố.
- Báo cáo theo operation; ít support thì UNCALIBRATED/DISABLED.
- Không bổ sung các bản paraphrase cùng một evidence rồi coi là mẫu độc lập.
  Các câu cùng công ty/báo cáo có tương quan; báo thêm phân tích theo nhóm hoặc
  cluster bootstrap khi ước lượng uncertainty.
- Nếu policy proof-only được đề xuất ở Phase 2, đánh giá riêng cùng tập và cùng
  yêu cầu sai chấp nhận; không gộp precision của nó với policy-calibrated.

Điều kiện đạt: có operating point đáp ứng cận dưới precision và coverage target
đã khóa. Nếu không đạt, giữ HOLD; không hạ threshold chỉ để đổi số 0%.

## 8. Nhánh dự phòng — cải tiến HRM, không bắt buộc trước certification

Mặc định bỏ qua nhánh này và giữ nguyên checkpoint HRM để đánh giá toàn pipeline.
Chỉ kích hoạt khi chẩn đoán sau khi hoàn thiện tầng kiểm chứng chứng minh lỗi
proposal/neural còn đáng kể: gold thường xuyên thiếu trong danh sách hoặc lựa
chọn vẫn sai do tín hiệu neural, không phải do binding/proof/policy chưa đúng.
Program exact 82,68% hoặc accepted coverage 0% riêng lẻ không đủ để kích hoạt.

- Gold operand ngoài pool: cải thiện retrieval/embedding.
- Operand đủ nhưng thiếu program: sửa generator/arity/beam.
- Gold program có nhưng winner sai: trước hết kiểm tra tầng symbolic/chọn phương
  án có áp dụng đủ chứng cứ chưa. Chỉ khi lỗi còn thuộc ranking neural mới train
  reranker hoặc pointer/membership/count bằng hard negatives train-only.
- Operation sai: mới xem xét operation head.
- Evidence/proof thiếu trong nguồn: không giải quyết bằng train thêm.

Đối chiếu baseline scratch hiện tại với thay đổi, dùng cùng protocol và nguồn.
Ablation cần tách neural proposal, structural search, symbolic gate và calibration;
output từ neural-only chỉ phục vụ đo lường.

Chỉ khi kích hoạt nghiên cứu huấn luyện bổ sung, với cấu hình finalist mới chạy
seed 31/43/71 và báo mean/std trên cùng split;
không chỉ chọn seed tốt nhất. Giữ thuật ngữ chính xác: HRM/adapter scratch,
semantic encoder BKAI pretrained và frozen. Scratch HRM không chứng minh
encoder nền chưa từng thấy tài liệu nào.

## 9. Phase 5 — certification và LEAN tùy chọn

Sau khi development/calibration đạt, đóng băng weights, mọi policy/gate,
runtime, dependency, schema metric, data hash và chế độ inference.
Chạy protected cohort một lần theo protocol đã khóa; không tune trên kết quả.

LEAN chỉ là nhánh audit tùy chọn: khóa Lean/Mathlib version, kiểm các phép toán,
mixed scale và DAG; ghi UNAVAILABLE khi backend thiếu. Chứng minh số học không
tự chứng minh OCR, evidence selection hay ý nghĩa câu hỏi đúng.

Công bố đồng thời accuracy, coverage, accepted-wrong và support/uncertainty.
Accepted-wrong bằng 0 trên mẫu đánh giá không bảo đảm sai số thực bằng 0.

Cập nhật kiểm tra thực tế nhiều báo cáo (2026-09-08): [10 câu HKT/HAV/SSG/ASM](reports/finhrm_v14_2_practical_results_v2.md), runtime giữ nguyên reading v2. **7/9 câu có dữ liệu chọn khớp tham chiếu cơ bản; 0/10 đủ proof, 0/10 công bố; ASM thiếu đúng 1/1**. HAV ổn định khi thêm “Cho biết”; SSG chuyển sang vị trí mất đơn vị. Hỏi kế hoạch SSG 2024 lại chọn 60.847 triệu thực hiện, metadata đoạn văn bị gán nhầm kế hoạch; nguồn PDF có số KH 5.242 triệu. Chưa sửa các lỗi này trong lượt thử. Không coi nguồn development đã biết là báo cáo chưa từng thấy. Evaluator v1 nhầm thứ tự citation với operand HAV, v2 sửa cách đọc output; oracle/raw HTTP/model giữ nguyên.

Cập nhật sửa nguyên nhân metadata/framing (2026-09-08): [context fix v4](reports/finhrm_v14_2_context_fix_implementation_v4.md). Đã phục hồi ô kế hoạch SSG 5.242 triệu bị loại nhầm thành mã ghi chú, chặn header bảng lan sang prose, chuẩn hóa tiền tố câu hỏi và KH/TH. Nhóm 10 câu cũ: 7/9 → 9/9 khớp tham chiếu, proof 0 → 1/10; nhóm 4 câu bổ sung có 3/3 chọn đúng, 1 câu mơ hồ bị chặn nhưng lời hỏi lại còn thiếu. 308 test/258 subtest PASS. Current runtime `config/finhrm_v14_2_context_fix_protocol_v4.json`; HRM/nguồn/calibration giữ nguyên. Các lỗi SSG/frame trong cập nhật trước là lịch sử trước bản sửa này.

## 10. Việc nên làm ngay ở lượt triển khai kế tiếp

1. Baseline lịch sử trước dọn: `config/finhrm_v14_2_context_fix_protocol_v4.json`; [báo cáo](reports/finhrm_v14_2_context_fix_implementation_v4.md), [audit](reports/finhrm_v14_2_context_fix_delivery_v4.json). Không train lại HRM; giữ checkpoint V14.1 scratch seed31.
2. Đã xử lý nguyên nhân ô tiền có kỳ bị loại thành mã ghi chú, header bảng lan vào câu văn và tiền tố “Cho biết” làm đổi metric/identity. Không coi các lỗi này là chưa sửa hoặc tiếp tục vá theo số/công ty. Bộ 10/4 câu chấm riêng; không cộng vào 73 output lịch sử.
3. Ưu tiên còn lại là các trường hợp chọn đúng số/thuộc tính nhưng AMBIGUOUS giữa nhiều công bố. Phân biệt công bố lặp cùng fact với khác phạm vi/độ chính xác; dùng nguồn và nhóm câu thực tế để xác định guard nào cần sửa, không mặc định nới toàn bộ proof.
4. Cải thiện phản hồi cho câu kỳ mơ hồ: đã chặn công bố nhưng cần hỏi rõ năm người dùng muốn; không dùng thông báo lỗi chọn bằng chứng chung.
5. Trước kết luận tổng quát hoặc vai trò riêng của HRM, chọn cohort development độc lập với nguồn/câu dùng sửa, khóa oracle trước chạy; giữ cùng pool/executor/verifier/policy để so sánh selector. Không mở protected để phát triển.
6. Chỉ chuẩn bị calibration khi có đủ dữ liệu độc lập và điều kiện hiệu chỉnh. Không hạ gate/threshold để đổi accepted 0. Nguồn, model và dataset candidate 438 mẫu vẫn giữ nguyên; bảo toàn snapshot và báo cáo đối chứng.

Calibration DISABLED; useful coverage/generalization/certification HOLD. Không training mới.

## 11. Sửa quy trình nhập tài liệu — 2026-09-08

Chỉ dùng MinerU để trích xuất nội dung PDF, rồi hỏi đáp trên Markdown có nguồn và phân trang. Không dùng native-text hoặc công cụ đọc chữ PDF khác làm fallback. Thiếu artifact/môi trường MinerU thì dừng chuẩn bị và báo đúng nguyên nhân.

Lượt thử 8 câu ACG/AGP/ALT vừa rồi dùng sai đường nhập, đã rút toàn bộ khỏi data/markdown, config và reports hoạt động; bản gốc lưu trong archive/retired_pdf_route_20260908_v1.zip để kiểm toán. Không dùng kết quả 2/7 hoặc 0/8 của lượt này để kết luận pipeline MinerU hay chọn hướng sửa HRM/parser.

Runtime đã thay đổi khi gỡ đường đọc cũ: protocol context-fix v4 là bản lịch sử, không đại diện code sau dọn. Trước inference kế tiếp phải khóa runtime mới, chuẩn bị toàn báo cáo bằng MinerU, xác nhận artifact/provenance và chạy lại câu hỏi đúng đường nhập. Chưa chạy lại OCR/inference trong bước dọn này.


## 12. Kết quả chạy đúng MinerU — tập con đã hoàn tất

[kết quả 5 câu ACG/AGP](reports/finhrm_v14_2_mineru_ready_results_v1.md), [raw HTTP](reports/finhrm_v14_2_mineru_ready_observations_v1.json), [trạng thái toàn lượt](reports/finhrm_v14_2_mineru_rerun_execution_state_v1.json).

- Dự kiến ban đầu 3 báo cáo/225 trang là quá nặng cho thử nhanh; sau lỗi tài nguyên và phản hồi về thời gian, đã ưu tiên 5 câu trên 2 báo cáo có Markdown MinerU đầy đủ: ACG 123 trang, AGP 58 trang. ALT mới cache 16/44 trang, 3 câu ALT chưa chạy; không gộp chúng vào mẫu số.
- Kết quả: 5/5 HTTP 200 và 5/5 chọn khớp số/thang đo/chỉ tiêu/kỳ/nguồn cơ bản; proof eligible 0/5, accepted 0/5. ACG chọn 4.608,7 tỷ; AGP chọn đúng 930 tỷ KH 2026, 844,7 tỷ TH 2025, và cặp 844,7/793,3 tỷ cho phép so sánh hai năm. Không gọi các số này là đáp án đã công bố.
- ACG còn UNKNOWN về column role/logical cell. AGP AMBIGUOUS giữa công bố cần truy vết cụ thể; chưa chứng minh nên nới guard nào. Calibration giữ DISABLED.
- Protocol sau dọn đã khóa riêng: `config/finhrm_v14_2_mineru_ready_protocol_v1.json`; model/QA logic không sửa trong lượt inference. Tham chiếu giữ nguyên trước inference. Không dùng kết quả native đã rút làm baseline.
- WSL đã gặp lỗi hết RAM (kernel OOM ngắt MinerU); phải tiếp tục cache và tách môi trường MinerU/HRM. Hiện không để OCR chạy nền tiếp. Lần thử nhanh sau ưu tiên artifact MinerU sẵn có; kiểm kê và nói rõ số trang/thời gian chuẩn bị trước khi mở rộng nguồn.
- Việc tiếp theo: khai thác hai báo cáo đã sẵn sàng để chẩn đoán xác nhận bằng chứng/phản hồi; ALT là phần chưa hoàn tất, không tự coi bộ 8 câu đã chạy xong. Không training hoặc nới gate dựa trên 5 câu này.

## 13. Chẩn đoán 5 câu MinerU đã chạy

[Phân tích nguyên nhân](reports/finhrm_v14_2_mineru_ready_diagnosis_v2.md), [dấu vết](reports/finhrm_v14_2_mineru_ready_diagnosis_v2.json). Chỉ replay symbolic từ operand đã lưu; không OCR/neural inference, không sửa runtime.

- ACG: caption “Kết quả Hợp nhất…” thay caption “Kết quả kinh doanh…”, giữ scope nhưng làm mất ACTUAL; role thiếu kéo theo logical-cell key thiếu. Hai UNKNOWN là hệ quả chung, không phải hai lỗi độc lập.
- Đính chính AGP tra cứu: một ô tiền VALID và ô 10,10% UNKNOWN tạo mơ hồ giả. Slot `currency_or_source` nhận cả percent, rồi mapping đếm cả UNKNOWN. Trong phép thử trên bản sao contract, ràng buộc currency khôi phục UNIQUE cho đúng hai câu này; chưa phải sửa runtime.
- AGP so sánh: 3 vị trí năm 2025 và 2 vị trí năm 2024, khác scale/độ làm tròn/nhãn/section; chưa được chứng minh tương đương. 51,4 tỷ theo bảng làm tròn và 51,413 tỷ theo bảng triệu đồng cần đối chiếu cùng metric/scope trước khi gộp.
- Calibration DISABLED vẫn là chặn độc lập. Ưu tiên sửa loại đại lượng và ngữ cảnh tiêu đề, rồi xử lý tương đương công bố; không train hoặc mở rộng OCR theo các ca này. Giữ nguyên 5 câu và phản ví dụ cho lần kiểm tra sau.