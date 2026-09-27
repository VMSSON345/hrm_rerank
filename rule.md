# Quy tắc phát triển FinHRM

## 0. Nguồn tài liệu bắt buộc

PDF → MinerU → Markdown có provenance/phân trang → hỏi đáp. MinerU là công cụ duy nhất trích xuất nội dung PDF. Không dùng native-text hay đổi sang công cụ đọc chữ PDF khác để thử nhanh. PDFium đi kèm MinerU chỉ dùng kiểm tra số trang, không tạo evidence. Nếu MinerU chưa sẵn sàng thì báo trạng thái chuẩn bị, không tự thay đường nhập. Không dùng kết quả của đường nhập đã loại bỏ để chấm hoặc điều chỉnh hệ thống.

## 0.1. Policy áp dụng từ 2026-09-10

Theo yêu cầu người dùng, các lượt phát triển/đánh giá tiếp theo chỉ dùng B và so B trước/sau; không tiếp tục so A/B hoặc tối ưu calibration để mở gate A. Code, test và kết quả A giữ làm lịch sử, dọn sau. Test riêng A không là điều kiện hoàn tất lượt B; không sửa/xóa kỳ vọng để tuyên bố PASS. Test phần dùng chung và đường công bố B vẫn áp dụng.

B giữ kiểm chứng bằng chứng, executor, certificate và tính đúng của phản hồi. Calibration chỉ là chẩn đoán, không chặn B. Phải xác nhận policy B trong cấu hình và response thực tế; chọn B không đồng nghĩa đã chứng minh khả năng tổng quát hoặc production. Xem [kế hoạch hiện hành](ke_hoach_B_cong_bo_lap_va_do_chinh_xac_v1.md).

## 0.2. Đơn vị tiền tệ mặc định — cập nhật 2026-09-11

Theo yêu cầu người dùng: khi số liệu tiền tệ không ghi đơn vị, mặc định là đồng (VND), thang đo 1. Đơn vị được ghi rõ trong ô/hàng/cột/chú thích vẫn được ưu tiên; đơn vị mâu thuẫn không phải đơn vị thiếu. Không áp dụng cho tỷ lệ, số lượng, cổ phiếu, sản lượng hoặc các đại lượng khác. Ghi `unit_assumption=missing_monetary_unit_vnd_v1` trong bằng chứng và citation; phản hồi nêu rõ đơn vị mặc định. Đây là quy ước sử dụng, không phải đơn vị được xác nhận từ PDF. Các kiểm tra chỉ tiêu, kỳ, KH/TH, doanh nghiệp, phạm vi và số học vẫn giữ. Kết quả lịch sử và tham chiếu chưa xác định đơn vị giữ nguyên; đánh giá có giả định đơn vị phải ghi riêng.

## 0.3. Câu hỏi so sánh mở — theo yêu cầu hiện hành của người dùng

Khi câu hỏi nêu rõ một chỉ tiêu và hai năm, hoặc KH/TH của cùng năm, nhưng chỉ yêu cầu “so sánh”, có thể trả lời nhiều phần: số liệu của từng kỳ/loại và chênh lệch. Với hai năm, ghi rõ chiều năm sau trừ năm trước; với KH/TH, ghi rõ thực hiện trừ kế hoạch. Không tự chọn một loại số liệu rồi bỏ phần còn lại.

Mỗi phần phải qua cùng kiểm chứng bằng chứng, executor và certificate của B. Nếu một phần thiếu dữ liệu, vẫn hiển thị phần đã xác minh và nói rõ phần còn thiếu; không tính chênh lệch từ một đầu vào chưa xác minh. Câu hỏi đã chỉ rõ phép tính giữ nguyên phép tính đó. Không tự đặt năm, doanh nghiệp hoặc phạm vi khi câu hỏi chưa xác định được các chiều này.
## 1. Mục tiêu và phạm vi

Mục tiêu chính là hệ thống đọc đúng báo cáo, lấy đúng thông tin và trả lời hữu ích cho người dùng. HRM, parser, symbolic verifier và calibration là các thành phần phục vụ mục tiêu này.

Đây là quy tắc định hướng phát triển và kiểm chứng. Việc tạo file này không đồng nghĩa các hành vi dưới đây đã được triển khai trong mã. Chỉ công bố một khả năng là đã hỗ trợ khi đã chạy kiểm tra tương ứng.

- Ưu tiên kết quả trên câu hỏi thực tế hơn số lượng rule, test hoặc proof đã thêm.
- Không mặc định phải giữ HRM bằng mọi giá. Giữ hoặc thay cách sử dụng HRM dựa trên lợi ích đo được.
- Không viết lại toàn bộ kiến trúc khi chưa xác định được chặng gây lỗi.
- Mọi thay đổi phải nằm trong phạm vi công việc người dùng đã giao.

## 2. Định nghĩa “đọc tốt”

Một kết quả được coi là đọc đúng khi khớp đầy đủ:

1. Chỉ tiêu được hỏi: doanh thu, lợi nhuận trước thuế/sau thuế, tài sản…
2. Năm hoặc kỳ được hỏi.
3. Loại số liệu: thực hiện, kế hoạch, kế hoạch điều chỉnh hoặc dự báo.
4. Doanh nghiệp và phạm vi báo cáo: riêng, hợp nhất, công ty con…
5. Giá trị, dấu âm/dương, đơn vị và thang đo.
6. Vị trí nguồn có thể kiểm tra: tài liệu, trang, hàng/cột hoặc đoạn văn.

Trùng con số nhưng sai một thuộc tính trên không được tính là đúng. Số điện thoại, mã số, tỷ lệ hoặc số của một doanh nghiệp khác không được dùng thay cho chỉ tiêu tài chính đang hỏi.

Khi đọc đúng nhưng chưa được phép công bố đáp án, phải ghi riêng trạng thái đó; không đánh đồng với đọc sai hoặc thiếu dữ liệu.

## 3. Quy ước hiểu câu hỏi tự nhiên

### 3.1. Loại số liệu mặc định

- KH là kế hoạch; TH là thực hiện.
- Với câu hỏi thông thường như “Doanh thu năm 2023 là bao nhiêu?”, mặc định ý hỏi là số thực hiện.
- Câu trả lời phải thể hiện đang dùng số thực hiện, để người dùng có thể kiểm tra hoặc sửa ý hỏi.
- Nếu câu hỏi nêu kế hoạch, dự kiến, dự báo hoặc loại số liệu khác, yêu cầu đó thay thế mặc định.
- Nếu người dùng chủ động yêu cầu phân biệt hoặc so sánh KH/TH, phải giữ các loại số liệu tách biệt.

Mặc định này chỉ xác định ý hỏi. Nó không cho phép gán TH cho một ô nguồn chưa rõ loại số liệu, không cho phép suy TH vì năm đã qua, và không cho phép chọn theo số khớp gold.

### 3.2. Khi nào hỏi lại

- Không bắt người dùng biết số mục, trang hoặc luôn phải gõ “TH” mới hỏi được.
- Tự sử dụng ngữ cảnh tài liệu để giải quyết những thông tin đã rõ.
- Chỉ hỏi lại khi còn nhiều cách hiểu hợp lý ảnh hưởng đến đáp án: doanh nghiệp, kỳ, riêng/hợp nhất hoặc số liệu có nguồn chưa thống nhất.
- Không hỏi lại KH/TH chỉ vì báo cáo có cả hai, nếu quy ước mặc định hoặc câu hỏi đã xác định loại cần lấy.
- Hỏi ngắn, đúng điểm còn thiếu; không yêu cầu người dùng chẩn đoán lỗi parser hoặc chọn candidate nội bộ.

Ví dụ: nếu chưa rõ riêng/hợp nhất, hỏi “Bạn muốn số riêng công ty mẹ hay số hợp nhất?” thay vì yêu cầu người dùng tìm mục trong PDF.

## 4. Vai trò của từng thành phần

| Thành phần | Trách nhiệm |
|---|---|
| Đọc tài liệu/parser | Giữ cấu trúc bảng, hàng/cột, header, chú thích, đơn vị và tọa độ nguồn |
| Truy hồi | Đưa bằng chứng liên quan vào danh sách ứng viên |
| Hiểu câu hỏi | Xác định chỉ tiêu, kỳ, loại số liệu, phạm vi và phép tính cần thực hiện |
| Bộ chọn/HRM | Chọn bằng chứng và chương trình phù hợp trong những gì hệ thống đã đọc được |
| Symbolic verifier/executor | Kiểm tra tính phù hợp của lựa chọn và thực hiện phép tính chính xác |
| Policy/calibration | Quyết định điều kiện công bố đáp án theo chính sách đã xác định |
| Trình bày | Trả lời dễ hiểu, dẫn nguồn và giải thích đúng lý do khi chưa trả lời được |

Không dùng HRM để che việc mất thông tin nguồn. Không dùng verifier để tạo số liệu hoặc tự chọn theo gold. Không coi câu thông báo khó hiểu là lỗi neural khi nó do template/policy tạo ra.

## 5. Quy tắc sử dụng ngữ cảnh và bằng chứng

- Header cụ thể của cột được ưu tiên trước nhãn chung của bảng.
- Chỉ kế thừa role/scope/unit từ tiêu đề hoặc chú thích khi xác định được chúng áp dụng cho bảng/cột đó.
- Phải xử lý ranh giới giữa bảng, mục, trang, đoạn văn và doanh nghiệp. Không để ngữ cảnh lan sang bảng khác.
- Mỗi kết luận kế thừa phải lưu căn cứ nguồn, gồm trang, dòng/nội dung và định danh hoặc hash phù hợp.
- Không coi nhiều vị trí nguồn là mơ hồ chỉ vì số lượng vị trí lớn hơn một. Kiểm tra chúng có cùng mô tả một fact hay không.
- Chỉ đồng nhất công bố lặp khi chỉ tiêu, kỳ, loại số liệu, entity/scope, đơn vị và độ chính xác tương thích; giữ các vị trí dẫn chứng.
- Không âm thầm làm tròn, đổi thang đo hoặc chọn một bên khi các công bố chưa thống nhất.
- Nếu nguồn có số đúng nhưng parser gắn nhầm hàng/cột, sửa chặng đọc/binding. Không sửa gold để hợp thức hóa kết quả của parser.
- Không đưa mã công ty, số trang, giá trị đáp án hoặc case ID vào điều kiện runtime để làm một ca cụ thể vượt gate. Những thông tin này được dùng trong fixture, dữ liệu tham chiếu và báo cáo truy vết, không làm ngoại lệ thuật toán.

## 6. Quy trình xử lý một câu trả lời sai

Thực hiện theo thứ tự và dừng việc sửa ở đúng chặng gây lỗi:

1. **Xác nhận nguồn:** PDF/báo cáo thực sự có thông tin được hỏi hay không? Nếu cần xét bố cục, xem trang gốc thay vì chỉ đọc chuỗi OCR.
2. **Kiểm tra trích xuất:** số đúng đã được đọc chưa, và metadata hàng/cột/kỳ/role/scope/unit có đúng không?
3. **Kiểm tra truy hồi và cắt danh sách:** bằng chứng đúng có trong danh sách đưa cho bộ chọn không? Nếu bị loại, xác định bước loại và lý do.
4. **Kiểm tra lựa chọn:** khi bằng chứng đúng còn trong danh sách, HRM/bộ chọn có chọn đúng không? Ràng buộc câu hỏi có được dùng trong search hay chỉ đến lúc từ chối mới kiểm tra?
5. **Kiểm tra công bố và hiển thị:** lựa chọn đúng và proof đủ có bị policy chặn không? Thông báo có nói đúng nguyên nhân không?

Mỗi sửa đổi phải trả lời được bốn câu:

- Sai ở đâu, bằng chứng nào xác nhận?
- Vì sao thay đổi này giải quyết nguyên nhân đó?
- Ca đúng nào được phục hồi và phản ví dụ nào vẫn bị chặn?
- Điều gì vẫn chưa giải quyết được?

Không chuyển ngay sang training khi chưa hoàn tất việc phân loại nguyên nhân. Không kết luận HRM vô ích chỉ từ số câu bị policy từ chối.

## 7. Chạy thử nhanh trước khi đánh giá toàn bộ

- Mặc định bắt đầu bằng 6–10 câu hỏi trên ít nhất 3 báo cáo; bao gồm tra cứu số, so sánh hai kỳ và một trường hợp thiếu/mơ hồ dữ liệu.
- Dùng câu hỏi tự nhiên. Không chỉ chọn những câu đã được viết theo cấu trúc mà parser dễ xử lý.
- Xác nhận đáp án tham chiếu từ nguồn trước khi chấm; chưa chắc chắn thì ghi chưa xác định, không dùng làm oracle đúng/sai.
- Với lỗi trọng tâm, thêm biến thể diễn đạt lại và phản ví dụ gần: đổi năm, KH/TH, trước/sau thuế hoặc riêng/hợp nhất.
- Giữ riêng kết quả câu hỏi gốc, câu đã làm rõ và câu bổ sung. Không cộng chúng vào cùng mẫu số để tạo cảm giác coverage tăng.
- Đọc phản hồi cuối cùng mà người dùng nhận được, không chỉ kiểm tra log nội bộ.
- Khi sửa hiển thị mà không đổi inference/quyết định, ưu tiên fixture/envelope đã lưu; không mặc định chạy lại model toàn bộ.
- Chạy hồi quy rộng và replay toàn bộ khi bản sửa đã chứng minh tác dụng trên ca trọng tâm và cần kiểm tra ảnh hưởng của logic dùng chung hoặc chốt phiên bản.
- Nếu đã có contract kiểm chứng bắt buộc áp dụng, thực hiện đủ. Không đổi oracle hoặc bỏ gate sau khi thấy kết quả để biến FAIL thành PASS.

Không tạo thêm một vòng audit lớn nếu nó không giúp giải quyết một nghi vấn cụ thể hoặc đáp ứng điều kiện chốt phiên bản.

## 8. So sánh HRM với bộ chọn đơn giản

### 8.1. Điều kiện công bằng

- Với từng câu hỏi, hai bộ chọn phải dùng cùng danh sách bằng chứng, metadata và định danh nguồn. Lưu danh sách này để tái lập.
- Bộ chọn đơn giản phải có quy tắc rõ: lọc theo ý hỏi đã xác định, xếp hạng bằng các đặc trưng được phép và xử lý hòa điểm một cách xác định.
- Chỉ phần lựa chọn/xếp hạng được khác nhau. Giữ chung executor, verifier, policy và cách chấm.
- Không cung cấp gold, vị trí đáp án hoặc metadata được sửa thủ công chỉ cho một phía.
- Đo riêng thời gian lựa chọn và thời gian toàn pipeline; ghi điều kiện thiết bị, cache và lần khởi tạo.
- Kiểm tra đổi cách diễn đạt. Phân biệt thử nghiệm giữ nguyên pool với thử nghiệm end-to-end có truy hồi lại.

### 8.2. Các chỉ số phải báo cáo

1. Bằng chứng đúng có mặt trong pool bao nhiêu câu?
2. Trong các câu có bằng chứng đúng, mỗi bộ chọn chọn đúng bao nhiêu?
3. Có chọn đúng đồng thời metric/kỳ/role/scope/unit không?
4. Đổi cách hỏi tương đương có làm đổi sai kết quả không?
5. Có bao nhiêu câu thực sự trả lời đúng, bao nhiêu câu trả lời sai và bao nhiêu câu từ chối?
6. Thời gian lựa chọn và thời gian phản hồi thực tế là bao nhiêu?

### 8.3. Quyết định sau so sánh

- Tiếp tục đầu tư vào HRM khi nó tạo lợi ích rõ trên độ đúng, độ ổn định hoặc chi phí xử lý với cùng điều kiện.
- Nếu bằng chứng đúng có trong pool nhưng HRM chọn sai, kiểm tra search/ranking và sự phù hợp của dữ liệu huấn luyện; chỉ training khi lỗi thực sự thuộc phần học được.
- Nếu bộ chọn đơn giản tương đương hoặc tốt hơn, ưu tiên phương án đơn giản hoặc thu hẹp HRM vào nhóm tác vụ có lợi ích đo được.
- Không kết luận từ vài câu thuận lợi, từ tập đã dùng để sửa hoặc từ chênh lệch nhỏ của một lần chạy. Trước kết luận tổng quát, cần tập kiểm tra tách khỏi dữ liệu dùng để điều chỉnh; không dùng protected để phát triển.

## 9. Quy tắc trả lời và từ chối

- Khi đủ điều kiện công bố: trả lời trực tiếp bằng số liệu, nêu chỉ tiêu/kỳ/loại số liệu, đơn vị và nguồn. Với phép tính, hiển thị công thức và các số đầu vào.
- Khi thực sự thiếu dữ liệu: nói rõ thiếu thông tin gì trong tài liệu đang dùng; không khẳng định doanh nghiệp không có thông tin đó ở mọi nguồn.
- Khi câu hỏi chưa rõ: hỏi đúng điểm cần làm rõ, không dùng thông báo lỗi kỹ thuật chung.
- Khi lựa chọn sai role/kỳ/scope: nói hệ thống chưa xác định được bằng chứng phù hợp; không tự kết luận báo cáo mâu thuẫn.
- Chỉ nói nguồn mâu thuẫn khi đã có các bằng chứng cùng ý nghĩa nhưng không thống nhất.
- Khi proof đủ nhưng policy/calibration chưa cho công bố: thông báo đúng trạng thái này, không nói “không đủ bằng chứng”. Diễn đạt dễ hiểu cho người dùng; chi tiết mã trạng thái để trong phần chẩn đoán.
- Không tắt kiểm chứng bằng chứng, hạ ngưỡng hoặc đổi tiêu chí sau khi xem kết quả chỉ để có thêm đáp án. Với B đã chọn tại mục 0.1, calibration không là điều kiện công bố; các điều kiện dữ liệu, executor, certificate và phản hồi vẫn phải được xác định và kiểm tra.
- Số liệu chẩn đoán chưa được chấp nhận không được trình bày như đáp án đã xác minh. Nếu cần hiển thị để kiểm tra, phải tách và ghi rõ trạng thái.

## 10. Điều kiện kết thúc một lượt sửa

Chỉ gọi một lỗi đã được giải quyết khi:

- Ca lỗi được tái hiện và nguyên nhân được xác định.
- Bản sửa làm ca đó đạt hành vi mục tiêu; có phản ví dụ bảo vệ hành vi không nên thay đổi.
- Đã chạy các kiểm tra phù hợp với phạm vi thay đổi và không còn regression chưa giải thích trong phần cần bảo toàn.
- Đã xem phản hồi người dùng thực sự nhận được nếu thay đổi ảnh hưởng luồng trả lời.
- Báo cáo nêu rõ trước/sau, lợi ích, số câu thử và giới hạn. Liên kết đến câu hỏi, nguồn và output có thể kiểm tra.

Nếu chỉ cải thiện metadata hoặc thông báo nhưng chưa phục hồi câu trả lời, ghi đúng như vậy. Nếu ca trọng tâm vẫn sai, ghi “chưa giải quyết”, không gọi là đã sửa xong chỉ vì test kỹ thuật PASS.

Không đánh đồng:

- HTTP 200 với trả lời đúng.
- Đủ proof với được policy chấp nhận.
- Từ chối toàn bộ với hệ thống hữu ích.
- Test PASS với khả năng tổng quát hóa.
- Kết quả toàn pipeline với đóng góp riêng của HRM.

Sau một thay đổi có ý nghĩa, cập nhật `next.md` theo kết quả thực tế. Hướng tiếp theo phải xuất phát từ lỗi hoặc nhu cầu còn lại, không chỉ từ việc còn thêm được rule/proof.
