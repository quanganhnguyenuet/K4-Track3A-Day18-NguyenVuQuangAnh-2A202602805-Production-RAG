# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** Nguyễn Vũ Quang Anh
**Khóa:** K4 - Track 3B

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|---|---:|---:|---:|
| Faithfulness | 0.8333 | 0.8500 | +0.0167 |
| Answer Relevancy | 0.8146 | 0.8639 | +0.0493 |
| Context Precision | 0.9250 | 0.9500 | +0.0250 |
| Context Recall | 0.9250 | 0.9333 | +0.0083 |

Pipeline production vượt baseline ở cả bốn metric. Mức cải thiện lớn nhất nằm ở Answer Relevancy (+0.0493), cho thấy bước hybrid retrieval, parent-level reranking và prompt trả lời trực tiếp đã giúp câu trả lời bám sát câu hỏi hơn. Context Recall chỉ tăng nhẹ (+0.0083), vì vậy retrieval cho câu hỏi nhiều điều kiện và nhiều tài liệu vẫn là điểm cần tối ưu tiếp.

> Lưu ý: `ragas_report.json` hiện chỉ lưu metric và chẩn đoán, không lưu nguyên văn answer/context của từng câu. Vì vậy mục **Got** bên dưới mô tả chính xác hiện tượng do RAGAS ghi nhận, không giả định lại câu chữ của câu trả lời đã sinh.

## Bottom-5 Failures

### #1 — Câu hỏi tổng hợp phép năm và lương

- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024, nhân viên có 18 ngày phép (15 ngày cơ bản + 3 ngày thâm niên); lương Senior P3-P4 nằm trong khoảng 20-35 triệu VNĐ/tháng.
- **Got:** RAGAS đánh dấu `faithfulness = 0.25`; câu trả lời chỉ được context hỗ trợ một phần hoặc đã kết hợp thêm claim chưa có đủ bằng chứng.
- **Worst metric:** Faithfulness — 0.25.
- **Error Tree:** Output chưa được hỗ trợ đầy đủ → Context cần ghép từ hai tài liệu → Query gồm hai ý độc lập → Retrieval/reranking chưa bảo đảm giữ đủ cả hai nguồn.
- **Root cause:** Đây là câu hỏi multi-hop. Thông tin phép nằm trong `nghi_phep_nam_v2024.md`, còn khung lương nằm trong `bang_luong_2024.md`. Một lần rerank top-3 có thể ưu tiên mạnh một nhánh và thiếu bằng chứng cho nhánh còn lại.
- **Suggested fix:** Tách query thành hai sub-query (`phép năm theo thâm niên` và `khung lương Senior`), retrieve riêng từng nhánh, sau đó hợp nhất context và yêu cầu LLM trả lời đủ hai phần.

### #2 — Tính phí tạm ứng quá hạn

- **Question:** Nhân viên tạm ứng 15 triệu, sau 20 ngày mới thanh toán. Bị phạt bao nhiêu?
- **Expected:** Quá hạn 5 ngày; mức phí 2%/tháng trên 15 triệu là 300.000 VNĐ/tháng, tính pro-rata 5 ngày khoảng 50.000 VNĐ.
- **Got:** RAGAS đánh dấu `faithfulness = 0.25`; phần tính toán hoặc cách quy đổi theo số ngày chưa được context hỗ trợ rõ ràng.
- **Worst metric:** Faithfulness — 0.25.
- **Error Tree:** Output có phép tính → Context chỉ nêu hạn 15 ngày và mức 2%/tháng → Query yêu cầu suy luận số học → LLM tự tính nhưng không trình bày đủ căn cứ.
- **Root cause:** Tài liệu nguồn nêu mức phí theo tháng nhưng không ghi trực tiếp kết quả pro-rata cho 5 ngày. Đây là bài toán kết hợp retrieval với calculation, không chỉ là trích xuất văn bản.
- **Suggested fix:** Thêm calculator/post-processing có công thức kiểm soát: `15.000.000 × 2% × 5/30 = 50.000 VNĐ`; yêu cầu câu trả lời trình bày thời hạn, số ngày quá hạn, mức phí tháng và kết quả pro-rata.

### #3 — Quy trình mua laptop 30 triệu

- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Giám đốc phòng ban phê duyệt; cần xác nhận cấu hình kỹ thuật của phòng CNTT; đồng thời phải có ít nhất 3 báo giá vì giá trị trên 10 triệu.
- **Got:** RAGAS đánh dấu `faithfulness = 0.50`; câu trả lời có ít nhất một chi tiết chưa được context hỗ trợ đầy đủ hoặc thiếu liên kết giữa ba điều kiện.
- **Worst metric:** Faithfulness — 0.50.
- **Error Tree:** Output gồm nhiều yêu cầu → Context đúng nhưng nằm ở các đoạn khác nhau của cùng tài liệu → Query yêu cầu tổng hợp → Generation có thể bỏ sót hoặc thêm điều kiện.
- **Root cause:** Câu hỏi cần tổng hợp ba rule: ngưỡng phê duyệt, xác nhận CNTT và số báo giá. Nếu retrieval chỉ lấy child chunk chứa bảng phê duyệt hoặc chỉ lấy đoạn yêu cầu CNTT thì bằng chứng không đầy đủ.
- **Suggested fix:** Khi một child của tài liệu mua sắm được chọn, mở rộng sang parent/sibling chunks cùng `source`; dùng prompt checklist bắt buộc trả lời đủ `người phê duyệt`, `xác nhận CNTT`, `báo giá`.

### #4 — Nghỉ không lương 20 ngày

- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** Nghỉ 16-30 ngày cần CEO phê duyệt; vì nghỉ trên 14 ngày, nhân viên phải tự đóng phần bảo hiểm của mình.
- **Got:** RAGAS đánh dấu `faithfulness = 0.50`; câu trả lời có thể nêu đúng cấp phê duyệt nhưng phần điều kiện bảo hiểm chưa được context chứng minh đầy đủ, hoặc ngược lại.
- **Worst metric:** Faithfulness — 0.50.
- **Error Tree:** Output đúng một phần → Context liên quan nằm ở hai đoạn khác nhau → Query ngắn nhưng ground truth có thêm ngoại lệ bảo hiểm → Parent expansion chưa luôn giữ cả hai đoạn.
- **Root cause:** Quy tắc phê duyệt và quy tắc bảo hiểm nằm ở hai đoạn trong `nghi_phep_khong_luong.md`. Reranking theo một đoạn ngắn dễ ưu tiên thông tin phê duyệt và bỏ mất hệ quả bảo hiểm.
- **Suggested fix:** Group theo `source + parent_id`, mở rộng thêm sibling cùng nguồn khi phát hiện câu hỏi chứa thời lượng; prompt phải kiểm tra cả ngưỡng phê duyệt và các hệ quả đi kèm thời lượng.

### #5 — Chu kỳ đổi mật khẩu

- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Chính sách hiện hành v2.0 yêu cầu đổi mỗi 120 ngày; chính sách cũ v1.0 yêu cầu 90 ngày nhưng đã bị thay thế.
- **Got:** RAGAS đánh dấu `context_precision = 0.50`; context chứa cả tài liệu liên quan và tài liệu/version gây nhiễu.
- **Worst metric:** Context Precision — 0.50.
- **Error Tree:** Output có thể đúng → Context chứa hai con số 90 và 120 ngày → Query không chỉ rõ phiên bản → Retrieval lexical/dense đều xem hai bản là liên quan.
- **Root cause:** Hai tài liệu v1.0 và v2.0 có nội dung gần giống nhau. Hybrid search có xu hướng lấy cả hai nhưng chưa áp dụng metadata `version/status/effective_date` để ưu tiên chính sách hiện hành.
- **Suggested fix:** Trích xuất metadata phiên bản, trạng thái và ngày hiệu lực khi ingest; filter/boost `status=current`, đồng thời giữ bản cũ làm bằng chứng giải thích nhưng xếp sau bản hiện hành.

## Case Study (cho presentation)

**Question chọn phân tích:** Bao lâu phải đổi mật khẩu một lần?

**Error Tree walkthrough:**

1. **Output đúng?** Có khả năng đúng về con số 120 ngày, nhưng evaluation cho thấy context đi kèm chưa tinh gọn.
2. **Context đúng?** Đúng một phần; cả `mat_khau_v1.md` và `mat_khau_v2.md` đều được xem là liên quan, nhưng bản v1.0 đã bị thay thế.
3. **Query rewrite OK?** Chưa đủ. Query chỉ có “đổi mật khẩu” nên khớp mạnh với cả hai phiên bản; cần bổ sung ý định “quy định hiện hành/mới nhất”.
4. **Fix ở bước:** M5 trích xuất `version`, `status`, `effective_date`; M2 áp dụng metadata filter/boost; M3 ưu tiên tài liệu hiện hành và dùng bản cũ chỉ để giải thích thay đổi.

**Nếu có thêm 1 giờ, sẽ optimize:**

- Lưu `answer`, `contexts` và điểm từng metric theo từng câu vào report để failure analysis có thể đối chiếu nguyên văn.
- Bổ sung version-aware retrieval cho các chính sách có nhiều phiên bản.
- Thêm query decomposition cho câu hỏi multi-hop và calculator cho câu hỏi số học.
- Chạy ablation `raw vs enrichment`, `dense vs hybrid`, `with vs without reranker` để đo đóng góp thực tế của từng module.
