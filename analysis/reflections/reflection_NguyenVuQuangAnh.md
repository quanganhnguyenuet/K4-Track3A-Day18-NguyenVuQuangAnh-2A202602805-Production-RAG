# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** Nguyễn Vũ Quang Anh
**Khóa:** K4 - Track 3A
**Ngày hoàn thành:** 05/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|---|---|---|---|
| Semantic và hierarchical chunking | M1 | `chunk_semantic()`, `chunk_hierarchical()` | Với cùng toàn bộ corpus, basic tạo 51 chunks, độ dài trung bình 410 ký tự; semantic với threshold 0.85 tạo 208 chunks, trung bình 99 ký tự. Semantic giữ ranh giới câu/chủ đề nhưng có nguy cơ over-fragmentation (chunk ngắn nhất chỉ 6 ký tự). Pipeline production vì vậy dùng hierarchical chunking: index child nhỏ để tăng precision, sau đó khôi phục parent để tạo câu trả lời đủ ngữ cảnh. Khi xử lý từng tài liệu, hệ thống tạo 26 parents và 100 children từ 26 tài liệu có text. |
| BM25 + Dense fusion | M2 | `segment_vietnamese()`, `BM25Search.search()`, `DenseSearch.search()`, `reciprocal_rank_fusion()` | BM25 bắt tốt từ khóa, con số và tên riêng; BGE-M3 bắt quan hệ ngữ nghĩa. RRF hợp nhất thứ hạng mà không cần chuẩn hóa hai loại score khác thang đo. Sau khi kết hợp hybrid retrieval với các cải tiến phía sau, Context Precision tăng từ 0.9250 lên 0.9500 và Context Recall tăng từ 0.9250 lên 0.9333. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker._load_model()`, `CrossEncoderReranker.rerank()` | BGE reranker đọc đồng thời query và document nên chính xác hơn cosine similarity nhưng tốn chi phí tính toán. Trên môi trường hiện tại, lần load model mất khoảng 11.8 giây; sau khi warm-up, rerank 3 documents trung bình khoảng 228 ms (min 189 ms, max 301 ms). Pipeline gom các child cùng parent trước khi rerank để top-3 không bị trùng nội dung. |
| RAGAS 4 metrics | M4 | `evaluate_ragas()`, `failure_analysis()` | Final production đạt Faithfulness 0.8500, Answer Relevancy 0.8639, Context Precision 0.9500 và Context Recall 0.9333; cả bốn đều cao hơn baseline. Answer Relevancy tăng nhiều nhất (+0.0493). Failure analysis cho thấy điểm yếu còn lại tập trung ở câu hỏi multi-hop, phép tính pro-rata và xung đột nhiều phiên bản chính sách. |
| Contextual enrichment / HyQA | M5 | `_enrich_single_call()`, `enrich_chunks()` | M5 chạy sau M1 nhưng trước M2 ở runtime. Một API call tạo context, summary, hypothesis questions và metadata. Nội dung sinh bởi AI chỉ dùng làm retrieval text; reranker và LLM trả lời từ parent text gốc để tránh enrichment hallucination làm giảm faithfulness. Cách tách retrieval text và answer context là thay đổi quan trọng giúp production vượt baseline. |

### Kết quả so sánh cuối cùng

| Metric | Naive Baseline | Production | Δ |
|---|---:|---:|---:|
| Faithfulness | 0.8333 | 0.8500 | +0.0167 |
| Answer Relevancy | 0.8146 | 0.8639 | +0.0493 |
| Context Precision | 0.9250 | 0.9500 | +0.0250 |
| Context Recall | 0.9250 | 0.9333 | +0.0083 |

Điều tôi rút ra là thêm nhiều module không tự động làm hệ thống tốt hơn. Ở lần chạy đầu, production thấp hơn baseline ở cả bốn metric vì enrichment text được đưa thẳng vào answer context và nhiều child của cùng parent chiếm top-k. Sau khi tách enrichment cho retrieval, khôi phục parent text sạch và gom candidate theo parent, production mới cải thiện ổn định.

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

### 1. Virtual environment bị hỏng sau khi đổi tên

- **Exact error:** `Fatal error in launcher: Unable to create process using '...\.venv311\Scripts\python.exe ...\.venv\Scripts\pytest.exe': The system cannot find the file specified.`
- **Nguyên nhân gốc rễ:** Các launcher như `pytest.exe` lưu đường dẫn tuyệt đối đến Python tại thời điểm tạo environment. Việc đổi tên `.venv311` thành `.venv` khiến launcher vẫn trỏ đến đường dẫn cũ.
- **Quá trình debug:** Kiểm tra `Get-Command python,pytest`, đọc `.venv\pyvenv.cfg`, đối chiếu đường dẫn embedded trong thông báo lỗi và xác nhận `.venv311\Scripts\python.exe` không còn tồn tại.
- **Cách giải quyết:** Tạo lại `.venv` trực tiếp bằng CPython 3.11, cài dependencies từ `requirements.txt` và chạy test bằng `python -m pytest` để chắc chắn dùng đúng interpreter.

### 2. NumPy dùng bản MinGW không phù hợp production

- **Exact warning:** `Numpy built with MINGW-W64 on Windows 64 bits is experimental... CRASHES ARE TO BE EXPECTED.`
- **Nguyên nhân gốc rễ:** Environment cũ dùng Python 3.13 từ Conda cùng NumPy 1.26.4 build MinGW, trong khi repository định hướng Python 3.11.
- **Quá trình debug:** In `sys.executable`, `sys.base_prefix`, phiên bản Python/NumPy và vị trí package. Kết quả cho thấy `.venv` không khớp với `.python-version`.
- **Cách giải quyết:** Chuyển sang CPython 3.11 x64 và tạo environment sạch. Model cache được giữ ngoài `.venv`, vì vậy không cần tải lại sau khi rebuild environment.

### 3. Lỗi cú pháp chặn pytest collection

- **Exact error:** `SyntaxError: '(' was never closed` tại `src/m1_chunking.py`, dòng 139.
- **Nguyên nhân gốc rễ:** Lời gọi `chunks.append(Chunk(...))` ở cuối `chunk_semantic()` thiếu một dấu `)`.
- **Quá trình debug:** Mở vùng dòng 110-175, đếm cặp ngoặc của `append` và `Chunk`, thêm dấu đóng còn thiếu rồi chạy `py_compile` trước khi chạy pytest.
- **Cách giải quyết:** Sửa đúng một dấu ngoặc, sau đó dùng `python -m py_compile` và test module để phát hiện lỗi cú pháp sớm.

### 4. Model lớn, cache và dịch vụ Qdrant

- **Exact warning:** `Failed to obtain server version. Unable to check client-server compatibility.` và cảnh báo Hugging Face unauthenticated request.
- **Nguyên nhân gốc rễ:** Qdrant ở `localhost:6333` chưa chạy; ba model Hugging Face chiếm khoảng 6.5 GB cache thực tế nên việc tải/lưu sai ổ có thể làm đầy ổ hệ thống.
- **Quá trình debug:** Kiểm tra port 6333, chạy `docker compose ps`, quét cache Hugging Face và đo dung lượng từng model.
- **Cách giải quyết:** Khởi động Qdrant bằng `docker compose up -d`, đặt `HF_HOME` sang ổ có đủ dung lượng, tải model tuần tự và giữ fallback Qdrant in-memory để unit test vẫn chạy khi service chưa sẵn sàng.

### 5. Production ban đầu kém hơn baseline

- **Hiện tượng:** Lần đánh giá đầu cho thấy Faithfulness 0.7708, Answer Relevancy 0.6345, Context Precision 0.8875 và Context Recall 0.8167 — đều thấp hơn baseline.
- **Nguyên nhân gốc rễ:** Pipeline dùng enriched child text cho cả retrieval lẫn answer, không khôi phục parent context, đồng thời nhiều child cùng parent có thể chiếm hết top-3.
- **Quá trình debug:** Đọc `ragas_report.json`, phân tích bottom failures, map từng metric về retrieval/generation và thực hiện ablation ở mức kiến trúc.
- **Cách giải quyết:** Chỉ dùng enrichment cho index; lưu `original_text` và `parent_text` trong metadata; gom child theo `source + parent_id`; rerank parent sạch; đặt generation temperature bằng 0 và yêu cầu trả lời trực tiếp. Kết quả cuối vượt baseline ở cả bốn metric.

### Kiến thức còn thiếu và cách bổ sung

- Tôi cần hiểu sâu hơn về version-aware retrieval vì các tài liệu v1/v2 có nội dung rất giống nhau. Tôi sẽ bổ sung metadata `version`, `status`, `effective_date` và thử metadata filtering/boosting trong Qdrant.
- Câu hỏi multi-hop và câu hỏi tính toán không thể giải quyết tốt chỉ bằng tăng top-k. Tôi sẽ học query decomposition và kết hợp retrieval với calculator/tool calling.
- RAGAS dựa trên LLM nên có chi phí và độ biến thiên. Tôi sẽ bổ sung bộ deterministic retrieval tests, lưu answer/context từng câu và chạy đánh giá lặp để theo dõi mean cùng variance.
- Hai PDF scan bị bỏ qua cho thấy ingestion production cần OCR. Tôi sẽ thử PyMuPDF/Tesseract hoặc dịch vụ OCR trước bước chunking.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

### Project: Trợ lý hỏi đáp chính sách nội bộ doanh nghiệp

#### 1. Hiện trạng

- **Pipeline hiện tại:** Markdown/PDF text → hierarchical chunking → combined enrichment → BM25 + BGE-M3/Qdrant → RRF → parent-level BGE reranking → GPT-4o-mini grounded answer → RAGAS evaluation.
- **Kết quả hiện tại:** Production đạt 0.8500 faithfulness, 0.8639 answer relevancy, 0.9500 context precision và 0.9333 context recall trên 20 câu hỏi.
- **Bottlenecks:** PDF scan chưa OCR; BGE-M3 và BGE reranker nặng; first-load latency cao; version conflict; multi-hop retrieval; phép tính chính sách; report chưa lưu nguyên văn answer/context theo từng câu.

#### 2. Kế hoạch cải tiến

1. **Chunking strategy:** Giữ hierarchical chunking làm mặc định vì cần child precision và parent context. Với bảng Markdown và quy trình đánh số, kết hợp structure-aware chunking để không cắt giữa bảng/rule. Thêm OCR cho PDF scan và cache kết quả parsing theo checksum.
2. **Search retrieval:** Tiếp tục hybrid BM25 + BGE-M3 + RRF. Bổ sung query decomposition cho câu nhiều ý, metadata filter cho `source/category/version/status`, và boost tài liệu `current` khi có nhiều phiên bản chính sách.
3. **Reranking:** Dùng `BAAI/bge-reranker-v2-m3` cho top-20 candidates, gom theo parent trước khi lấy top-3. Cache model singleton và benchmark P50/P95. Nếu triển khai CPU nhỏ, đánh giá FlashRank hoặc model quantized làm phương án tiết kiệm.
4. **Evaluation:** Mở rộng test set từ 20 lên ít nhất 100 câu, bao gồm exact lookup, multi-hop, version conflict, negative/no-answer và calculation. Theo dõi RAGAS 4 metrics cùng Recall@k, MRR, latency, token cost và tỷ lệ từ chối đúng.
5. **Enrichment:** Giữ combined single-call để giảm 4 calls xuống 1 call/chunk. Enrichment chỉ phục vụ retrieval; answer luôn dùng raw parent. Cache enrichment theo content hash và validate JSON schema để tránh dữ liệu sinh sai đi vào index.
6. **Observability và safety:** Lưu query, candidate ranks, source, answer, contexts, latency từng stage và model version. Thêm citation theo source, guardrail “không đủ thông tin” và kiểm tra quyền truy cập theo metadata trước khi trả lời.

#### 3. Timeline triển khai

- **Tuần 1 — Ingestion & chunking:** OCR hai PDF scan, chuẩn hóa metadata, version/status extraction, test structure-aware và hierarchical chunking.
- **Tuần 2 — Retrieval:** Query decomposition, metadata filtering, version-aware boosting; benchmark BM25, dense và hybrid bằng Recall@k/MRR.
- **Tuần 3 — Reranking & generation:** Cache model, benchmark latency, thử quantization/FlashRank, thêm citation và grounded prompt cho multi-hop/calculation.
- **Tuần 4 — Evaluation & deployment:** Mở rộng test set, chạy RAGAS nhiều lần, xây dashboard latency/cost/failure, đóng gói Docker và kiểm thử end-to-end.

#### 4. Tiêu chí hoàn thành

- Faithfulness ≥ 0.88 và Answer Relevancy ≥ 0.90.
- Context Precision và Context Recall ≥ 0.95 trên test set mở rộng.
- 100% câu hỏi version conflict chọn đúng chính sách hiện hành.
- P95 retrieval + reranking dưới 1 giây sau warm-up trên máy triển khai mục tiêu.
- Mọi câu trả lời đều có source citation hoặc trả về “Không tìm thấy thông tin” khi bằng chứng không đủ.
