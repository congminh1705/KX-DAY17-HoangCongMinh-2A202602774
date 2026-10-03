# Bài làm Memory Systems for AI Agent

Đã hoàn thiện các module theo cấu trúc scaffold. Mặc định chạy offline, không cần API key. Dataset được giữ nguyên.

## Chạy trên Windows

```powershell
.venv/Scripts/python.exe src/benchmark.py
.venv/Scripts/python.exe -m pytest src/test_agents.py -v
```

Nếu chưa có môi trường: dùng Python >=3.11, tạo `.venv`, rồi `pip install -r requirements.txt`. `pytest.ini` đặt thư mục tạm ở `state/pytest` để chạy trong workspace bị giới hạn quyền. Thư mục này dành riêng cho pytest và được pytest dọn khi chạy lại.

## Cấu trúc và luồng memory

- `config.py`: đường dẫn, env, model chính/judge, ngưỡng compact và confidence.
- `model_provider.py`: lazy import cho openai, custom, gemini, anthropic, ollama, openrouter; báo lỗi provider không hợp lệ.
- `memory_store.py`: heuristic token, CRUD User.md, structured facts, extraction, summary và compact.
- `agent_baseline.py`: giữ nguyên history theo thread, không có memory file.
- `agent_advanced.py`: trích fact, upsert User.md, compact history, dựng prompt profile + summary + recent messages, rồi trả lời.
- `benchmark.py`: cùng input, recall sau từng conversation ở thread mới riêng cho từng câu hỏi; state sạch ở mỗi suite.
- `test_agents.py`: memory CRUD, correction, noise, confidence, cô lập user/thread, persistence sau khởi tạo lại, compact, token load, cả hai dataset và dispatch provider.

User.md nằm tại `state/profiles/<sha256(user_id)>/User.md`. Hash tránh path traversal và tránh hai user có slug giống nhau. Ghi qua file tạm rồi replace để tránh file bị ghi dở. Một field chỉ có một giá trị; correction thay thế giá trị cũ. Thread không được dùng lại cho user khác.

## Kết quả offline

Token được ước lượng bằng ceil(số ký tự đã strip / 4). Tất cả lượt train và recall đều được tính vào bộ đếm. Prompt tính cả system prompt, history và profile/summary nếu có. Không dùng API usage hoặc LLM judge; cấu hình judge được chuẩn bị cho phần mở rộng.

### Standard Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 1945 | 18854 | 0% | 0% | 0 | 0 |
| Advanced | 1954 | 26643 | 100% | 100% | 274 | 0 |

### Long-Context Stress Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| --- | --- | --- | --- | --- | --- | --- |
| Baseline | 338 | 23059 | 0% | 0% | 0 | 0 |
| Advanced | 391 | 13552 | 100% | 100% | 227 | 3 |

Memory bytes là kích thước UTF-8 với newline LF; số đo trước khi sửa newline Windows có thể khác. Các phần trăm recall tính tỷ lệ expected_contains xuất hiện trong câu trả lời, không lấy expected_contains để tạo câu trả lời. Response quality dùng cùng tỷ lệ coverage, nên không phải một đánh giá độc lập về độ tự nhiên hoặc khả năng reasoning.

## Phân tích

Advanced nhớ qua thread mới và cả sau khi khởi tạo lại vì đọc User.md. Baseline chỉ dùng history của thread hiện tại nên recall qua session mới bằng 0, đúng thiết kế. Hai agent dùng chung extractor và bộ sinh phản hồi offline để so sánh lớp memory công bằng.

Trong benchmark ngắn, Advanced xử lý nhiều prompt token hơn khoảng 41% vì luôn mang thêm profile. Compact không kích hoạt ở ngưỡng mặc định 1200; đây là chi phí thực của persistent memory trong các cuộc nói chuyện ngắn.

Trong stress test, compact kích hoạt 3 lần. Advanced xử lý 13552 so với 23059 prompt token của Baseline, giảm khoảng 41.2%. Các câu trả lời Advanced dài hơn khi recall thành công nên agent output tokens tăng; compact chủ yếu giảm chi phí ngữ cảnh được xử lý lặp lại. Profile ổn định vẫn được giữ dù history cũ bị nén.

User.md được upsert theo field thay vì append mỗi lần nhắc, vì vậy lặp lại fact không làm file tăng vô hạn. Summary giới hạn 1400 ký tự và tối đa một nửa ngưỡng token theo heuristic; recent messages giữ nguyên. Một message rất lớn vẫn có thể vượt ngưỡng: compact là nén history, không phải bảo đảm cứng context window.

## Bonus đã triển khai

1. Entity extraction có field: name, location, profession, interests, response_style, favorite_drink, favorite_food, pet.
2. Confidence threshold: mỗi assertion phù hợp rule có confidence heuristic 0.95, chỉ lưu khi vượt ngưỡng cấu hình. Ngưỡng 0.99 từ chối các assertion đó; confidence không phải xác suất được hiệu chuẩn.
3. Conflict handling: correction được xét theo thứ tự và thay thế field cũ; không giữ đồng thời backend engineer và MLOps engineer hay nơi ở cũ/mới.
4. Lọc câu hỏi, giả định, phủ định và câu đùa trước khi ghi. Hà Nội trong tin đi họp không tự thành nơi ở; câu hỏi recall không được ghi ngược vào profile.

Bonus giảm false writes và tránh profile phình do correction. Đổi lại, regex có thể bỏ sót assertion diễn đạt khác hoặc cả một message vừa có fact vừa có câu hỏi. Extraction nơi ở hiện hỗ trợ một nhóm thành phố Việt Nam; tên và nghề cũng theo các mẫu minh họa, không phải NER tổng quát. Production cần extractor có schema, nguồn fact, timestamp và confidence được đánh giá trên dữ liệu ngoài benchmark. Chưa triển khai memory decay vì xóa nhầm fact ít được nhắc có thể làm giảm recall; đây là hướng mở rộng khác với bonus đã làm.

Summary offline giữ snippets có giới hạn và facts, nên có thể mất nội dung tạm thời rất cũ. Benchmark stress hiện chỉ chấm recall profile qua session mới, chưa chứng minh việc giữ đủ abstraction của cả bốn chủ đề news. Không nên diễn giải recall 100% thành khả năng reasoning đầy đủ trên context dài.

## Chế độ live tùy chọn

Sao chép `.env.example` thành `.env`, đặt `LAB_LIVE=1`, provider, model và key. Có thể dùng `LLM_API_KEY` hoặc `<PROVIDER>_API_KEY`, Gemini hỗ trợ thêm `GOOGLE_API_KEY`. Custom cần base URL; Ollama dùng server local. Judge dùng nhóm biến `JUDGE_*`, nhưng chưa được gọi trong benchmark offline.

Hai agent dùng LangChain create_agent, history được truyền tường minh. Advanced thực hiện profile update và compaction trước invoke thay vì để model tùy ý ghi file qua tool. Không dùng thêm checkpointer vì sẽ giữ bản history thứ hai ngoài history đã compact. Chế độ live chưa gọi API thật trong kiểm chứng này; lỗi cấu hình hoặc thiếu package được báo trực tiếp, không âm thầm chuyển offline.

## Kiểm chứng

21 tests passed. Benchmark offline chạy thành công cả hai suite. Các test provider dùng constructor giả lập để kiểm tra dispatch không gọi mạng; không khẳng định đã tích hợp API thật của cả sáu dịch vụ.
