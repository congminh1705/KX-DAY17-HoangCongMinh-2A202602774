# Bước 8: Phân tích kết quả benchmark

Hai agent chạy offline trên cùng dữ liệu benchmark. Baseline chỉ có short-term memory theo thread; Advanced bổ sung persistent memory bằng `User.md` và compact memory. Câu hỏi recall được hỏi ở thread mới.

## Kết quả benchmark

| Bộ benchmark | Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Standard | Baseline | 1945 | 18854 | 0% | 0% | 0 | 0 |
| Standard | Advanced | 1954 | 26643 | 100% | 100% | 274 | 0 |
| Long-context stress | Baseline | 338 | 23059 | 0% | 0% | 0 | 0 |
| Long-context stress | Advanced | 391 | 13552 | 100% | 100% | 227 | 3 |

Số liệu được xác nhận bằng chạy lại benchmark offline. Token được ước lượng bằng `ceil(số ký tự đã strip / 4)`, không phải usage từ API; tính cả lượt hội thoại và recall. Prompt bao gồm system prompt, history và profile/summary nếu có. Recall đo tỷ lệ chuỗi `expected_contains` xuất hiện trong câu trả lời. Response quality dùng cùng tỷ lệ coverage, chưa đánh giá độc lập độ tự nhiên hay khả năng reasoning.

## 1. Vì sao Advanced có recall tốt hơn Baseline?

Baseline chỉ giữ lịch sử trong thread hiện tại. Khi hỏi ở thread mới, agent không còn facts của phiên trước nên cross-session recall bằng 0%.

Advanced trích facts ổn định như tên, nơi ở, nghề nghiệp và phong cách trả lời để lưu vào `User.md`, rồi đọc lại khi xử lý lượt mới. Vì vậy agent nhớ qua thread mới và sau khi khởi tạo lại. Advanced đạt recall 100% trên cả hai bộ dữ liệu; kết quả này chỉ thể hiện khả năng nhớ các facts được chấm trong dataset.

## 2. Vì sao Advanced có thể tốn hơn ở hội thoại ngắn?

Advanced luôn mang thêm profile vào prompt. Khi lịch sử còn ngắn, phần ngữ cảnh bổ sung tạo chi phí trong khi chưa có nhiều nội dung để nén.

Ở Standard Benchmark, compact không kích hoạt với ngưỡng mặc định 1200 token. Advanced xử lý 26643 prompt token so với 18854 của Baseline, tăng khoảng 41,3%. Output token tăng nhẹ từ 1945 lên 1954. Đổi lại, agent trả lời được câu hỏi recall qua phiên mới. Persistent memory cải thiện khả năng nhớ nhưng không luôn tiết kiệm token ở hội thoại ngắn.

## 3. Vì sao compact giúp Advanced có lợi thế ở hội thoại dài?

Baseline đưa toàn bộ lịch sử thread vào prompt ở các lượt tiếp theo, làm nội dung cũ được xử lý lặp lại. Advanced tóm tắt lịch sử cũ khi vượt ngưỡng, giữ summary và các message gần nhất. Facts ổn định vẫn được lưu riêng trong `User.md`.

Ở Stress Benchmark, Advanced compact 3 lần và xử lý 13552 prompt token so với 23059 của Baseline, giảm khoảng 41,2%, trong khi recall profile vẫn đạt 100%. Output token tăng từ 338 lên 391 vì Advanced trả lời được thông tin đã nhớ. Compact chủ yếu giảm `Prompt tokens processed`, không bảo đảm giảm token sinh ra.

Summary có giới hạn độ dài nên có thể mất chi tiết tạm thời rất cũ. Stress benchmark chấm recall profile, chưa kiểm chứng đầy đủ việc giữ mọi nội dung các chủ đề dài. Một message quá lớn vẫn có thể vượt ngưỡng dù compact hoạt động.

## 4. File memory tăng trưởng ra sao và có rủi ro gì?

Baseline không tạo file persistent memory. Profile Advanced tăng từ rỗng lên 274 byte trong Standard Benchmark và 227 byte trong Stress Benchmark. Hai bộ chạy với state riêng, nên đây không phải các mốc tăng trưởng nối tiếp của một profile. Kích thước được đo theo UTF-8 với newline LF.

`User.md` dùng upsert theo field: nhắc lại cùng fact không tạo dòng trùng; correction thay giá trị cũ. Ví dụ, Huế thay Đà Nẵng và MLOps engineer thay backend engineer. Cách này hạn chế tăng trưởng do lặp lại và tránh giữ facts mâu thuẫn. File vẫn tăng khi thêm field hoặc giá trị dài hơn; profile lớn cũng tăng chi phí prompt.

Rủi ro gồm lưu sai fact, giữ thông tin lỗi thời, bỏ sót correction và mất chi tiết khi tóm tắt. Bài làm lọc câu hỏi, giả định, phủ định, câu đùa và áp dụng confidence threshold trước khi ghi. Tuy nhiên, regex có thể bỏ sót cách diễn đạt khác; confidence heuristic cố định 0.95 chưa phải xác suất được hiệu chuẩn. Chưa có memory decay, nên thông tin cũ cần được cập nhật rõ ràng.

## Chạy lại benchmark

Từ thư mục gốc, sau khi cài dependency:

```bash
python src/benchmark.py
```

Nếu `state/` không có quyền ghi, đặt `STATE_DIR` đến thư mục có quyền ghi. Hướng dẫn chi tiết và bonus nằm trong [src/README.md](src/README.md).