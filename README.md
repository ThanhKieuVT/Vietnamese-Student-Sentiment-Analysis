---
title: Vietnamese Student Sentiment Analysis
emoji: 📝
colorFrom: blue
colorTo: green
sdk: gradio
python_version: "3.10"
app_file: app.py
pinned: false
license: mit
---

# 📝 Vietnamese Student Sentiment Analysis

Chào mừng bạn đến với dự án **Phân tích Cảm xúc Phản hồi Sinh viên Việt Nam** (Vietnamese Student Sentiment Analysis). Ứng dụng này được phát triển để tự động phân tích và đánh giá cảm xúc từ các lời nhận xét của sinh viên, hỗ trợ giảng viên và nhà trường trong việc thấu hiểu mong muốn cũng như mức độ hài lòng của người học.

🔗 **Link trải nghiệm ứng dụng:** [Hugging Face Spaces](https://huggingface.co/spaces/oriontk24/Vietnamese-Sentiment-Analysis)

---

## 📊 1. Mô tả Dữ liệu (Data Description)
Dữ liệu sử dụng trong dự án là các đoạn nhận xét, đánh giá của sinh viên về khóa học và giảng viên được viết bằng tiếng Việt. Nguồn data được lấy từ bộ **UIT-VSFC** (Vietnamese Students’ Feedback Corpus).

🔗 **Link dataset:** [uit-nlp/vietnamese_students_feedback](https://huggingface.co/datasets/uit-nlp/vietnamese_students_feedback)

*   **Đặc trưng (Features):** Các đoạn văn bản (text feedback).
*   **Nhãn phân lớp (Labels):** Dữ liệu được gán theo 3 nhãn cảm xúc chính:
    *   **Tích cực (Positive):** Phản hồi tốt, khen ngợi.
    *   **Trung lập (Neutral):** Nhận xét mang tính đóng góp chung chung, không bộc lộ sắc thái rõ ràng.
    *   **Tiêu cực (Negative):** Phản hồi không tốt, chê trách hoặc không hài lòng.

## 🧹 2. Quá trình Xử lý Dữ liệu (Data Preprocessing)
Để các mô hình Machine Learning và Deep Learning có thể học văn bản một cách hiệu quả, dữ liệu tiếng Việt được đưa qua các bước làm sạch và tiền xử lý kỹ lưỡng:
*   **Làm sạch văn bản cơ bản:** Chuyển đổi toàn bộ sang chữ in thường (lowercase), loại bỏ hoàn toàn các dấu câu và ký tự đặc biệt, xóa các ký tự lặp lại (ví dụ: "qáaaaa" -> "qá"), và làm gọn các khoảng trắng dư thừa.
*   **Chuẩn hóa tiếng Việt:** Ứng dụng thư viện `underthesea` để chuẩn hóa bộ gõ văn bản (`text_normalize`). 
*   **Tách từ (Word Tokenization):** Tiếng Việt có đặc điểm từ vựng ghép (nhiều âm tiết tạo thành 1 từ, ví dụ: "sinh_viên"), do đó hàm `word_tokenize` của `underthesea` kết hợp với underscore (gạch dưới) được sử dụng để ghép các từ lại với nhau nhằm bảo toàn ý nghĩa sắc thái của cụm từ.
*   **Vector hóa / Tokenize cho Transformer:** Sử dụng `AutoTokenizer` chuyên dụng của PhoBERT, kết hợp các kỹ thuật đệm (padding) và cắt ngắn (truncation) ở độ dài tối đa (`max_length = 256`) nhằm tạo ra các ma trận tensor độ dài cố định đưa vào mạng nơ-ron.

## 🤖 3. Các Giải Thuật Được Thử Nghiệm

Dự án đã tiến hành xây dựng, huấn luyện và đối chiếu quy mô lớn trên hàng loạt mô hình, từ Machine Learning truyền thống tới các mô hình Deep Learning phức tạp. Dưới đây là bảng tổng hợp kết quả (Accuracy, F1-Score, Precision, Recall) của tất cả các mô hình đã chạy trong notebook:

| Mô hình (Model) | Độ chính xác (Accuracy) | F1-Score | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **PhoBERT (SOTA Transformer)** | **96.70%** | **96.69%** | **96.73%** | **96.70%** |
| Random Forest | 93.67% | 93.62% | 93.70% | 93.67% |
| GRU | 93.35% | 93.33% | 93.35% | 93.35% |
| Bidirectional LSTM | 92.14% | 92.08% | 92.32% | 92.14% |
| Support Vector Machine (SVM) | 90.58% | 90.55% | 90.80% | 90.58% |
| Stacked ML Models (Ensemble) | 90.58% | 90.57% | 90.68% | 90.58% |
| Fully Connected Layers (Dense) | 90.05% | 90.00% | 90.12% | 90.05% |
| Logistic Regression (Baseline) | 89.70% | 89.69% | 90.01% | 89.70% |

### 🏆 Mô hình Xuất sắc nhất (Best Model): PhoBERT
Kết thúc quá trình đánh giá toàn diện, mô hình **PhoBERT** đã chứng minh sức mạnh áp đảo hoàn toàn các giải thuật khác với chất lượng vượt trội. Khả năng phân tích ngữ cảnh ưu việt giúp PhoBERT dễ dàng đạt độ chính xác ~96.7% trong quá trình đánh giá và duy trì được chuẩn >94% khi test trên dữ liệu ngẫu nhiên thực tế. Do đó, **PhoBERT** đã được chọn làm mô hình Production đưa lên ứng dụng.

## 🚀 4. Triển khai Ứng dụng (Deployment)
Sau quá trình huấn luyện thành công ở bước 3, pipeline triển khai model lên môi trường trực tuyến hoàn thiện qua các bước sau:

1.  **Đóng gói Mô hình:** Trọng số đã học (`keras_model`) và `tokenizer` của PhoBERT được đóng gói chung vào một siêu tệp nén (`phobert_production_bundle.zip`) để tối ưu triệt để dung lượng lưu trữ trên Cloud.
2.  **Xây dựng Giao diện (UI):** Tích hợp thư viện **Gradio** tạo giao diện Web tương tác trực quan cho người dùng cuối gồm 2 module:
    *   *Phân tích 1 câu (Single Inference):* Dự đoán tính realtime và trả về bar-chart phần trăm các cảm xúc.
    *   *Phân tích hàng loạt từ File (Batch Inference):* Tính năng Business, hỗ trợ upload file (.CSV/Excel), hệ thống phân tách auto tìm cột text sau đó suy luận và vẽ biểu đồ Matplotlib Pie Chart (chế độ nền trong suốt phù hợp Hugging Face Dark Mode).
3.  **Triển khai trên Hugging Face Spaces:** Chuyển hệ thống code (`app.py`), components, cấu hình `requirements.txt` lên HF Spaces. Khi khởi động ứng dụng, dòng code thông minh tự động extract file `.zip` vào phân mục `model/phobert_bundle`, sẵn sàng load kiến trúc vào `tf_keras` chờ các requests.

---

### Hướng dẫn Cài đặt & Chạy cục bộ (Local)

Bạn hoàn toàn có thể sao chép và host project này trên máy tính của bạn:

```bash
# 1. Clone source code
git clone https://huggingface.co/spaces/oriontk24/Vietnamese-Sentiment-Analysis
cd Vietnamese-Sentiment-Analysis

# 2. Cài đặt các thư viện cần thiết
pip install -r requirements.txt

# 3. Khởi chạy Server Gradio 
# (Mô hình sẽ tự giải nén trong lần boot đầu tiên)
python app.py
```
*Truy cập bảng điều khiển trên Local URL hiện ra ở Terminal (Thường là `http://localhost:7860`).*
