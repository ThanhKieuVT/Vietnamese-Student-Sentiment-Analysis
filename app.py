import os
# --- Step 1: Force Legacy Keras ---
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import re
import string
import numpy as np
import tensorflow as tf
import tf_keras as keras  
import gradio as gr
from underthesea import word_tokenize, text_normalize
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer

# --- Step 1.5: Auto-extract PhoBERT bundle if missing ---
bundle_extracted = False
if not os.path.exists("phobert_bundle") and not os.path.exists("model/phobert_bundle"):
    bundle_name = "phobert_production_bundle.zip"
    if os.path.exists(bundle_name):
        print(f"Đang giải nén {bundle_name}...")
        with zipfile.ZipFile(bundle_name, 'r') as zip_ref:
            zip_ref.extractall("model/phobert_bundle")
    elif os.path.exists(f"model/{bundle_name}"):
        print(f"Đang giải nén model/{bundle_name}...")
        with zipfile.ZipFile(f"model/{bundle_name}", 'r') as zip_ref:
            zip_ref.extractall("model/phobert_bundle")

# --- Step 3: Model Finding Logic ---
possible_paths = [
    "phobert_bundle",
    "model/phobert_bundle"
]

model = None
tokenizer = None
load_error = None

for path in possible_paths:
    keras_path = os.path.join(path, "keras_model")
    tok_path = os.path.join(path, "tokenizer")
    
    # HF Spaces might extract it directly without the root dir sometimes
    if not os.path.exists(keras_path):
        keras_path = path
        tok_path = path

    if os.path.exists(keras_path) and os.path.exists(tok_path):
        try:
            print(f"Attempting to load PhoBERT Tokenizer and Model from {path}...")
            tokenizer = AutoTokenizer.from_pretrained(tok_path)
            model = keras.models.load_model(keras_path, compile=False)
            print(f"PhoBERT loaded successfully from {path}!")
            break
        except Exception as e:
            print(f"Failed to load PhoBERT from {path}: {e}")
            load_error = str(e)

if model is None or tokenizer is None:
    load_error = load_error or "Không tìm thấy thư mục phobert_bundle có chứa keras_model và tokenizer."

# --- Preprocessing ---
def clean_text(text):
    if not text:
        return ""
    text = str(text).lower()
    text = re.sub(r'[^\w\s,]', '', text) 
    text = re.sub(r'(([a-z])\2{1,})+', r'\2', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    text = text_normalize(text)
    text = word_tokenize(text, format="text")
    return text

# --- Prediction Logic ---
def predict(comment):
    if model is None or tokenizer is None:
        return f"Mô hình chưa tải được. Lỗi: {load_error}"
    
    try:
        cleaned_comment = clean_text(comment)
        
        # Tokenize sequence for PhoBERT
        encoded = tokenizer(
            text=cleaned_comment,
            padding='max_length',
            truncation=True,
            max_length=256,
            return_tensors='np'
        )
         
        input_data = np.array([cleaned_comment])
        
        prediction = model(input_data)
         
        pred_arr = prediction.numpy()[0] if hasattr(prediction, 'numpy') else prediction[0]
        
        confidences = {
            "Tiêu cực (Negative)": float(pred_arr[0]),
            "Trung lập (Neutral)": float(pred_arr[1]),
            "Tích cực (Positive)": float(pred_arr[2])
        }
        return confidences
    except Exception as e:
        return f"Lỗi dự báo: {str(e)}"

# --- predict_batch logic ---
def predict_batch(file_obj):
    if file_obj is None:
        return None, "Vui lòng tải lên một file (.csv, .xlsx)."
    
    try:
        if file_obj.name.endswith('.csv'):
            df = pd.read_csv(file_obj.name)
        elif file_obj.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_obj.name)
        else:
            return None, "Lỗi: Chỉ hỗ trợ file .csv hoặc .xlsx"
        
        text_col = None
        for col in df.columns:
            col_lower = str(col).lower()
            if 'comment' in col_lower or 'text' in col_lower or 'nhận xét' in col_lower or 'nội dung' in col_lower:
                text_col = col
                break
        
        if text_col is None:
            for col in df.columns:
                if df[col].dtype == object or df[col].dtype == str:
                    text_col = col
                    break
            if text_col is None:
                text_col = df.columns[0]
                
        labels = []
        for text in df[text_col]:
            if pd.isna(text):
                continue
            res = predict(str(text))
            
            if isinstance(res, dict):
                best_label = max(res, key=res.get)
                if "Tiêu cực" in best_label:
                    labels.append("Tiêu cực")
                elif "Trung lập" in best_label:
                    labels.append("Trung lập")
                else:
                    labels.append("Tích cực")
            else:
                pass
                
        if len(labels) == 0:
            return None, "Không tìm thấy dữ liệu hợp lệ để phân tích."
        
        from collections import Counter
        counts = Counter(labels)
        
        color_map = {
            "Tích cực": "#2ecc71",   
            "Tiêu cực": "#e74c3c",   
            "Trung lập": "#95a5a6"   
        }
        
        # Setup background trong suốt để hợp với Dark Mode của Hugging Face
        fig, ax = plt.subplots(figsize=(7, 6))
        fig.patch.set_alpha(0.0)
        ax.patch.set_alpha(0.0)
        
        plot_labels = list(counts.keys())
        plot_sizes = list(counts.values())
        plot_colors = [color_map.get(l, "#3498db") for l in plot_labels]
        
        # Hiệu ứng nổ (tách rời) cho miếng bánh to nhất
        explode = [0.08 if size == max(plot_sizes) else 0.0 for size in plot_sizes]
        
        # Vẽ biểu đồ không kèm label viền ngoài để tránh bị cắt chữ
        wedges, texts, autotexts = ax.pie(
            plot_sizes, 
            labels=None, 
            colors=plot_colors, 
            autopct='%1.1f%%', 
            startangle=140,
            explode=explode,
            shadow=True,
            textprops={'fontsize': 13, 'weight': 'bold', 'color': 'white'}
        )
        
        # Thêm chú thích (Legend) tách hẳn ra ngoài và đẩy lên cao
        legend = ax.legend(wedges, plot_labels, title="Cảm xúc", loc="upper left", 
                           bbox_to_anchor=(1.05, 0.95), prop={'size': 12, 'weight': 'bold'})
        plt.setp(legend.get_title(), color='white', weight='bold', fontsize=13)
        for text in legend.get_texts():
            text.set_color("white")
        legend.get_frame().set_alpha(0.2) # Khung chú thích trong suốt nhẹ
        
        # Tiêu đề màu trắng
        ax.set_title("Biểu đồ Phân bố Cảm xúc", fontsize=18, fontweight='bold', color='white', pad=20)
        ax.axis('equal')
        
        # Ép khung hình vừa vặn không bị cắt xén
        fig.tight_layout()
        
        return fig, f"Đã xử lý thành công {len(labels)} dòng dữ liệu từ cột '{text_col}'."
    except Exception as e:
        return None, f"Lỗi xử lý file: {str(e)}"

# --- UI Design ---
description = """
Ứng dụng phân tích cảm xúc lời nhận xét tiếng Việt của sinh viên (Positive, Negative, Neutral).

Mô hình sử dụng: PhoBERT (Transformers).
"""

with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown("# Vietnamese Student Sentiment Analysis")
    gr.Markdown(description)
    
    with gr.Tabs():
        # --- TAB 1: Phân tích từng nhận xét ---
        with gr.Tab("Phân tích 1 nhận xét"):
            with gr.Row():
                with gr.Column(scale=2):
                    txt_input = gr.Textbox(
                        lines=5, 
                        placeholder="Nhập lời nhận xét của sinh viên tại đây...",
                        label="Lời nhận xét"
                    )
                    with gr.Row():
                        submit_btn = gr.Button("Phân tích", variant="primary")
                        clear_btn = gr.Button("Xóa")
                with gr.Column(scale=1):
                    label_output = gr.Label(label="Kết quả phân tích", num_top_classes=3)
            
            gr.Examples(
                examples=[
                    ["Giảng viên dạy rất nhiệt tình, bài giảng dễ hiểu."],
                    ["Tài liệu khóa học hơi cũ, cần cập nhật thêm."],
                    ["Khóa học quá tệ, không đúng như mong đợi."]
                ],
                inputs=txt_input
            )
            
            submit_btn.click(fn=predict, inputs=txt_input, outputs=label_output)
            clear_btn.click(fn=lambda: ("", None), inputs=None, outputs=[txt_input, label_output])
            
        # --- TAB 2: Phân tích hàng loạt từ file ---
        with gr.Tab("Phân tích file"):
            gr.Markdown("Tải lên file danh sách các lời nhận xét để hệ thống phân tích tổng quan bằng biểu đồ tròn.")
            with gr.Row():
                with gr.Column(scale=1):
                    file_input = gr.File(label="Tải lên file .csv hoặc .xlsx")
                    file_submit_btn = gr.Button("Phân tích File", variant="primary")
                    text_output = gr.Textbox(label="Thông báo trạng thái", interactive=False)
                
                with gr.Column(scale=2):
                    plot_output = gr.Plot(label="Biểu đồ thống kê Tỷ lệ Cảm xúc")
            
            file_submit_btn.click(fn=predict_batch, inputs=file_input, outputs=[plot_output, text_output])

if __name__ == "__main__":
    iface.launch(server_name="0.0.0.0", server_port=7860)
