

# 📝 Vietnamese Student Sentiment Analysis

Welcome to the **Vietnamese Student Sentiment Analysis** project. This application is developed to automatically analyze and evaluate sentiments from Vietnamese student feedback, supporting lecturers and schools in understanding students' desires and satisfaction levels.

🔗 **App Live Demo:** [Hugging Face Spaces](https://huggingface.co/spaces/oriontk24/Vietnamese-Sentiment-Analysis)

---

## 📊 1. Data Description
The data used in this project consists of comments and evaluations of courses and lecturers written in Vietnamese. The data source is taken from the **UIT-VSFC** (Vietnamese Students’ Feedback Corpus).

🔗 **Dataset Link:** [uit-nlp/vietnamese_students_feedback](https://huggingface.co/datasets/uit-nlp/vietnamese_students_feedback)

*   **Features:** Text feedback segments.
*   **Classification Labels:** Data is assigned to 3 main sentiment labels:
    *   **Positive:** Good feedback, praise.
    *   **Neutral:** General constructive feedback, no clear emotional state.
    *   **Negative:** Negative feedback, criticism, or dissatisfaction.

## 🧹 2. Data Preprocessing
To enable machine learning and deep learning models to learn text effectively, Vietnamese data goes through thorough cleaning and preprocessing:
*   **Basic Text Cleaning:** Lowercasing all characters, removing punctuation and special characters, deleting repeated characters (e.g., "qáaaaa" -> "qá"), and condensing redundant whitespaces.
*   **Vietnamese Normalization:** Utilizing the `underthesea` library for standard Vietnamese input normalization (`text_normalize`).
*   **Word Tokenization:** Vietnamese uses compound words (multiple syllables forming one word, e.g., "sinh_viên"), so the `word_tokenize` function from `underthesea` combined with underscores is used to maintain the semantic integrity of phrases.
*   **Vectorization / Transformer Tokenization:** Using the specialized `AutoTokenizer` for PhoBERT, combined with padding and truncation techniques at a maximum length (`max_length = 256`) to create fixed-length tensor matrices for the neural network.

## 🤖 3. Tested Algorithms

The project built, trained, and benchmarked on a large scale across various models, from traditional machine learning to complex deep learning. Below is a summary table of performance metrics (Accuracy, F1-Score, Precision, Recall) of all tested models:

| Model | Accuracy | F1-Score | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| **PhoBERT (SOTA Transformer)** | **96.70%** | **96.69%** | **96.73%** | **96.70%** |
| Random Forest | 93.67% | 93.62% | 93.70% | 93.67% |
| GRU | 93.35% | 93.33% | 93.35% | 93.35% |
| Bidirectional LSTM | 92.14% | 92.08% | 92.32% | 92.14% |
| Support Vector Machine (SVM) | 90.58% | 90.55% | 90.80% | 90.58% |
| Stacked ML Models (Ensemble) | 90.58% | 90.57% | 90.68% | 90.58% |
| Fully Connected Layers (Dense) | 90.05% | 90.00% | 90.12% | 90.05% |
| Logistic Regression (Baseline) | 89.70% | 89.69% | 90.01% | 89.70% |

### 🏆 Best Model: PhoBERT
After a comprehensive evaluation, the **PhoBERT** model proved its overwhelming power over other algorithms with superior quality. Its superior context analysis capability helped PhoBERT easily reach ~96.7% accuracy during evaluation and maintain >94% when tested on real-world random data. Therefore, **PhoBERT** was chosen as the Production model for the application.

## 🚀 4. Deployment
After successful training, the model deployment pipeline was finalized through the following steps:

1.  **Model Packaging:** Learned weights (`keras_model`) and the PhoBERT `tokenizer` were bundled into an archive file (`phobert_production_bundle.zip`) to optimize cloud storage utilization.
2.  **UI Building:** Integrated the **Gradio** library to create an intuitive web interface for end-users, consisting of two modules:
    *   *Single Inference:* Real-time prediction returning a bar chart showing sentiment percentages.
    *   *Batch Inference:* Business-focused module supporting CSV/Excel file uploads, with automated text column detection, inference processing, and visualization via Matplotlib Pie Charts (with transparent backgrounds for Hugging Face Dark Mode compatibility).
3.  **Deployment on Hugging Face Spaces:** Uploading the source code (`app.py`), components, and `requirements.txt` to HF Spaces. On application startup, the system automatically extracts the `.zip` file into the `model/phobert_bundle` directory, making the model ready for `tf_keras` loading and request handling.

---

### Local Installation & Running Guide

You can clone and host this project locally on your machine:

```bash
# 1. Clone source code
git clone https://huggingface.co/spaces/oriontk24/Vietnamese-Sentiment-Analysis
cd Vietnamese-Sentiment-Analysis

# 2. Install required dependencies
pip install -r requirements.txt

# 3. Launch the Gradio Server 
# (The model will automatically unzip on the first launch)
python app.py
```
*Access the local control panel via the URL provided in the terminal (Typically `http://localhost:7860`).*
