import tensorflow as tf
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
from transformers import AutoTokenizer

try:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('model/phobert_bundle/tokenizer')
    print("Tokenizer loaded successfully.")
    
    print("Loading keras model...")
    import tf_keras as keras
    model = keras.models.load_model('model/phobert_bundle/keras_model', compile=False)
    print("Model loaded successfully.")
    print("Model inputs:", model.inputs)
    
    text = "Giảng viên này dạy quá tệ, không hiểu gì hết!"
    print("Test text:", text)
    
    # Tokenize
    inputs = tokenizer(text, padding='max_length', truncation=True, max_length=256, return_tensors='tf')
    print("Tokenized inputs:", inputs.keys())
    
    # Predict
    outputs = model(inputs)
    print("Outputs:", outputs)
    
except Exception as e:
    import traceback
    traceback.print_exc()
