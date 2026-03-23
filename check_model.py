import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

try:
    print("Loading raw saved_model...")
    # Load using pure tf.saved_model, not keras, to avoid tf_keras segfaults
    model = tf.saved_model.load('model/phobert_bundle/keras_model')
    print("Signatures:")
    for key, sig in model.signatures.items():
        print(f"\nSignature key: {key}")
        print("Inputs:")
        for tensor in sig.inputs:
            print(f" - {tensor.name}: shape={tensor.shape}, dtype={tensor.dtype}")
except Exception as e:
    print(f"Error loading model: {e}")
