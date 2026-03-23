import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tf_keras as keras

print("Loading model...")
model = keras.models.load_model('model/phobert_bundle/keras_model', compile=False)
print("\n--- Model Summary ---")
model.summary()
print("\n--- Layers list ---")
for i, layer in enumerate(model.layers):
    print(f"Layer {i}: {layer.name} - {layer.__class__.__name__}")
