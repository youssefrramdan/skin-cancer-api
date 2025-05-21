import tensorflow as tf
import keras
import numpy as np
import os

def optimize_model(model_path, output_path):
    print("Loading original model...")
    model = keras.models.load_model(model_path)

    # 1. Convert to float16 precision
    print("Converting to float16 precision...")
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    model_fp16 = tf.keras.models.clone_model(model)
    model_fp16.set_weights(model.get_weights())

    # 2. Quantization aware training
    print("Applying quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model_fp16)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    # 3. Convert to TFLite format
    print("Converting to TFLite format...")
    tflite_model = converter.convert()

    # Save the optimized model
    print("Saving optimized model...")
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Original model size: {os.path.getsize(model_path) / 1024 / 1024:.2f} MB")
    print(f"Optimized model size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    return output_path

if __name__ == "__main__":
    optimize_model("model/Skin.keras", "model/Skin_optimized.tflite")
