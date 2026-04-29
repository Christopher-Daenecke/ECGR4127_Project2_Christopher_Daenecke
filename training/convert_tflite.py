import tensorflow as tf
import numpy as np
import pathlib

MODEL_FILE   = "kws_model.h5"
TFLITE_FILE  = "kws_model.tflite"


print(f"Loading {MODEL_FILE}...")
model = tf.keras.models.load_model(MODEL_FILE)

def representative_dataset_gen():
    for _ in range(100):
        data = np.random.uniform(-1, 1, (1, 49, 40, 1)).astype(np.float32)
        yield [data]


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen


converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()

pathlib.Path(TFLITE_FILE).write_bytes(tflite_model)
print(f"Saved TFLite model to {TFLITE_FILE}")
print(f"Model size: {len(tflite_model) / 1024:.1f} KB")

interpreter = tf.lite.Interpreter(model_path=TFLITE_FILE)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input Type:  {input_details[0]['dtype']}")
print(f"Output Type: {output_details[0]['dtype']}")
print(f"Input Shape: {input_details[0]['shape']}")

if input_details[0]['dtype'] == np.int8:
    print("SUCCESS: Model is fully quantized to INT8")
else:
    print("WARNING: Model is NOT int8. Check your quantization settings.")