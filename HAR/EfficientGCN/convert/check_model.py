import tensorflow as tf
import numpy as np

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="data/tflite-model.tflite")
interpreter.allocate_tensors()

# Get input and output tensors
print("\n-----------details-------------\n")
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(f"input_details = {input_details}\n")
print(f"output_details = {output_details}\n")

# Prepare sample input data
input_shape = input_details[0]['shape']
print(f"input_shape = {input_shape}\n")
input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

# Set input tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])
print(f"output_data = {output_data}\n")
