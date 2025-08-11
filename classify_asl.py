import numpy as np
import os
import tensorflow as tf
from PIL import Image

MODEL_PATH = './ei-asl-rubikpi-project-transfer-learning-tensorflow-lite-float32-model.3.lite'
IMG_PATH = 'input.jpg'
IMG_SIZE = (160, 160)

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

img = Image.open(IMG_PATH).resize(IMG_SIZE).convert('RGB')
img_np = np.expand_dims(np.array(img,dtype=np.float32)/255.0, 0) # normalizing??????

# Get input and output tensors.
input_index = interpreter.get_input_details()[0]['index']
interpreter.set_tensor(input_index, img_np)

interpreter.invoke()

output_index = interpreter.get_output_details()[0]['index']
output = interpreter.get_tensor(output_index)

pred = np.argmax(output)
print("Predicted class index " + pred)
print("Raw probs: " + output)