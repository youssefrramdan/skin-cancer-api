import os
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from flask import Flask, request, jsonify
import base64

# Initialize Flask app
app = Flask(__name__)

def load_model(model_path):
    # Load TFLite model and allocate tensors
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

# Load the model
interpreter = load_model('model/Skin_optimized.tflite')
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class names for prediction
class_names = ["BCC", "MEL", "SCC"]

def process_image_from_bytes(image_bytes):
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Process image
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image

def process_image_from_base64(base64_string):
    # Decode base64 string to image bytes
    img_data = base64.b64decode(base64_string)
    return process_image_from_bytes(img_data)

def get_prediction(image):
    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], image)

    # Run inference
    interpreter.invoke()

    # Get output tensor
    prediction = interpreter.get_tensor(output_details[0]['index'])
    return prediction

def interpret_prediction(prediction):
    predicted_class_idx = np.argmax(prediction, axis=1)[0]
    confidence = float(np.max(prediction) * 100)
    predicted_class = class_names[predicted_class_idx]

    if predicted_class in class_names:
        diagnosis = f"Cancerous ({predicted_class})"
    else:
        diagnosis = "Normal"

    return predicted_class, confidence, diagnosis

@app.route('/', methods=['GET'])
def home():
    return jsonify({
        'status': 'API is running',
        'endpoints': {
            '/api/predict': 'POST - Send image for prediction'
        }
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            file_bytes = file.read()
            processed_image = process_image_from_bytes(file_bytes)

        # Handle base64 image
        elif 'image' in request.json:
            base64_string = request.json['image']
            processed_image = process_image_from_base64(base64_string)
        else:
            return jsonify({
                'error': 'No image provided'
            }), 400

        # Make prediction
        prediction = get_prediction(processed_image)
        predicted_class, confidence, diagnosis = interpret_prediction(prediction)

        # Return result
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence,
            'diagnosis': diagnosis
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    # Get port from environment variable or use 5000
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
