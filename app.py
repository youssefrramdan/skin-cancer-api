import os
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from flask import Flask, request, jsonify
import requests
from urllib.parse import urlparse

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

def is_valid_url(url):
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except:
        return False

def process_image_from_url(image_url):
    # Download image from URL
    response = requests.get(image_url, timeout=10)
    if response.status_code != 200:
        raise Exception("Failed to download image")

    # Convert to numpy array
    image_array = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if image is None:
        raise Exception("Invalid image format")

    # Process image
    image = cv2.resize(image, (256, 256))
    image = image / 255.0
    image = image.astype(np.float32)
    image = np.expand_dims(image, axis=0)
    return image

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
            '/api/predict': 'POST - Send image URL for prediction'
        },
        'example_request': {
            'url': 'https://example.com/image.jpg'
        }
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    try:
        # Get image URL from request
        data = request.get_json()
        if not data or 'url' not in data:
            return jsonify({
                'error': 'No image URL provided',
                'example': {
                    'url': 'https://example.com/image.jpg'
                }
            }), 400

        image_url = data['url']

        # Validate URL
        if not is_valid_url(image_url):
            return jsonify({
                'error': 'Invalid URL format'
            }), 400

        # Process image from URL
        try:
            processed_image = process_image_from_url(image_url)
        except Exception as e:
            return jsonify({
                'error': 'Failed to process image',
                'details': str(e)
            }), 400

        # Make prediction
        prediction = get_prediction(processed_image)
        predicted_class, confidence, diagnosis = interpret_prediction(prediction)

        # Return result
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence,
            'diagnosis': diagnosis,
            'input_url': image_url
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
