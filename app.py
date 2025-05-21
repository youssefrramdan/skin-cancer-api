import os
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from flask import Flask, request, jsonify
import requests
from urllib.parse import urlparse
from functools import lru_cache
from time import time

# Initialize Flask app
app = Flask(__name__)

# Cache configuration
CACHE_TIMEOUT = 3600  # 1 hour in seconds
prediction_cache = {}

def clean_expired_cache():
    current_time = time()
    expired_keys = [k for k, v in prediction_cache.items() if current_time - v['timestamp'] > CACHE_TIMEOUT]
    for k in expired_keys:
        prediction_cache.pop(k)

@lru_cache(maxsize=32)
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
    try:
        # Download image from URL with timeout
        response = requests.get(image_url, timeout=5)  # Reduced timeout to 5 seconds
        if response.status_code != 200:
            raise Exception("Failed to download image")

        # Convert to numpy array more efficiently
        image_array = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if image is None:
            raise Exception("Invalid image format")

        # Process image more efficiently
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
        image = (image / 255.0).astype(np.float32)
        return np.expand_dims(image, axis=0)
    except requests.Timeout:
        raise Exception("Image download timed out")
    except requests.RequestException as e:
        raise Exception(f"Error downloading image: {str(e)}")

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

        # Check cache first
        if image_url in prediction_cache:
            cache_entry = prediction_cache[image_url]
            if time() - cache_entry['timestamp'] < CACHE_TIMEOUT:
                return jsonify(cache_entry['result'])
            else:
                prediction_cache.pop(image_url)

        # Clean expired cache entries periodically
        if len(prediction_cache) > 100:  # Arbitrary threshold
            clean_expired_cache()

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

        # Prepare result
        result = {
            'predicted_class': predicted_class,
            'confidence': confidence,
            'diagnosis': diagnosis,
            'input_url': image_url
        }

        # Cache the result
        prediction_cache[image_url] = {
            'result': result,
            'timestamp': time()
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
