from flask import Flask, request, render_template, url_for
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from werkzeug.utils import secure_filename
import uuid

from class_names import CLASS_NAMES

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model
MODEL_PATH = 'model/traffic_sign_model.keras'
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully!")

IMG_HEIGHT, IMG_WIDTH = 64, 64  # Must match your training size

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    img_url = None

    if request.method == 'POST':
        if 'file' not in request.files:
            return "No file uploaded", 400

        file = request.files['file']
        if file.filename == '':
            return "No file selected", 400

        # Save uploaded file
        filename = secure_filename(str(uuid.uuid4()) + "_" + file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Preprocess image
        img = Image.open(filepath).convert('RGB')
        img = img.resize((IMG_HEIGHT, IMG_WIDTH))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict
        predictions = model.predict(img_array)
        class_idx = np.argmax(predictions[0])
        confidence_score = float(np.max(predictions[0]) * 100)

        prediction = CLASS_NAMES[class_idx]
        confidence = round(confidence_score, 2)
        img_url = url_for('static', filename=f'uploads/{filename}')

    return render_template('index.html', prediction=prediction, confidence=confidence, img_url=img_url)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
