from flask import Flask, render_template, request
import os
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

# Create Flask instance
app = Flask(__name__)

# Configure upload folder and max size
UPLOAD_FOLDER = 'static/images'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # Max 10MB

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_image(filename):
    img = load_img(filename, color_mode='grayscale', target_size=(28, 28))
    img = img_to_array(img)
    img = img.reshape(1, 28, 28, 1)
    img = img.astype('float32') / 255.0
    return img

LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]

@app.route("/", methods=['GET', 'POST'])
def home():
    return render_template('index.html')

@app.route("/predict", methods=['POST'])
def predict():
    ...
    file = request.files.get('file')
    if file and allowed_file(file.filename):
        file.save(filepath)
        img = read_image(filepath)
        model = load_model('fashion_mnist_model.keras')
        prediction = model.predict(img)
        predicted_class = np.argmax(prediction[0])
        fashionitem = LABELS[predicted_class]
        return render_template('predict.html', fashion_item=fashionitem, user_image=filepath)

if __name__ == "__main__":
    app.run()