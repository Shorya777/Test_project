from flask import Flask, render_template ,request ,jsonify
import base64
from PIL import Image
from io import BytesIO
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('trained_model.h5')

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Decode the base64 image
    img_data = base64.b64decode(data['image'].split(',')[1])
    image = Image.open(BytesIO(img_data)).convert('L')
    image = image.resize((28, 28))  # Resize to 28x28
    image_array = np.array(image)
    image_array = 255 - image_array  # Invert colors (white background to black)
    image_array = image_array / 255.0  # Normalize
    # print(image_array)
    import matplotlib.pyplot as plt
    plt.imshow(image_array, cmap = plt.get_cmap('gray'))
    plt.show()
    image_array = image_array.reshape(1, 28, 28)  # Add batch dimension

    # Predict using the model
    prediction = model.predict(image_array)
    digit = np.argmax(prediction)

    return jsonify({'prediction': int(digit)})

if __name__ == '__main__':
    app.run(debug=True)
