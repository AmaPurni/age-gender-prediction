from flask import Flask, request, jsonify
from PIL import Image, ImageOps
import numpy as np
import io
from flask_cors import CORS 


from tensorflow.keras.models import load_model

app = Flask(__name__)
CORS(app)

# Load age and gender prediction models
agemodel = load_model("Agemodel1.h5", compile=False)
genmodel = load_model('Genmodel1.h5', compile=False)

def process_and_predict(file):
    im = Image.open(file)
    width, height = im.size
    
    # Resize or crop image to 200x200 using LANCZOS for antialiasing
    if width == height:
        im = im.resize((200, 200), Image.LANCZOS)
    else:
        if width > height:
            left = (width - height) / 2
            right = (width + height) / 2
            top = 0
            bottom = height
            im = im.crop((left, top, right, bottom))
            im = im.resize((200, 200), Image.LANCZOS)
        else:
            left = 0
            right = width
            top = 0
            bottom = width
            im = im.crop((left, top, right, bottom))
            im = im.resize((200, 200), Image.LANCZOS)
    
    # Convert image to numpy array and normalize
    ar = np.asarray(im)
    ar = ar.astype('float32') / 255.0
    ar = ar.reshape(-1, 200, 200, 3)
    
    # Predict age
    age = int(agemodel.predict(ar)[0])
    
    # Predict gender (0 for male, 1 for female)
    gender_pred = genmodel.predict(ar)
    gender = 'male' if gender_pred < 0.5 else 'female'
    
    return age, gender, im.resize((300, 300), Image.LANCZOS)  # Resize result image with LANCZOS

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        age, gender, result_image = process_and_predict(file)
        
        # Save result image to bytes for response
        img_byte_arr = io.BytesIO()
        result_image.save(img_byte_arr, format='JPEG')
        img_byte_arr = img_byte_arr.getvalue()

        return jsonify({
            'age': age,
            'gender': gender,
            'result_image': img_byte_arr.decode('latin1')  # Return image as base64 string
        })
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
