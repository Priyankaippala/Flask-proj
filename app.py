from flask import Flask, render_template, request, send_from_directory
import os
from werkzeug.utils import secure_filename
import face_recognition
import cv2 as cv
import numpy as np

app = Flask(__name__)

app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB max upload size
upload_folder = os.path.join('static', 'uploads')
output_folder = os.path.join('static', 'output')
app.config['UPLOAD_FOLDER'] = upload_folder
app.config['OUTPUT_FOLDER'] = output_folder

if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def findEncodings(image_path):
    img = cv.imread(image_path)
    resized_img = cv.resize(img, (0, 0), None, 0.25, 0.25)
    resized_img = cv.cvtColor(resized_img, cv.COLOR_BGR2RGB)
    face_loc = face_recognition.face_locations(resized_img)
    encodings = face_recognition.face_encodings(resized_img, face_loc)
    return encodings

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['img']
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        encodeList = findEncodings(file_path)


        return render_template('index2.html', encodes=encodeList, output_image=filename)
    return render_template('index2.html')

@app.route('/compare-encodings', methods=['GET', 'POST'])
def compare():
    if request.method == 'POST':
        file = request.files['img']
        known_encodings = request.form.getlist('encodings')  # List of known encodings as strings
        known_encodings = [np.array(eval(enc)) for enc in known_encodings]  # Convert string back to numpy array
        
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        encodeList = findEncodings(file_path)

        is_match = False
        if encodeList:
            face_distances = face_recognition.face_distance(known_encodings, encodeList[0])
            is_match = np.any(face_distances <= 0.6)  # Threshold of 0.6 for matching
        
        return render_template('compare.html', value=is_match, output_image=filename)
    return render_template('compare.html')

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/static/output/<filename>')
def output_file(filename):
    return send_from_directory(app.config['OUTPUT_FOLDER'], filename)


