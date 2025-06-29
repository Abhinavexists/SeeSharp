import os
import io
import base64
from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
from PIL import Image
from test_interface import ERSVRTester

app = Flask(__name__, template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

tester = None

def init_model():
    global tester
    model_path = 'student_models/student_best.pth'
    if os.path.exists(model_path):
        tester = ERSVRTester(model_path)
        print("Model loaded successfully!")
    else:
        print(f"Warning: Model file {model_path} not found!")

def image_to_base64(image_array):
    pil_image = Image.fromarray(image_array)
    
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    
    img_base64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_base64}"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if tester is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        image_bytes = file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        temp_path = 'temp_input.png'
        cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        input_frames = tester.load_single_frame_as_triplet(temp_path)
        sr_output = tester.super_resolve(input_frames)
        
        input_img = tester.tensor_to_image(input_frames[0, :, 1, :, :])
        sr_img = tester.tensor_to_image(sr_output.squeeze(0))
        
        bicubic_img = cv2.resize(input_img, (sr_img.shape[1], sr_img.shape[0]), 
                               interpolation=cv2.INTER_CUBIC)
        
        input_b64 = image_to_base64(input_img)
        bicubic_b64 = image_to_base64(bicubic_img)
        sr_b64 = image_to_base64(sr_img)
        
        bicubic_float = bicubic_img.astype(np.float32) / 255.0
        sr_float = sr_img.astype(np.float32) / 255.0
        mse = np.mean((bicubic_float - sr_float) ** 2)
        psnr = 20 * np.log10(1.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify({
            'success': True,
            'input_image': input_b64,
            'bicubic_image': bicubic_b64,
            'sr_image': sr_b64,
            'metrics': {
                'input_size': f"{input_img.shape[1]}x{input_img.shape[0]}",
                'output_size': f"{sr_img.shape[1]}x{sr_img.shape[0]}",
                'scale_factor': f"{sr_img.shape[0] // input_img.shape[0]}x",
                'psnr_improvement': f"{psnr:.2f} dB"
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<result_type>')
def download_result(result_type):
    return jsonify({'error': 'Download not implemented'}), 501

if __name__ == '__main__':
    print("Initializing ERSVR Web Interface...")
    init_model()
    print("\nStarting web server...")
    print("Open your browser and go to: http://localhost:5000")
    app.run(debug=True, host='0.0.0.0', port=5000) 