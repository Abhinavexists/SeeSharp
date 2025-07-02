import os
import io
import base64
import tempfile
from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
from PIL import Image
from test_interface import ERSVRTester
import threading
import time
import torch
import gc

app = Flask(__name__, template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size for videos

tester = None
processing_status = {}

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

def is_video_file(filename):
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    return any(filename.lower().endswith(ext) for ext in video_extensions)

def extract_frames_from_video(video_path, max_frames=150):
    """Extract frames from video, limiting to max_frames for performance"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Reduce max frames further to prevent memory issues
    frame_skip = max(1, total_frames // max_frames)
    
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        frame_count += 1
        
        if len(frames) >= max_frames:
            break
    
    cap.release()
    return frames, fps

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    gc.collect()

def process_video_frames(frames, task_id):
    """Process video frames with memory-efficient approach"""
    temp_dir = None
    try:
        processing_status[task_id] = {'progress': 0, 'status': 'processing', 'total': len(frames)}
        
        temp_dir = tempfile.mkdtemp()
        frame_paths = []
        
        # Check available GPU memory and decide processing mode
        use_cpu_for_video = False
        if torch.cuda.is_available():
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory
                gpu_memory_gb = gpu_memory / (1024**3)
                if gpu_memory_gb < 6:  # Less than 6GB, force CPU for videos
                    use_cpu_for_video = True
                    print(f"GPU memory ({gpu_memory_gb:.1f}GB) limited, using CPU for video processing")
            except:
                use_cpu_for_video = True
        
        # Temporarily switch model to CPU for video processing if needed
        original_device = tester.device
        if use_cpu_for_video and original_device != 'cpu':
            print("Switching to CPU for video processing to avoid CUDA memory issues")
            tester.model = tester.model.cpu()
            tester.device = 'cpu'
        
        for i, frame in enumerate(frames):
            try:
                # Clear GPU memory before each frame
                clear_gpu_memory()
                
                with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
                    cv2.imwrite(tmp_file.name, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    
                    with torch.no_grad():
                        input_frames = tester.load_single_frame_as_triplet(tmp_file.name)
                        sr_output = tester.super_resolve(input_frames)
                        sr_frame = tester.tensor_to_image(sr_output.squeeze(0))
                    
                    # Clear tensors immediately
                    del input_frames, sr_output
                    clear_gpu_memory()
                    
                    frame_filename = f"frame_{i:06d}.png"
                    frame_path = os.path.join(temp_dir, frame_filename)
                    cv2.imwrite(frame_path, cv2.cvtColor(sr_frame, cv2.COLOR_RGB2BGR))
                    frame_paths.append(frame_path)
                    
                    os.unlink(tmp_file.name)
                
                progress = int((i + 1) / len(frames) * 100)
                processing_status[task_id]['progress'] = progress
                
                # Force garbage collection every 10 frames
                if i % 10 == 0:
                    clear_gpu_memory()
                
            except Exception as frame_error:
                print(f"Error processing frame {i}: {frame_error}")
                # Continue with next frame instead of failing completely
                continue
        
        # Restore original device if changed
        if use_cpu_for_video and original_device != 'cpu':
            print("Restoring model to original device")
            tester.model = tester.model.to(original_device)
            tester.device = original_device
            clear_gpu_memory()
        
        processing_status[task_id]['status'] = 'completed'
        processing_status[task_id]['temp_dir'] = temp_dir
        processing_status[task_id]['frame_paths'] = frame_paths
        
    except Exception as e:
        processing_status[task_id]['status'] = 'error'
        processing_status[task_id]['error'] = str(e)
        # Clean up temp directory on error
        if temp_dir and os.path.exists(temp_dir):
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

def create_video_from_frame_paths(frame_paths, fps, output_path):
    """Create video from frame file paths"""
    if not frame_paths:
        return False
    
    # Read first frame to get dimensions
    first_frame = cv2.imread(frame_paths[0])
    if first_frame is None:
        return False
    
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_path in frame_paths:
        if os.path.exists(frame_path):
            frame = cv2.imread(frame_path)
            if frame is not None:
                out.write(frame)
    
    out.release()
    return True

def create_video_from_frames(frames, fps, output_path):
    """Create video from processed frames (fallback method)"""
    if not frames:
        return False
    
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame in frames:
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)
    
    out.release()
    return True

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if tester is None:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    is_video = is_video_file(file.filename)
    
    try:
        if is_video:
            return process_video_upload(file)
        else:
            return process_image_upload(file)
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_image_upload(file):
    """Process single image upload"""
    try:
        # Clear GPU memory before processing
        clear_gpu_memory()
        
        image_bytes = file.read()
        image_array = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'Invalid image format'}), 400
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        temp_path = 'temp_input.png'
        cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        with torch.no_grad():
            input_frames = tester.load_single_frame_as_triplet(temp_path)
            sr_output = tester.super_resolve(input_frames)
            
            input_img = tester.tensor_to_image(input_frames[0, :, 1, :, :])
            sr_img = tester.tensor_to_image(sr_output.squeeze(0))
        
        # Clear tensors
        del input_frames, sr_output
        clear_gpu_memory()
        
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
            'type': 'image',
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
        clear_gpu_memory()
        return jsonify({'error': f'Image processing failed: {str(e)}'}), 500

def process_video_upload(file):
    """Process video upload"""
    with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_video:
        file.save(tmp_video.name)
        temp_video_path = tmp_video.name
    
    try:
        # Extract frames with reduced limit for memory efficiency
        frames, fps = extract_frames_from_video(temp_video_path, max_frames=100)
        
        if len(frames) == 0:
            return jsonify({'error': 'Could not extract frames from video'}), 400
        
        task_id = str(int(time.time() * 1000))
        
        # Start processing in background thread
        thread = threading.Thread(target=process_video_frames, args=(frames, task_id))
        thread.start()
        
        return jsonify({
            'success': True,
            'type': 'video',
            'task_id': task_id,
            'total_frames': len(frames),
            'fps': fps,
            'message': 'Video processing started (memory-optimized mode)'
        })
        
    finally:
        os.unlink(temp_video_path)

@app.route('/progress/<task_id>')
def get_progress(task_id):
    """Get processing progress for video"""
    if task_id not in processing_status:
        return jsonify({'error': 'Task not found'}), 404
    
    status = processing_status[task_id].copy()
    
    # Don't send the actual frames in progress updates
    for key in ['frames', 'temp_dir', 'frame_paths']:
        if key in status:
            del status[key]
    
    return jsonify(status)

@app.route('/result/<task_id>')
def get_result(task_id):
    """Get final result for video processing"""
    if task_id not in processing_status:
        return jsonify({'error': 'Task not found'}), 404
    
    status = processing_status[task_id]
    
    if status['status'] != 'completed':
        return jsonify({'error': 'Processing not completed'}), 400
    
    try:
        frame_paths = status.get('frame_paths', [])
        
        # Create sample images for preview from middle frame
        if frame_paths:
            middle_idx = len(frame_paths) // 2
            if middle_idx < len(frame_paths):
                sample_frame_path = frame_paths[middle_idx]
                if os.path.exists(sample_frame_path):
                    sample_frame = cv2.imread(sample_frame_path)
                    sample_frame = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2RGB)
                    sample_b64 = image_to_base64(sample_frame)
                    
                    return jsonify({
                        'success': True,
                        'sample_input': sample_b64,
                        'sample_sr': sample_b64,
                        'total_frames': len(frame_paths),
                        'message': 'Video processing completed'
                    })
        
        return jsonify({'error': 'No frames available'}), 400
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_video/<task_id>')
def download_video(task_id):
    """Download processed video"""
    if task_id not in processing_status:
        return jsonify({'error': 'Task not found'}), 404
    
    status = processing_status[task_id]
    
    if status['status'] != 'completed':
        return jsonify({'error': 'Processing not completed'}), 400
    
    try:
        frame_paths = status.get('frame_paths', [])
        fps = 30
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_output:
            output_path = tmp_output.name
        
        if create_video_from_frame_paths(frame_paths, fps, output_path):
            def cleanup_files(paths, temp_dir, output_file):
                try:
                    # Clean up frame files and temp directory
                    if temp_dir and os.path.exists(temp_dir):
                        import shutil
                        shutil.rmtree(temp_dir, ignore_errors=True)
                    # Clean up output file after delay
                    time.sleep(60)
                    if os.path.exists(output_file):
                        os.unlink(output_file)
                except:
                    pass
            
            temp_dir = status.get('temp_dir')
            threading.Timer(1, cleanup_files, args=[frame_paths, temp_dir, output_path]).start()
            
            return send_file(output_path, 
                           as_attachment=True, 
                           download_name=f'ersvr_output_{task_id}.mp4',
                           mimetype='video/mp4')
        else:
            return jsonify({'error': 'Failed to create output video'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<result_type>')
def download_result(result_type):
    return jsonify({'error': 'Download not implemented for images'}), 501

if __name__ == '__main__':
    print("Initializing ERSVR Web Interface...")
    init_model()
    print("\nStarting web server...")
    print("Open your browser and go to: http://localhost:5000")
    print("Supports both image and video uploads!")
    print("Memory-optimized for GPU systems with limited VRAM")
    app.run(debug=True, host='0.0.0.0', port=5000) 