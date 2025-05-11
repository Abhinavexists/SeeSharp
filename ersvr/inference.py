import torch
import torch.nn.functional as F
import cv2
import numpy as np
from models.ersvr import ERSVR
from tqdm import tqdm

def process_video(model, input_path, output_path, device):
    """Process a video file using the ERSVR model"""
    # Open video file
    cap = cv2.VideoCapture(input_path)
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(
        output_path,
        fourcc,
        fps,
        (width * 4, height * 4)  # 4x upscaling
    )
    
    # Initialize frame buffer
    frame_buffer = []
    
    # Process video
    with tqdm(total=total_frames, desc='Processing video') as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            
            # Add to buffer
            frame_buffer.append(frame)
            
            # Process when we have 3 frames
            if len(frame_buffer) == 3:
                # Prepare input tensor
                input_frames = np.stack(frame_buffer, axis=0)  # (3, H, W, 3)
                input_frames = np.transpose(input_frames, (3, 0, 1, 2))  # (3, 3, H, W)
                input_frames = torch.from_numpy(input_frames).unsqueeze(0).to(device)
                
                # Process with model
                with torch.no_grad():
                    output = model(input_frames)
                
                # Convert output to numpy
                output = output.squeeze(0).cpu().numpy()
                output = np.transpose(output, (1, 2, 0))
                output = np.clip(output * 255, 0, 255).astype(np.uint8)
                
                # Convert RGB to BGR
                output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
                
                # Write frame
                out.write(output)
                
                # Remove oldest frame
                frame_buffer.pop(0)
            
            pbar.update(1)
    
    # Release resources
    cap.release()
    out.release()

def main():
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ERSVR(scale_factor=4).to(device)
    
    # Load checkpoint
    checkpoint = torch.load('checkpoints/ersvr_best.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Process video
    process_video(
        model,
        input_path='input_video.mp4',
        output_path='output_video.mp4',
        device=device
    )

if __name__ == '__main__':
    main() 