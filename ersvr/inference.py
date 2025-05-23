import torch
import torch.nn.functional as F
import cv2
import numpy as np
from models.ersvr import ERSVR
from tqdm import tqdm
import argparse
from train import calculate_psnr, calculate_ssim, calculate_motion_consistency

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

def evaluate_video(model, input_path, gt_path, device):
    """
    Evaluate model performance on a video with ground truth
    Args:
        model: ERSVR model
        input_path: Path to input low-resolution video
        gt_path: Path to ground truth high-resolution video
        device: Device to run model on
    Returns:
        Dictionary of evaluation metrics
    """
    # Open video files
    cap_lr = cv2.VideoCapture(input_path)
    cap_hr = cv2.VideoCapture(gt_path)
    
    # Get video properties
    total_frames = min(
        int(cap_lr.get(cv2.CAP_PROP_FRAME_COUNT)),
        int(cap_hr.get(cv2.CAP_PROP_FRAME_COUNT))
    )
    
    # Initialize metrics
    metrics = {
        'psnr': 0.0,
        'ssim': 0.0,
        'moc': 0.0
    }
    
    # Initialize frame buffers
    lr_frame_buffer = []
    hr_frames = []
    sr_frames = []
    
    # Process video
    with tqdm(total=total_frames, desc='Evaluating video') as pbar:
        for _ in range(total_frames):
            # Read frames
            ret_lr, frame_lr = cap_lr.read()
            ret_hr, frame_hr = cap_hr.read()
            
            if not ret_lr or not ret_hr:
                break
            
            # Convert BGR to RGB
            frame_lr = cv2.cvtColor(frame_lr, cv2.COLOR_BGR2RGB)
            frame_hr = cv2.cvtColor(frame_hr, cv2.COLOR_BGR2RGB)
            
            # Normalize to [0, 1]
            frame_lr = frame_lr.astype(np.float32) / 255.0
            frame_hr = frame_hr.astype(np.float32) / 255.0
            
            # Add LR frame to buffer
            lr_frame_buffer.append(frame_lr)
            
            # Process when we have 3 frames
            if len(lr_frame_buffer) == 3:
                # Prepare input tensor
                input_frames = np.stack(lr_frame_buffer, axis=0)  # (3, H, W, 3)
                input_frames = np.transpose(input_frames, (3, 0, 1, 2))  # (3, 3, H, W)
                input_frames = torch.from_numpy(input_frames).unsqueeze(0).to(device)
                
                # Process with model
                with torch.no_grad():
                    output = model(input_frames)
                
                # Convert output to numpy
                output_np = output.squeeze(0).cpu().numpy()
                output_np = np.transpose(output_np, (1, 2, 0))
                output_np = np.clip(output_np, 0, 1)
                
                # Store frames for metrics calculation
                hr_frames.append(torch.from_numpy(frame_hr).permute(2, 0, 1))
                sr_frames.append(output.squeeze(0).cpu())
                
                # Remove oldest frame
                lr_frame_buffer.pop(0)
            
            pbar.update(1)
    
    # Release resources
    cap_lr.release()
    cap_hr.release()
    
    # Convert frame lists to tensors
    hr_tensor = torch.stack(hr_frames)  # (N, 3, H, W)
    sr_tensor = torch.stack(sr_frames)  # (N, 3, H, W)
    
    # Calculate metrics on full sequences
    psnr = 0
    ssim = 0
    
    # Calculate per-frame metrics
    for i in range(len(hr_frames)):
        psnr += calculate_psnr(sr_tensor[i], hr_tensor[i])
        ssim += calculate_ssim(sr_tensor[i], hr_tensor[i])
    
    # Average per-frame metrics
    metrics['psnr'] = psnr / len(hr_frames)
    metrics['ssim'] = ssim / len(hr_frames)
    
    # Calculate motion consistency on the full sequence
    metrics['moc'] = calculate_motion_consistency(sr_tensor, hr_tensor)
    
    return metrics

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='ERSVR Inference')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True, help='Input video path')
    parser.add_argument('--output', type=str, help='Output video path')
    parser.add_argument('--gt', type=str, help='Ground truth video path for evaluation')
    parser.add_argument('--evaluate', action='store_true', help='Run evaluation if ground truth is provided')
    args = parser.parse_args()
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ERSVR(scale_factor=4).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # If evaluation mode is enabled and ground truth is provided
    if args.evaluate and args.gt:
        print(f"Evaluating model on {args.input} with ground truth {args.gt}")
        metrics = evaluate_video(model, args.input, args.gt, device)
        print(f"Evaluation metrics:")
        print(f"PSNR: {metrics['psnr']:.2f} dB")
        print(f"SSIM: {metrics['ssim']:.4f}")
        print(f"MOC: {metrics['moc']:.4f}")
    
    # If output path is provided, process the video
    if args.output:
        print(f"Processing video {args.input} to {args.output}")
        process_video(model, args.input, args.output, device)
        print(f"Video processing completed!")

if __name__ == '__main__':
    main() 