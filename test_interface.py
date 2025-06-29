import torch
import cv2
import numpy as np
import sys
import os
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from ersvr.models.student import StudentSRNet

sys.path.append('ersvr')

class ERSVRTester:
    def __init__(self, model_path, device=None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = self.load_model(model_path)
        
    def load_model(self, model_path):
        model = StudentSRNet(scale_factor=4).to(self.device)
        
        if os.path.exists(model_path):
            print(f"Loading model from: {model_path}")
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    print("Loaded model state dict from checkpoint")
                    if 'val_psnr' in checkpoint:
                        print(f"Model PSNR: {checkpoint['val_psnr']:.2f}")
                else:
                    model.load_state_dict(checkpoint)
                    print("Loaded model weights")
            except Exception as e:
                print(f"Error loading state dict: {e}")
                return None
        else:
            print(f"Warning: Model file {model_path} not found!")
            model = StudentSRNet(scale_factor=4).to(self.device)
            
        model.eval()
        return model
    
    def load_and_preprocess_frames(self, frame_paths):
        if len(frame_paths) != 3:
            raise ValueError("Exactly 3 frames are required (previous, current, next)")
        
        frames = []
        for frame_path in frame_paths:
            if not os.path.exists(frame_path):
                raise FileNotFoundError(f"Frame not found: {frame_path}")
            
            frame = cv2.imread(frame_path)
            if frame is None:
                raise ValueError(f"Failed to load image: {frame_path}")
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = frame.astype(np.float32) / 255.0
            frames.append(frame)
        
        frames_array = np.stack(frames, axis=0)
        frames_tensor = torch.from_numpy(frames_array).permute(3, 0, 1, 2)
        frames_tensor = frames_tensor.unsqueeze(0)
        
        return frames_tensor.to(self.device)
    
    def load_single_frame_as_triplet(self, frame_path):
        if not os.path.exists(frame_path):
            raise FileNotFoundError(f"Frame not found: {frame_path}")
        
        frame = cv2.imread(frame_path)
        if frame is None:
            raise ValueError(f"Failed to load image: {frame_path}")
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0        
        frames = [frame, frame, frame]

        frames_array = np.stack(frames, axis=0)
        frames_tensor = torch.from_numpy(frames_array).permute(3, 0, 1, 2)
        frames_tensor = frames_tensor.unsqueeze(0)
        
        return frames_tensor.to(self.device)
    
    def super_resolve(self, input_frames):
        with torch.no_grad():
            sr_output = self.model(input_frames)
            sr_output = torch.clamp(sr_output, 0, 1)
            
            return sr_output
    
    def tensor_to_image(self, tensor):
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        
        image = tensor.cpu().permute(1, 2, 0).numpy()
        image = (image * 255).astype(np.uint8)
        return image
    
    def create_comparison_visualization(self, input_frame, sr_output, save_path=None):
        if input_frame.dim() == 5:
            center_frame = input_frame[0, :, 1, :, :]
        else:
            center_frame = input_frame
        
        input_img = self.tensor_to_image(center_frame)
        sr_img = self.tensor_to_image(sr_output.squeeze(0))
        
        input_upsampled = cv2.resize(input_img, (sr_img.shape[1], sr_img.shape[0]), 
                                   interpolation=cv2.INTER_CUBIC)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(input_img)
        axes[0].set_title(f'Input LR\n{input_img.shape[1]}x{input_img.shape[0]}')
        axes[0].axis('off')
        
        axes[1].imshow(input_upsampled)
        axes[1].set_title(f'Bicubic Upsampling\n{input_upsampled.shape[1]}x{input_upsampled.shape[0]}')
        axes[1].axis('off')
        
        axes[2].imshow(sr_img)
        axes[2].set_title(f'ERSVR Super Resolution\n{sr_img.shape[1]}x{sr_img.shape[0]}')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Comparison saved to: {save_path}")
        
        plt.show()
        
        return input_img, input_upsampled, sr_img
    
    def test_single_image(self, image_path, output_dir='results'):
        print(f"Testing single image: {image_path}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        input_frames = self.load_single_frame_as_triplet(image_path)
        print(f"Input shape: {input_frames.shape}")
        
        print("Running super resolution...")
        sr_output = self.super_resolve(input_frames)
        print(f"Output shape: {sr_output.shape}")
        
        base_name = Path(image_path).stem
        comparison_path = os.path.join(output_dir, f'{base_name}_comparison.png')
        
        input_img, bicubic_img, sr_img = self.create_comparison_visualization(
            input_frames, sr_output, comparison_path
        )
        
        sr_path = os.path.join(output_dir, f'{base_name}_super_resolved.png')
        cv2.imwrite(sr_path, cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR))
        print(f"Super resolved image saved to: {sr_path}")
        
        self.calculate_metrics(bicubic_img, sr_img, input_img)
        
        return sr_img
    
    def test_frame_sequence(self, frame_paths, output_dir='results'):
        if len(frame_paths) != 3:
            raise ValueError("Exactly 3 frame paths required")
        
        print(f"Testing frame sequence: {frame_paths}")
        
        os.makedirs(output_dir, exist_ok=True)
        
        input_frames = self.load_and_preprocess_frames(frame_paths)
        print(f"Input shape: {input_frames.shape}")
        
        print("Running super resolution...")
        sr_output = self.super_resolve(input_frames)
        print(f"Output shape: {sr_output.shape}")
        
        base_name = Path(frame_paths[1]).stem
        comparison_path = os.path.join(output_dir, f'{base_name}_sequence_comparison.png')
        
        sr_img = self.create_comparison_visualization(
            input_frames, sr_output, comparison_path
        )
        
        sr_path = os.path.join(output_dir, f'{base_name}_sequence_super_resolved.png')
        cv2.imwrite(sr_path, cv2.cvtColor(sr_img, cv2.COLOR_RGB2BGR))
        print(f"Super resolved sequence saved to: {sr_path}")
        
        return sr_img
    
    def calculate_metrics(self, bicubic_img, sr_img, original_img):
        bicubic_float = bicubic_img.astype(np.float32) / 255.0
        sr_float = sr_img.astype(np.float32) / 255.0
        
        mse_bicubic = np.mean((bicubic_float - sr_float) ** 2)
        if mse_bicubic > 0:
            psnr = 20 * np.log10(1.0 / np.sqrt(mse_bicubic))
            print(f"PSNR improvement over bicubic: {psnr:.2f} dB")
        
        print(f"Original image size: {original_img.shape[:2]}")
        print(f"Super resolved size: {sr_img.shape[:2]}")
        print(f"Scale factor: {sr_img.shape[0] // original_img.shape[0]}x")


def main():
    parser = argparse.ArgumentParser(description='Test ERSVR Super Resolution Model')
    parser.add_argument('--model', type=str, default='student_models/student_best.pth', help='Path to trained model checkpoint')
    parser.add_argument('--image', type=str, help='Path to single test image')
    parser.add_argument('--frames', nargs=3, help='Paths to 3 consecutive frames')
    parser.add_argument('--output', type=str, default='results', help='Output directory for results')
    parser.add_argument('--device', type=str, choices=['cpu', 'cuda'], help='Device to run inference on')
    
    args = parser.parse_args()
    
    tester = ERSVRTester(args.model, args.device)
    
    if args.image:
        tester.test_single_image(args.image, args.output)
    elif args.frames:
        tester.test_frame_sequence(args.frames, args.output)
    else:
        print("Please provide either --image for single image testing or --frames for sequence testing")
        print("\nExample usage:")
        print("  python test_interface.py --image sample.jpg")
        print("  python test_interface.py --frames frame1.jpg frame2.jpg frame3.jpg")


if __name__ == '__main__':
    main() 