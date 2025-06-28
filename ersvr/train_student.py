import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import numpy as np
from models.student import StudentSRNet
from models.ersvr import ERSVR
from dataset import VimeoDataset
import torch.nn.functional as F

"""
Student Model Training Script with Knowledge Distillation
-------------------------------------------------------
- Loads a pre-trained teacher model (ERSVR)
- Trains a lightweight student model to mimic the teacher and ground truth
- Uses a combined loss: L1 to ground truth + distillation loss (MSE to teacher output)
- Saves the best student model by validation SSIM
- Prints PSNR and SSIM during validation
- Supports --full_dataset flag to use all data
"""

def calculate_ssim(img1, img2, window_size=11, sigma=1.5, L=1.0):
    """Calculate SSIM between two images (C, H, W) in [0,1]"""
    if img1.dim() == 4 and img1.size(0) == 1:
        img1 = img1.squeeze(0)
        img2 = img2.squeeze(0)
    if img1.dim() != 3 or img2.dim() != 3:
        raise ValueError("Input images must be 3D tensors [C, H, W]")
    window = _create_window(window_size, sigma, img1.size(0)).to(img1.device)
    mu1 = F.conv2d(img1.unsqueeze(0), window, padding=window_size//2, groups=img1.size(0))
    mu2 = F.conv2d(img2.unsqueeze(0), window, padding=window_size//2, groups=img2.size(0))
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1.unsqueeze(0) * img1.unsqueeze(0), window, padding=window_size//2, groups=img1.size(0)) - mu1_sq
    sigma2_sq = F.conv2d(img2.unsqueeze(0) * img2.unsqueeze(0), window, padding=window_size//2, groups=img2.size(0)) - mu2_sq
    sigma12 = F.conv2d(img1.unsqueeze(0) * img2.unsqueeze(0), window, padding=window_size//2, groups=img2.size(0)) - mu1_mu2
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item() if isinstance(psnr, torch.Tensor) else psnr

def _create_window(window_size, sigma, channels):
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2 / (2 * sigma**2)) for x in range(window_size)])
    gauss = gauss / gauss.sum()
    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channels, 1, window_size, window_size).contiguous()
    return window

def distillation_loss(student_out, teacher_out, gt, alpha=0.7):
    """
    Combined loss for knowledge distillation:
    - alpha: weight for distillation (student-teacher MSE)
    - (1-alpha): weight for L1 to ground truth
    """
    loss_gt = F.l1_loss(student_out, gt)
    loss_distill = F.mse_loss(student_out, teacher_out)
    return alpha * loss_distill + (1 - alpha) * loss_gt

def validate(model, dataloader, device):
    model.eval()
    total_ssim = 0
    total_psnr = 0
    total_loss = 0
    count = 0
    with torch.no_grad():
        for lr_frames, hr_frame in dataloader:
            lr_frames = lr_frames.to(device)
            hr_frame = hr_frame.to(device)
            student_out = model(lr_frames)
            hr_frame_up = F.interpolate(hr_frame, scale_factor=4, mode='bicubic', align_corners=False)
            loss = F.l1_loss(student_out, hr_frame_up)
            total_loss += loss.item() * lr_frames.size(0)
            for i in range(student_out.size(0)):
                ssim = calculate_ssim(student_out[i], hr_frame_up[i])
                psnr = calculate_psnr(student_out[i], hr_frame_up[i])
                total_ssim += ssim
                total_psnr += psnr
            count += student_out.size(0)
    return total_loss / count, total_ssim / count, total_psnr / count

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Train Student Model with Knowledge Distillation')
    parser.add_argument('--data_path', type=str, default='./archive', help='Path to dataset')
    parser.add_argument('--teacher_ckpt', type=str, default='../teacher_models/ersvr_best.pth', help='Path to teacher checkpoint')
    parser.add_argument('--output_path', type=str, default='./student_checkpoints', help='Where to save student models')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--alpha', type=float, default=0.7, help='Distillation loss weight')
    parser.add_argument('--max_sequences', type=int, default=None, help='Max sequences for fast training')
    parser.add_argument('--full_dataset', action='store_true', help='Use full dataset (overrides max_sequences)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load teacher (frozen)
    teacher = ERSVR(scale_factor=4).to(device)
    teacher_ckpt = torch.load(args.teacher_ckpt, map_location=device)
    teacher.load_state_dict(teacher_ckpt['model_state_dict'])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    # Load student
    student = StudentSRNet(scale_factor=4).to(device)

    # Dataset
    if args.full_dataset:
        train_dataset = VimeoDataset(args.data_path, sample_size=None, verbose=True)
        val_dataset = VimeoDataset(args.data_path, sample_size=None, verbose=False)
        # Split 90/10
        train_size = int(0.9 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    else:
        train_dataset = VimeoDataset(args.data_path, sample_size=args.max_sequences, verbose=True)
        val_dataset = VimeoDataset(args.data_path, sample_size=int(args.max_sequences*0.1) if args.max_sequences else 100, verbose=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Optimizer
    optimizer = optim.Adam(student.parameters(), lr=args.lr)

    # Training loop
    best_ssim = 0
    os.makedirs(args.output_path, exist_ok=True)
    for epoch in range(args.epochs):
        student.train()
        total_loss = 0
        with tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}') as pbar:
            for lr_frames, hr_frame in pbar:
                lr_frames = lr_frames.to(device)
                hr_frame = hr_frame.to(device)
                with torch.no_grad():
                    teacher_out = teacher(lr_frames)
                student_out = student(lr_frames)
                hr_frame_up = F.interpolate(hr_frame, scale_factor=4, mode='bicubic', align_corners=False)
                loss = distillation_loss(student_out, teacher_out, hr_frame_up, alpha=args.alpha)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * lr_frames.size(0)
                pbar.set_postfix({'loss': total_loss / ((pbar.n+1)*args.batch_size)})
        # Validation
        val_loss, val_ssim, val_psnr = validate(student, val_loader, device)
        print(f"Validation loss: {val_loss:.4f}, SSIM: {val_ssim:.4f}, PSNR: {val_psnr:.2f}")
        # Save best
        if val_ssim > best_ssim:
            best_ssim = val_ssim
            torch.save({'epoch': epoch, 'model_state_dict': student.state_dict()}, os.path.join(args.output_path, 'student_best.pth'))
            print(f"New best student model saved! SSIM: {best_ssim:.4f}")

if __name__ == '__main__':
    main()