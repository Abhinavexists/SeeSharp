import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import numpy as np
from models.ersvr import ERSVR
from dataset import VimeoDataset
import torch.nn.functional as F

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    with tqdm(dataloader, desc='Training') as pbar:
        for batch_idx, (lr_frames, hr_frames) in enumerate(pbar):
            try:
                # Move data to device
                lr_frames = lr_frames.to(device)  # (B, 3, 3, H, W)
                hr_frames = hr_frames.to(device)  # (B, 3, H, W)
                
                # Forward pass
                optimizer.zero_grad()
                sr_output = model(lr_frames)  # (B, 3, H*4, W*4)
                
                # Upsample target to match output size
                hr_frames = F.interpolate(
                    hr_frames,
                    scale_factor=4,
                    mode='bicubic',
                    align_corners=False
                )
                
                # Calculate loss
                loss = criterion(sr_output, hr_frames)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Update progress
                total_loss += loss.item()
                pbar.set_postfix({'loss': total_loss / (batch_idx + 1)})
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue
    
    return total_loss / max(1, len(dataloader))

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

def calculate_ssim(img1, img2, window_size=11, sigma=1.5, L=1.0):
    """
    Calculate the Structural Similarity Index Measure (SSIM) between two images
    Both inputs should be in range [0, 1]
    """
    # Convert inputs to correct dimensions if necessary
    if img1.dim() == 4 and img1.size(0) == 1:  # Remove batch dim if batch size is 1
        img1 = img1.squeeze(0)
        img2 = img2.squeeze(0)
    
    # Check if input images are in the right shape [C, H, W]
    if img1.dim() != 3 or img2.dim() != 3:
        raise ValueError("Input images must be 3D tensors [C, H, W]")
    
    # Create a Gaussian kernel
    window = _create_window(window_size, sigma, img1.size(0)).to(img1.device)
    
    # Mean of img1, img2
    mu1 = F.conv2d(img1.unsqueeze(0), window, padding=window_size//2, groups=img1.size(0))
    mu2 = F.conv2d(img2.unsqueeze(0), window, padding=window_size//2, groups=img2.size(0))
    
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    
    # Variance of img1, img2
    sigma1_sq = F.conv2d(img1.unsqueeze(0) * img1.unsqueeze(0), window, padding=window_size//2, groups=img1.size(0)) - mu1_sq
    sigma2_sq = F.conv2d(img2.unsqueeze(0) * img2.unsqueeze(0), window, padding=window_size//2, groups=img2.size(0)) - mu2_sq
    sigma12 = F.conv2d(img1.unsqueeze(0) * img2.unsqueeze(0), window, padding=window_size//2, groups=img1.size(0)) - mu1_mu2
    
    # Constants for stability
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2
    
    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return ssim_map.mean()

def _create_window(window_size, sigma, channels):
    """Create a Gaussian window for SSIM calculation"""
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2 / (2 * sigma**2)) for x in range(window_size)])
    gauss = gauss / gauss.sum()
    
    _1D_window = gauss.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channels, 1, window_size, window_size).contiguous()
    return window

def calculate_motion_consistency(sr_batch, hr_batch):
    """
    Calculate motion consistency score between super-resolved and high-res frames
    Lower values indicate better motion consistency
    
    Args:
        sr_batch: Tensor of shape (B, 3, H, W) - batch of super-resolved frames
        hr_batch: Tensor of shape (B, 3, H, W) - batch of high-res ground truth frames
        
    Returns:
        Average motion consistency score
    """
    if sr_batch.size(0) < 2 or hr_batch.size(0) < 2:
        # Need at least 2 frames to calculate motion
        return 0.0
    
    # Calculate frame differences for both SR and HR sequences
    sr_diffs = torch.abs(sr_batch[1:] - sr_batch[:-1])
    hr_diffs = torch.abs(hr_batch[1:] - hr_batch[:-1])
    
    # Calculate difference in motion patterns
    motion_diff = torch.abs(sr_diffs - hr_diffs)
    
    # Average over all dimensions
    consistency_score = motion_diff.mean().item()
    
    # Convert to a 0-1 score where 1 is best (perfect consistency)
    # Using negative exponential: 1 - e^(-x) ranges from 0 to 1
    normalized_score = 1.0 - torch.exp(-10.0 * (1.0 - torch.tensor(consistency_score)))
    
    return normalized_score.item()

def calculate_metrics(sr_output, hr_frames):
    """Calculate various image quality metrics between SR output and HR frames"""
    # Ensure inputs are in the right shape and range
    psnr_val = calculate_psnr(sr_output, hr_frames)
    ssim_val = calculate_ssim(sr_output, hr_frames)
    
    # Motion consistency score - currently only applicable for batches
    moc_val = 0.0
    if sr_output.dim() > 3 and sr_output.size(0) > 1 and isinstance(sr_output, torch.Tensor) and isinstance(hr_frames, torch.Tensor):
        moc_val = calculate_motion_consistency(sr_output, hr_frames)
    
    return {
        'psnr': psnr_val.item() if isinstance(psnr_val, torch.Tensor) else psnr_val,
        'ssim': ssim_val.item() if isinstance(ssim_val, torch.Tensor) else ssim_val,
        'moc': moc_val,
    }

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    metrics = {'psnr': 0, 'ssim': 0, 'moc': 0}
    samples_count = 0
    
    with torch.no_grad():
        for lr_frames, hr_frames in dataloader:
            lr_frames = lr_frames.to(device)
            hr_frames = hr_frames.to(device)
            
            sr_output = model(lr_frames)
            
            # Upsample target to match output size
            hr_frames = F.interpolate(
                hr_frames,
                scale_factor=4,
                mode='bicubic',
                align_corners=False
            )
            
            loss = criterion(sr_output, hr_frames)
            total_loss += loss.item()
            
            # Calculate metrics for each image in batch
            for i in range(sr_output.size(0)):
                batch_metrics = calculate_metrics(sr_output[i], hr_frames[i])
                for k, v in batch_metrics.items():
                    metrics[k] += v
            
            samples_count += sr_output.size(0)
    
    # Average the metrics
    for k in metrics:
        metrics[k] /= samples_count
    
    # Add loss to metrics
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics

def check_dataset_structure(archive_path):
    """Check and analyze the dataset structure"""
    print(f"\n--- Dataset Structure Analysis ---")
    
    # Check if archive folder exists
    if not os.path.exists(archive_path):
        print(f"ERROR: {archive_path} does not exist!")
        return
    
    # List contents of archive folder
    print(f"Contents of {archive_path}:")
    archive_contents = os.listdir(archive_path)
    print(f"Found {len(archive_contents)} items: {archive_contents[:10]}...")
    
    # Check if sequence folder exists
    sequence_path = os.path.join(archive_path, 'sequence')
    if os.path.exists(sequence_path):
        print(f"Found sequence folder at {sequence_path}")
        
        # Count sequence folders
        sequence_dirs = [d for d in os.listdir(sequence_path) if os.path.isdir(os.path.join(sequence_path, d))]
        print(f"Found {len(sequence_dirs)} sequence directories")
        
        if sequence_dirs:
            # Check first sequence folder
            first_seq = sequence_dirs[0]
            first_seq_path = os.path.join(sequence_path, first_seq)
            subseqs = [d for d in os.listdir(first_seq_path) if os.path.isdir(os.path.join(first_seq_path, d))]
            print(f"Sequence {first_seq} contains {len(subseqs)} sub-sequences")
            
            if subseqs:
                # Check first sub-sequence
                first_subseq = subseqs[0]
                first_subseq_path = os.path.join(first_seq_path, first_subseq)
                files = os.listdir(first_subseq_path)
                print(f"Sub-sequence {first_seq}/{first_subseq} contains files: {files}")
    
    # Check for split lists
    train_list = os.path.join(archive_path, 'sep_trainlist.txt')
    test_list = os.path.join(archive_path, 'sep_testlist.txt')
    
    if os.path.exists(train_list):
        with open(train_list, 'r') as f:
            lines = f.readlines()
            print(f"Found sep_trainlist.txt with {len(lines)} entries")
            if lines:
                print(f"First 3 entries: {[line.strip() for line in lines[:3]]}")
    else:
        print(f"WARNING: {train_list} not found")
    
    if os.path.exists(test_list):
        with open(test_list, 'r') as f:
            lines = f.readlines()
            print(f"Found sep_testlist.txt with {len(lines)} entries")
    else:
        print(f"WARNING: {test_list} not found")
    
    print("--- End of Dataset Analysis ---\n")

def main():
    # Check available GPUs
    print("Available GPUs:")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Select appropriate GPU
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(1)  # Use GPU 1 if available
    
    # Analyze dataset structure
    archive_path = 'archive'  # Path to your dataset folder
    check_dataset_structure(archive_path)
    
    # Run dataset test first to validate the dataset
    try:
        print("\nRunning dataset test to validate structure...")
        import test_dataset
        test_dataset.test_data_loading(archive_path)
    except ImportError:
        print("Could not import test_dataset module. Skipping validation test.")
    
    # Hyperparameters
    batch_size = 2  # Reduced batch size
    num_epochs = 800
    learning_rate = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"Current GPU: {torch.cuda.current_device()}")
        print(f"GPU name: {torch.cuda.get_device_name()}")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Initialize model
    model = ERSVR(scale_factor=4).to(device)
    
    # Loss function and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999))
    
    # Make sure checkpoints directory exists
    os.makedirs('checkpoints', exist_ok=True)
    
    # Initialize data loaders with reduced num_workers
    train_list_path = os.path.join(archive_path, 'sep_trainlist.txt')
    test_list_path = os.path.join(archive_path, 'sep_testlist.txt')
    
    # Check if split lists exist, otherwise use the entire dataset
    use_split_list = os.path.exists(train_list_path) and os.path.exists(test_list_path)
    
    if use_split_list:
        print("Using train/test split lists")
        train_loader = DataLoader(
            VimeoDataset(archive_path, split_list=train_list_path),
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        
        val_loader = DataLoader(
            VimeoDataset(archive_path, split_list=test_list_path),
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
    else:
        print("WARNING: Split lists not found. Using entire dataset.")
        all_dataset = VimeoDataset(archive_path)
        
        # Split dataset manually (90% train, 10% val)
        train_size = int(0.9 * len(all_dataset))
        val_size = len(all_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            all_dataset, [train_size, val_size]
        )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
    
    # TensorBoard writer
    writer = SummaryWriter('runs/ersvr_training')
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs...")
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Training loss: {train_loss:.4f}")
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        print(f"Validation loss: {val_metrics['loss']:.4f}, PSNR: {val_metrics['psnr']:.2f}, SSIM: {val_metrics['ssim']:.4f}, MOC: {val_metrics['moc']:.4f}")
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_metrics['loss'], epoch)
        writer.add_scalar('Metrics/PSNR', val_metrics['psnr'], epoch)
        writer.add_scalar('Metrics/SSIM', val_metrics['ssim'], epoch)
        writer.add_scalar('Metrics/MOC', val_metrics['moc'], epoch)
        
        # Save checkpoint
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Saving checkpoint at epoch {epoch+1}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_metrics['loss'],
                'val_psnr': val_metrics['psnr'],
                'val_ssim': val_metrics['ssim'],
                'val_moc': val_metrics['moc'],
            }, f'checkpoints/ersvr_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    main() 