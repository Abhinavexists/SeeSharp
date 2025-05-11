import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models.ersvr import ERSVR
from dataset import VimeoDataset
import torch.nn.functional as F

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    with tqdm(dataloader, desc='Training') as pbar:
        for batch_idx, (lr_frames, hr_frames) in enumerate(pbar):
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
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    
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
    
    return total_loss / len(dataloader)

def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr

def main():
    # Check available GPUs
    print("Available GPUs:")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    # Select GPU 1 (RTX 3050)
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(1)  # Use GPU 1 (RTX 3050)
    
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
    
    # Initialize data loaders with reduced num_workers
    train_loader = DataLoader(
        VimeoDataset('archive', split_list='archive/sep_trainlist.txt'),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2  # Reduced workers
    )
    
    val_loader = DataLoader(
        VimeoDataset('archive', split_list='archive/sep_testlist.txt'),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2  # Reduced workers
    )
    
    # TensorBoard writer
    writer = SummaryWriter('runs/ersvr_training')
    
    # Training loop
    for epoch in range(num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        
        # Validate
        val_loss = validate(model, val_loader, criterion, device)
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        # Save checkpoint
        if (epoch + 1) % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, f'checkpoints/ersvr_epoch_{epoch+1}.pth')

if __name__ == '__main__':
    main() 