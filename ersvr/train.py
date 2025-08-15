import argparse
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ersvr.models.ersvr import ERSVR
from ersvr.dataset import VimeoDataset


def parse_args():
    parser = argparse.ArgumentParser(description="ERSVR Training Script")
    parser.add_argument(
        "--data_path",
        type=str,
        default="./archive",
        help="Path to the dataset directory (default: ./archive)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="./checkpoints",
        help="Path to save checkpoints (default: ./checkpoints)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size for training (default: 2)"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=800,
        help="Number of training epochs (default: 800)",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of dataloader workers (default: 2)",
    )
    parser.add_argument(
        "--gpu_id", type=int, default=None, help="GPU ID to use (default: auto-select)"
    )
    parser.add_argument(
        "--max_sequences",
        type=int,
        default=None,
        help="Maximum number of sequences to use for training (default: None)",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )
    parser.add_argument(
        "--tensorboard_dir",
        type=str,
        default="runs/ersvr_training",
        help="TensorBoard log directory (default: runs/ersvr_training)",
    )
    return parser.parse_args()


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    with tqdm(dataloader, desc="Training") as pbar:
        for batch_idx, (lr_frames, hr_frames) in enumerate(pbar):
            try:
                # Move data to device
                lr_frames = lr_frames.to(device)  # (B, 3, 3, H, W)
                hr_frames = hr_frames.to(device)  # (B, 3, H, W)

                # Forward pass
                optimizer.zero_grad()
                sr_output = model(lr_frames)  # (B, 3, H*4, W*4)

                hr_frames = F.interpolate(
                    hr_frames, scale_factor=4, mode="bicubic", align_corners=False
                )

                loss = criterion(sr_output, hr_frames)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                pbar.set_postfix({"loss": total_loss / (batch_idx + 1)})
            except Exception as e:
                print(f"Error in batch {batch_idx}: {e}")
                continue

    return total_loss / max(1, len(dataloader))


def calculate_psnr(img1, img2):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")
    max_pixel = 1.0
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr


def calculate_ssim(img1, img2, window_size=11, sigma=1.5, L=1.0):
    """
    Calculate the Structural Similarity Index Measure (SSIM) between two images
    Both inputs should be in range [0, 1]
    """
    if img1.dim() == 4 and img1.size(0) == 1:
        img1 = img1.squeeze(0)
        img2 = img2.squeeze(0)

    # Check if input images are in the right shape [C, H, W]
    if img1.dim() != 3 or img2.dim() != 3:
        raise ValueError("Input images must be 3D tensors [C, H, W]")

    # Create a Gaussian kernel
    window = _create_window(window_size, sigma, img1.size(0)).to(img1.device)

    mu1 = F.conv2d(
        img1.unsqueeze(0), window, padding=window_size // 2, groups=img1.size(0)
    )
    mu2 = F.conv2d(
        img2.unsqueeze(0), window, padding=window_size // 2, groups=img2.size(0)
    )

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = (
        F.conv2d(
            img1.unsqueeze(0) * img1.unsqueeze(0),
            window,
            padding=window_size // 2,
            groups=img1.size(0),
        )
        - mu1_sq
    )
    sigma2_sq = (
        F.conv2d(
            img2.unsqueeze(0) * img2.unsqueeze(0),
            window,
            padding=window_size // 2,
            groups=img2.size(0),
        )
        - mu2_sq
    )
    sigma12 = (
        F.conv2d(
            img1.unsqueeze(0) * img2.unsqueeze(0),
            window,
            padding=window_size // 2,
            groups=img1.size(0),
        )
        - mu1_mu2
    )

    # Constants for stability
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    # SSIM formula
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    )

    return ssim_map.mean()


def _create_window(window_size, sigma, channels):
    """Create a Gaussian window for SSIM calculation"""
    gauss = torch.Tensor(
        [
            np.exp(-((x - window_size // 2) ** 2) / (2 * sigma**2))
            for x in range(window_size)
        ]
    )
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
        return 0.0

    sr_diffs = torch.abs(sr_batch[1:] - sr_batch[:-1])
    hr_diffs = torch.abs(hr_batch[1:] - hr_batch[:-1])

    motion_diff = torch.abs(sr_diffs - hr_diffs)
    consistency_score = motion_diff.mean().item()

    # Convert to a 0-1 score where 1 is best (perfect consistency)
    # Using negative exponential: 1 - e^(-x) ranges from 0 to 1
    normalized_score = 1.0 - torch.exp(-10.0 * (1.0 - torch.tensor(consistency_score)))

    return normalized_score.item()


def calculate_metrics(sr_output, hr_frames):
    """Calculate various image quality metrics between SR output and HR frames"""
    psnr_val = calculate_psnr(sr_output, hr_frames)
    ssim_val = calculate_ssim(sr_output, hr_frames)

    # Motion consistency score - currently only applicable for batches
    moc_val = 0.0
    if (
        sr_output.dim() > 3
        and sr_output.size(0) > 1
        and isinstance(sr_output, torch.Tensor)
        and isinstance(hr_frames, torch.Tensor)
    ):
        moc_val = calculate_motion_consistency(sr_output, hr_frames)

    return {
        "psnr": psnr_val.item() if isinstance(psnr_val, torch.Tensor) else psnr_val,
        "ssim": ssim_val.item() if isinstance(ssim_val, torch.Tensor) else ssim_val,
        "moc": moc_val,
    }


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    metrics = {"psnr": 0.0, "ssim": 0.0, "moc": 0.0}
    samples_count = 0

    with torch.no_grad():
        for lr_frames, hr_frames in dataloader:
            lr_frames = lr_frames.to(device)
            hr_frames = hr_frames.to(device)

            sr_output = model(lr_frames)

            # Upsample target to match output size
            hr_frames = F.interpolate(
                hr_frames, scale_factor=4, mode="bicubic", align_corners=False
            )

            loss = criterion(sr_output, hr_frames)
            total_loss += loss.item()

            for i in range(sr_output.size(0)):
                batch_metrics = calculate_metrics(sr_output[i], hr_frames[i])
                for k, v in batch_metrics.items():
                    metrics[k] += v

            samples_count += sr_output.size(0)

    for k in metrics:
        metrics[k] /= samples_count

    metrics["loss"] = total_loss / len(dataloader)
    return metrics


def check_dataset_structure(data_path):
    """Check and analyze the dataset structure"""
    print("\n--- Dataset Structure Analysis ---")

    if not os.path.exists(data_path):
        print(f"ERROR: {data_path} does not exist!")
        return

    print(f"Contents of {data_path}:")
    data_contents = os.listdir(data_path)
    print(f"Found {len(data_contents)} items: {data_contents[:10]}...")

    sequence_path = os.path.join(data_path, "sequences")
    if os.path.exists(sequence_path):
        print(f"Found sequence folder at {sequence_path}")

        # Count sequence folders
        sequence_dirs = [
            d
            for d in os.listdir(sequence_path)
            if os.path.isdir(os.path.join(sequence_path, d))
        ]
        print(f"Found {len(sequence_dirs)} sequence directories")

        if sequence_dirs:
            # Check first sequence folder
            first_seq = sequence_dirs[0]
            first_seq_path = os.path.join(sequence_path, first_seq)
            subseqs = [
                d
                for d in os.listdir(first_seq_path)
                if os.path.isdir(os.path.join(first_seq_path, d))
            ]
            print(f"Sequence {first_seq} contains {len(subseqs)} sub-sequences")

            if subseqs:
                # Check first sub-sequence
                first_subseq = subseqs[0]
                first_subseq_path = os.path.join(first_seq_path, first_subseq)
                files = os.listdir(first_subseq_path)
                print(
                    f"Sub-sequence {first_seq}/{first_subseq} contains files: {files}"
                )

    train_list = os.path.join(data_path, "sep_trainlist.txt")
    test_list = os.path.join(data_path, "sep_testlist.txt")

    if os.path.exists(train_list):
        with open(train_list) as f:
            lines = f.readlines()
            print(f"Found sep_trainlist.txt with {len(lines)} entries")
            if lines:
                print(f"First 3 entries: {[line.strip() for line in lines[:3]]}")
    else:
        print(f"WARNING: {train_list} not found")

    if os.path.exists(test_list):
        with open(test_list) as f:
            lines = f.readlines()
            print(f"Found sep_testlist.txt with {len(lines)} entries")
    else:
        print(f"WARNING: {test_list} not found")

    print("--- End of Dataset Analysis ---\n")


def setup_device(gpu_id=None):
    """Setup the computing device"""
    if torch.cuda.is_available():
        print("Available GPUs:")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

        if gpu_id is not None and gpu_id < torch.cuda.device_count():
            torch.cuda.set_device(gpu_id)
            device = torch.device(f"cuda:{gpu_id}")
            print(f"Using specified GPU {gpu_id}: {torch.cuda.get_device_name(gpu_id)}")
        elif torch.cuda.device_count() > 1:
            torch.cuda.set_device(
                1
            )  # Use GPU 1 if available and no specific GPU specified
            device = torch.device("cuda:1")
            print(f"Using GPU 1: {torch.cuda.get_device_name(1)}")
        else:
            device = torch.device("cuda:0")
            print(f"Using GPU 0: {torch.cuda.get_device_name(0)}")

        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        print("CUDA not available, using CPU")

    return device


def main():
    args = parse_args()

    print("ERSVR Training Script")
    print(f"Data path: {args.data_path}")
    print(f"Output path: {args.output_path}")
    print(f"Batch size: {args.batch_size}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")

    device = setup_device(args.gpu_id)
    check_dataset_structure(args.data_path)

    try:
        print("\nRunning dataset test to validate structure...")
        import test_dataset

        test_dataset.test_data_loading(args.data_path)
    except ImportError:
        print("Could not import test_dataset module. Skipping validation test.")

    os.makedirs(args.output_path, exist_ok=True)

    model = ERSVR(scale_factor=4).to(device)

    criterion = nn.L1Loss()
    optimizer = optim.Adam(
        model.parameters(), lr=args.learning_rate, betas=(0.9, 0.999)
    )

    start_epoch = 0

    # Resume from checkpoint if specified
    if args.resume:
        if os.path.isfile(args.resume):
            print(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"Resumed from epoch {start_epoch}")
        else:
            print(f"No checkpoint found at '{args.resume}'")

    # Initialize data loaders
    train_list_path = os.path.join(args.data_path, "sep_trainlist.txt")
    test_list_path = os.path.join(args.data_path, "sep_testlist.txt")

    use_split_list = os.path.exists(train_list_path) and os.path.exists(test_list_path)

    if use_split_list:
        print("Using train/test split lists")
        train_loader = DataLoader(
            VimeoDataset(
                args.data_path,
                split_list=train_list_path,
                max_sequences=args.max_sequences,
            ),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

        val_loader = DataLoader(
            VimeoDataset(
                args.data_path,
                split_list=test_list_path,
                max_sequences=args.max_sequences // 10 if args.max_sequences else None,
            ),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )
    else:
        print("WARNING: Split lists not found. Using entire dataset.")
        all_dataset = VimeoDataset(args.data_path, max_sequences=args.max_sequences)

        # Split dataset manually (90% train, 10% val)
        train_size = int(0.9 * len(all_dataset))
        val_size = len(all_dataset) - train_size

        train_dataset, val_dataset = torch.utils.data.random_split(
            all_dataset, [train_size, val_size]
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
        )

    writer = SummaryWriter(args.tensorboard_dir)

    print(f"Starting training for {args.num_epochs} epochs...")
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")

        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Training loss: {train_loss:.4f}")

        val_metrics = validate(model, val_loader, criterion, device)
        print(
            f"Validation loss: {val_metrics['loss']:.4f}, PSNR: {val_metrics['psnr']:.2f}, SSIM: {val_metrics['ssim']:.4f}, MOC: {val_metrics['moc']:.4f}"
        )

        # Log metrics
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("Metrics/PSNR", val_metrics["psnr"], epoch)
        writer.add_scalar("Metrics/SSIM", val_metrics["ssim"], epoch)
        writer.add_scalar("Metrics/MOC", val_metrics["moc"], epoch)

        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Saving checkpoint at epoch {epoch + 1}")
            checkpoint_path = os.path.join(
                args.output_path, f"ersvr_epoch_{epoch + 1}.pth"
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_metrics["loss"],
                    "val_psnr": val_metrics["psnr"],
                    "val_ssim": val_metrics["ssim"],
                    "val_moc": val_metrics["moc"],
                },
                checkpoint_path,
            )

    final_model_path = os.path.join(args.output_path, "ersvr_final.pth")
    torch.save(
        {
            "epoch": args.num_epochs - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        final_model_path,
    )
    print(f"Final model saved to {final_model_path}")

    writer.close()
    print("Training completed!")


if __name__ == "__main__":
    main()
