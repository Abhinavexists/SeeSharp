import numpy as np
import torch
import torch.nn.functional as F


def calculate_psnr(img1, img2):
    """Calculate Peak Signal-to-Noise Ratio between two images"""
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
    window = create_window(window_size, sigma, img1.size(0)).to(img1.device)

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


def create_window(window_size, sigma, channels):
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