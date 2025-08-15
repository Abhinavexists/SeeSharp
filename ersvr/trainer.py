import torch.nn.functional as F
from tqdm import tqdm


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train the model for one epoch"""
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