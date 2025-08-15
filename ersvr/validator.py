import torch
import torch.nn.functional as F

from ersvr.metrics import calculate_metrics


def validate(model, dataloader, criterion, device):
    """Validate the model on the validation dataset"""
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