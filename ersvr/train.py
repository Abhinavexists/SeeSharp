import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ersvr.dataset import VimeoDataset
from ersvr.models.ersvr import ERSVR
from ersvr.trainer import train_epoch
from ersvr.utils import check_dataset_structure, setup_device
from ersvr.validator import validate


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
