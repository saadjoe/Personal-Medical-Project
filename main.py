import torch

from args import get_args
from dataset import get_dataloaders
from model import get_model
from trainer import train_epoch, validate_epoch
from utils import (
    plot_class_distribution,
    plot_image_size_distribution,
    plot_sample_images,
)


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Display sample images from the train set
    plot_sample_images(args.train_dir)

    # Visualize class distributions (with percentage and overall)
    plot_class_distribution(
        [args.train_dir, args.val_dir, args.test_dir], ["Train", "Validation", "Test"]
    )

    # Visualize image size distributions
    plot_image_size_distribution(args.train_dir)
    plot_image_size_distribution(args.val_dir)
    plot_image_size_distribution(args.test_dir)

    # Load data
    train_loader, val_loader, test_loader = get_dataloaders(
        args.train_dir, args.val_dir, args.test_dir, args.batch_size
    )

    # Initialize model and optimizer
    model = get_model(args.backbone_model, fine_tune=args.fine_tune).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Training loop
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, device)
        print(
            f"Epoch {epoch+1}/{args.epochs} - Train loss: {train_loss:.4f}, Train acc: {train_acc:.4f}, Val loss: {val_loss:.4f}, Val acc: {val_acc:.4f}"
        )


if __name__ == "__main__":
    main()
