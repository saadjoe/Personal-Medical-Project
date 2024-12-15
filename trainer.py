import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from utils import plot_training_metrics, save_model


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss, total_correct = 0, 0

    for images, labels in dataloader:
        images, labels = (
            images.to(device),
            labels.to(device).float(),
        )  # Convert labels to float for BCE loss
        optimizer.zero_grad()
        outputs = model(images).squeeze(1)  # Flatten output for BCE loss
        loss = F.binary_cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        total_correct += (
            ((outputs > 0.5) == labels).sum().item()
        )  # Threshold at 0.5 for binary classification

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / len(dataloader.dataset)
    return avg_loss, accuracy


def validate_epoch(model, dataloader, device):
    model.eval()
    total_loss, total_correct = 0, 0

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = (
                images.to(device),
                labels.to(device).float(),
            )  # Convert labels to float for BCE loss
            outputs = model(images).squeeze(1)  # Flatten output for BCE loss
            loss = F.binary_cross_entropy(outputs, labels)

            total_loss += loss.item() * images.size(0)
            total_correct += (
                ((outputs > 0.5) == labels).sum().item()
            )  # Threshold at 0.5

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / len(dataloader.dataset)
    return avg_loss, accuracy


def train_model(model, train_loader, val_loader, optimizer, device, epochs):
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)

        val_loss, val_acc = validate_epoch(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)

        print(
            f"Epoch: {epoch + 1}/{epochs} - Train Loss/Acc: {train_loss:.4f}/{train_acc:.4f} - "
            f"Val Loss/Acc: {val_loss:.4f}/{val_acc:.4f}"
        )

        scheduler.step()

    plot_training_metrics(
        train_losses, val_losses, train_accuracies, val_accuracies, epochs
    )

    save_model(model, "brain_tumor_model.pth")

    return model
