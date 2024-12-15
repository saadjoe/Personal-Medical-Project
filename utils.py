import os
import random

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay


def plot_sample_images(data_dir, num_samples=10):
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    fig.suptitle("Sample Images from Each Class")

    classes = os.listdir(data_dir)
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            sample_files = random.sample(
                os.listdir(class_dir), min(num_samples, len(os.listdir(class_dir)))
            )
            for j, img_name in enumerate(sample_files):
                img_path = os.path.join(class_dir, img_name)
                img = Image.open(img_path)
                axes[i, j].imshow(img)
                axes[i, j].axis("off")
                if j == 0:
                    axes[i, j].set_title(class_name)

    plt.show()


def count_images_in_sets(train_dir, val_dir, test_dir):
    data_counts = {"Set": [], "Class": [], "Count": []}

    for set_name, directory in zip(
        ["Train", "Validation", "Test"], [train_dir, val_dir, test_dir]
    ):
        for class_name in os.listdir(directory):
            class_dir = os.path.join(directory, class_name)
            if os.path.isdir(class_dir):
                count = len(os.listdir(class_dir))
                data_counts["Set"].append(set_name)
                data_counts["Class"].append(class_name.capitalize())
                data_counts["Count"].append(count)

    # Convert to DataFrame
    df = pd.DataFrame(data_counts)

    plt.figure(figsize=(10, 6))
    sns.barplot(
        x="Set", y="Count", hue="Class", data=df, palette=["#66c2a5", "#fc8d62"]
    )

    # Customize plot
    plt.title("Count of classes in each set")
    plt.xlabel("Set")
    plt.ylabel("Count")
    plt.legend(title="Class", loc="upper right")
    plt.show()


def plot_image_ratio_distribution(train_dir, val_dir, test_dir):
    data_ratios = {"Set": [], "Ratio": []}

    # Define function to calculate aspect ratios
    def calculate_ratios(directory, set_name):
        for class_folder in os.listdir(directory):
            class_path = os.path.join(directory, class_folder)
            if os.path.isdir(class_path):
                for image_name in os.listdir(class_path):
                    image_path = os.path.join(class_path, image_name)
                    try:
                        with Image.open(image_path) as img:
                            width, height = img.size
                            ratio = round(width / height, 2)  # Calculate aspect ratio
                            data_ratios["Set"].append(set_name)
                            data_ratios["Ratio"].append(ratio)
                    except Exception as e:
                        print(f"Error loading image {image_path}: {e}")

    # Calculate ratios for each dataset
    calculate_ratios(train_dir, "Train")
    calculate_ratios(val_dir, "Validation")
    calculate_ratios(test_dir, "Test")

    # Convert to DataFrame
    df = pd.DataFrame(data_ratios)

    # Plot the aspect ratio distribution
    plt.figure(figsize=(12, 6))
    sns.histplot(
        data=df,
        x="Ratio",
        hue="Set",
        multiple="stack",
        palette=["#66c2a5", "#fc8d62", "#8da0cb"],
        bins=20,
    )

    # Customize plot
    plt.title("Image Aspect Ratio Distribution Across Datasets")
    plt.xlabel("Aspect Ratio (Width / Height)")
    plt.ylabel("Count")
    plt.legend(title="Set", labels=["Train", "Validation", "Test"], loc="upper right")
    plt.show()


def save_model(model, path):
    torch.save(model.state_dict(), path)
    print(f"Model saved to {path}")


def plot_training_metrics(
    train_losses, val_losses, train_accuracies, val_accuracies, epochs
):
    plt.figure(figsize=(12, 5))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_accuracies, label="Train Accuracy")
    plt.plot(range(1, epochs + 1), val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.tight_layout()
    plt.show()


def display_test_metrics(accuracy, precision, recall, f1):
    print("\nTest Metrics:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")


def plot_confusion_matrix(conf_matrix, classes):
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=classes)
    disp.plot(cmap="Blues", values_format="d")
    plt.title("Confusion Matrix")
    plt.show()
