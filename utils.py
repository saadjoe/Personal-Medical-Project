import os
import random

import matplotlib.pyplot as plt
import torch
from PIL import Image


def plot_class_distribution(data_dirs, split_names):
    total_counts = {"yes": 0, "no": 0}
    split_counts = []

    # Calculate class counts per split
    for data_dir, split_name in zip(data_dirs, split_names):
        class_counts = {}
        for class_name in os.listdir(data_dir):
            class_dir = os.path.join(data_dir, class_name)
            if os.path.isdir(class_dir):
                count = len(os.listdir(class_dir))
                class_counts[class_name] = count
                total_counts[class_name] += count
        split_counts.append(class_counts)

        # Plot distribution for each split
        total = sum(class_counts.values())
        plt.figure(figsize=(8, 6))
        plt.bar(
            class_counts.keys(),
            [(count / total) * 100 for count in class_counts.values()],
            color=["skyblue", "salmon"],
        )
        plt.title(f"Class Distribution in {split_name} Set (Percentage)")
        plt.xlabel("Class")
        plt.ylabel("Percentage (%)")
        plt.show()

    # Plot overall distribution
    total_sum = sum(total_counts.values())
    plt.figure(figsize=(8, 6))
    plt.bar(
        total_counts.keys(),
        [(count / total_sum) * 100 for count in total_counts.values()],
        color=["skyblue", "salmon"],
    )
    plt.title("Overall Class Distribution (Percentage)")
    plt.xlabel("Class")
    plt.ylabel("Percentage (%)")
    plt.show()


def plot_image_size_distribution(data_dir):
    widths = []
    heights = []

    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                with Image.open(img_path) as img:
                    width, height = img.size
                    widths.append(width)
                    heights.append(height)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(widths, bins=30, color="skyblue")
    plt.title("Image Width Distribution")
    plt.xlabel("Width")
    plt.ylabel("Frequency")

    plt.subplot(1, 2, 2)
    plt.hist(heights, bins=30, color="salmon")
    plt.title("Image Height Distribution")
    plt.xlabel("Height")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()


def plot_sample_images(data_dir, num_samples=5):
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


def save_model(model, path):
    torch.save(model.state_dict(), path)
