import os
import shutil

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Define paths
IMG_PATH = "brain_tumor_dataset/"


def split_data():
    # Split the data into training, validation, and testing sets
    for CLASS in os.listdir(IMG_PATH):
        if not CLASS.startswith("."):
            class_dir = os.path.join(IMG_PATH, CLASS)
            os.makedirs(f"data/test/{CLASS.upper()}", exist_ok=True)
            os.makedirs(f"data/train/{CLASS.upper()}", exist_ok=True)
            os.makedirs(f"data/val/{CLASS.upper()}", exist_ok=True)

            IMG_NUM = len(os.listdir(class_dir))
            for n, FILE_NAME in enumerate(os.listdir(class_dir)):
                img = os.path.join(class_dir, FILE_NAME)
                if n < 5:
                    shutil.copy(img, f"data/test/{CLASS.upper()}/{FILE_NAME}")
                elif n < 0.8 * IMG_NUM:
                    shutil.copy(img, f"data/train/{CLASS.upper()}/{FILE_NAME}")
                else:
                    shutil.copy(img, f"data/val/{CLASS.upper()}/{FILE_NAME}")


# Define the BrainMRIDataset class
class BrainMRIDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Load all image paths and labels
        for label in os.listdir(data_dir):
            label_dir = os.path.join(data_dir, label)
            if os.path.isdir(label_dir):  # Check if it's a directory
                for img_file in os.listdir(label_dir):
                    self.image_paths.append(os.path.join(label_dir, img_file))
                    self.labels.append(
                        1 if label.lower() == "yes" else 0
                    )  # Map 'yes' to 1 and 'no' to 0

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


def get_dataloaders(train_dir, val_dir, test_dir, batch_size):
    # Define the transformation for the images
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    # Load datasets
    train_dataset = BrainMRIDataset(train_dir, transform=transform)
    val_dataset = BrainMRIDataset(val_dir, transform=transform)
    test_dataset = BrainMRIDataset(test_dir, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
