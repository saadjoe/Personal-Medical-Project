import argparse


def get_args():
    parser = argparse.ArgumentParser(description="Brain Tumor Detection")

    # Model and training parameters
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--fine_tune", type=bool, default=True)

    # Directories
    parser.add_argument("--train_dir", type=str, default="data/train")
    parser.add_argument("--val_dir", type=str, default="data/val")
    parser.add_argument("--test_dir", type=str, default="data/test")
    parser.add_argument("--output_dir", type=str, default="outputs/")

    return parser.parse_args()
