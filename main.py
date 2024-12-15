import torch

from args import get_args
from dataset import get_dataloaders
from evaluate import evaluate_model
from model import get_model
from trainer import train_model
from utils import display_test_metrics, plot_confusion_matrix


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = get_dataloaders(
        args.train_dir, args.val_dir, args.test_dir, args.batch_size
    )

    model = get_model(fine_tune=args.fine_tune).to(device)
    optimizer = torch.optim.RMSprop(model.parameters(), lr=args.learning_rate)

    model = train_model(model, train_loader, val_loader, optimizer, device, args.epochs)

    test_acc, precision, recall, f1, confusion_matrix = evaluate_model(
        model, test_loader, device
    )

    display_test_metrics(test_acc, precision, recall, f1)

    plot_confusion_matrix(confusion_matrix, classes=["No Tumor", "Tumor"])


if __name__ == "__main__":
    main()
