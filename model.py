import torch.nn as nn
import torchvision.models as models


def get_model(backbone_model, num_classes=2, fine_tune=False):
    model = models.__dict__[backbone_model](pretrained=True)

    if fine_tune:
        for param in model.parameters():
            param.requires_grad = False

    # Modify the final layer for binary classification
    if backbone_model.startswith("resnet"):
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        # Add a custom final layer for other models
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    return model
