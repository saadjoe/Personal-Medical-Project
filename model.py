import torch.nn as nn
import torchvision.models as models


def get_model(backbone_model, fine_tune=False):
    model = models.__dict__[backbone_model](pretrained=True)

    if fine_tune:
        for param in model.parameters():
            param.requires_grad = False

    # Modify the final layer for binary classification
    model.fc = nn.Linear(model.fc.in_features, 2)

    return model
