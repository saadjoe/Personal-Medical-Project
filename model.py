import torch.nn as nn
import torchvision.models as models


def get_model(fine_tune=False):
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    # Freeze all layers except the final layers for fine-tuning if needed
    if fine_tune:
        for param in model.features.parameters():
            param.requires_grad = False

    model.classifier[6] = nn.Sequential(
        nn.Linear(model.classifier[6].in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(256, 1),
        nn.Sigmoid(),
    )

    return model
