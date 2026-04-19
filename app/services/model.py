import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, ResNetForImageClassification


NUM_CLASSES = 4
CLASS_NAMES = ['overripe', 'ripe', 'rotten', 'unripe']


def load_model():
    model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

    # Freeze the model parameters
    for param in model.parameters():
        param.requires_grad = False


    # Replace the final classification layer
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features, NUM_CLASSES)
    )

    return model