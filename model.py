import torch.nn as nn
from torchvision import models

# Define function to create models
def create_model(model_name, num_classes=5):
    """
    Create and return a model based on the name and number of output classes.
    """
    if model_name == 'resnet':
        model = models.resnet50(weights=models.ResNet50_Weights)  # ResNet18 (can use ResNet50, ResNet101 as well)
        model.fc = nn.Linear(model.fc.in_features, num_classes)  # Modify final layer for custom classes
    
    elif model_name == 'vgg':
        model = models.vgg16(weights=models.VGG16_Weights)  # VGG16
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)  # Modify final layer
    
    elif model_name == 'densenet':
        model = models.densenet201(weights=models.DenseNet201_Weights)  # DenseNet121
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)  # Modify final layer
    
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    # Freeze all layers except the last ones
    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers
    
    # Unfreeze the last layer
    if model_name == 'resnet':
        model.fc.weight.requires_grad = True
    elif model_name == 'vgg':
        model.classifier[6].weight.requires_grad = True
    elif model_name == 'inceptionv3':
        model.fc.weight.requires_grad = True
    elif model_name == 'densenet':
        model.classifier.weight.requires_grad = True

    return model