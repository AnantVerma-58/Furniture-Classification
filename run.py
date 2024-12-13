import os
import sys
import warnings
warnings.filterwarnings('ignore')
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
from transform import transform

from model import create_model
from train import train_model
from evaluate import evaluate_model
from utils import *
import utils




def main(train_all:str, device:str, save:str, model:str):

    train_data = datasets.ImageFolder(root=os.path.join(final_data_folder, 'train'), transform=transform)
    test_data = datasets.ImageFolder(root=os.path.join(final_data_folder, 'test'), transform=transform)
    val_data = datasets.ImageFolder(root=os.path.join(final_data_folder, 'val'), transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    if train_all.lower()=='no':
        if model is None:model='densenet'
        model_names = [model]
    else:
        model_names = all_model_names

    for model_name in model_names:
        print(f"\nTraining {model_name} model:")
        
        model = create_model(model_name, num_classes=num_classes)
        model = model.to(device)

        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        if not trainable_params:
            raise ValueError("No trainable parameters found. Make sure final layers are unfrozen.")

        # Optimizer
        optimizer = optim.Adam(trainable_params, lr=lr)
        # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)

        # Train the model
        train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs, device=device)

        # Evaluate the model
        evaluate_model(model, test_loader, device=device)

        if save.lower() == 'yes' or save.lower()=='y':
            if os.path.isfile(save_path+"/"+model_name+'.pt'):
                print('Model Already Exists! Updating the model.')
                os.remove(save_path+"/"+model_name+'.pt')
            torch.save(model.state_dict(), f = save_path+"/"+model_name+'.pt')
            print(model_name_for_prediction+' Model saved at '+save_path+"/"+model_name+'.pt')

if __name__ == "__main__":
    

    # Parse the arguments
    parser = argparse.ArgumentParser(description="Example script to use --flag")

    parser.add_argument('--save', type=str, default='No', help="Need to save the trained model? (Y/N)")
    parser.add_argument('--trainall', type=str, default='No', help="Train all the models? (Y/N)")
    parser.add_argument('--model', type=str, default='densenet', help="Name of the model to train (e.g., densenet, resnet, vgg)")
    parser.add_argument('--device', type=str, default='cpu', help="Device to use (e.g., cpu, cuda)")


    args = parser.parse_args()

    model_name = args.model
    save = args.save
    train_all = args.trainall
    device = args.device    

    main(train_all=train_all, save=save, model=model_name, device=device)