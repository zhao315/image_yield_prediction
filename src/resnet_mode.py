#!/usr/bin/env python3
""" try resnet18 """
import json
import matplotlib.pyplot as plt
from matplotlib_inline.backend_inline import set_matplotlib_formats
set_matplotlib_formats("pdf")

import numpy as np
from scipy.stats import pearsonr
from tqdm import trange, tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, ConcatDataset, random_split

import torchvision
from torchvision import transforms

from cotton_dataset import CottonDataset
from utilies.visualize_learning_curve import visualize_learning_curve
from utilies.visualize_inference import visualize_inference


# random seed
seed = torch.Generator().manual_seed(421)


# load resnet18
model = torchvision.models.resnet18(weights="DEFAULT")

# redefine innput layer
model.conv1 = nn.Conv2d(42, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# redfine output layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, 1)


if __name__ == "__main__": 
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    transform = transforms.Compose([transforms.Normalize((0), (255))])

    # load datasets
    with open("src/dataset_conf.json") as fin:
        data_info = json.load(fin)

    full_data = []
    for full_data_dir, full_yield_dir, full_defoliation_day in zip(data_info["full_data_dir"], data_info["full_yield_dir"], data_info["full_defoliation_day"]):
        full_data.append(
            CottonDataset(full_data_dir, full_yield_dir, full_defoliation_day, normalize=True, img_transform=transform)
            )
    full_data = ConcatDataset(full_data)

    test_data = []
    for test_data_dir, test_yield_dir, test_defoliation_day in zip(data_info["test_data_dir"], data_info["test_yield_dir"], data_info["test_defoliation_day"]):
        test_data.append(
            CottonDataset(test_data_dir, test_yield_dir, test_defoliation_day, normalize=True, img_transform=transform)
            )
    test_data = ConcatDataset(test_data)

    # split training and validation data (default: 0.8:0.2)
    train_size = int(len(full_data) * 0.8)
    val_size = len(full_data) - train_size
    train_data, val_data = random_split(full_data, [train_size, val_size], generator=seed)

    # hyperparameters
    ###################################################################################### 
    num_epochs = 3
    batch_size = 4
    lr = 1e-7
    ###################################################################################### 

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    # training and validation model
    total_train_loss = []
    total_val_loss = []
    model = model.to(device)
    # early stopping
    best_loss = float("inf")
    patience = 3

    train_loader = tqdm(train_loader)
    val_loader = tqdm(val_loader)
    for epoch in range(num_epochs):
        # training
        model.train()
        epoch_train_loss = 0.0
        for idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.float)
            
            outputs = model(inputs)
            loss = loss_fn(outputs.view_as(targets), targets)
            epoch_train_loss += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            scheduler.step()
            
            train_loader.set_description(f"[{idx+1}/{epoch}]: train_loss: {loss.item():.3f}")
        total_train_loss.append(epoch_train_loss / (idx+1))

        # validation
        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for idx, (inputs, targets) in  enumerate(val_loader):
                inputs = inputs.to(device, dtype=torch.float)
                targets = targets.to(device,dtype=torch.float)

                outputs = model(inputs)
                loss = loss_fn(outputs.view_as(targets), targets)
                epoch_val_loss += loss.item()
                val_loader.set_description(f"[{idx+1}/{epoch}]: val_loss: {loss.item():.3f}")
            total_val_loss.append(epoch_val_loss / (idx+1))

        # early stopping
        if patience != 0:
            if epoch_val_loss < best_loss:
                best_loss = epoch_val_loss
                # save current model
                torch.save(
                    {"state_dict": model.state_dict()}, "saved_model.pt"
                )
                # reset patience number
                patience = 3
            else:
                patience -=1
        else:
            break
    
    # visualize learning curve
    visualize_learning_curve(total_train_loss, total_val_loss, "result")

    # visualize inference
    model = model.to("cpu")
    visualize_inference(model, "saved_model/saved_mode.pt", test_loader, "result")
