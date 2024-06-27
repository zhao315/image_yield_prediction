#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class ConttonModel(nn.Module):
    def __init__(self):
        super(ConttonModel, self).__init__()
        self.conv_1 = nn.Conv2d(in_channels=42, out_channels=64, kernel_size=5)
        self.conv_2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5)
        self.conv_3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=5)
        self.conv_4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5)
        self.conv_5 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=5)

        self.pool = nn.MaxPool2d(2, 2)
        self.linear_1 = nn.Linear(1024 * 8 * 8, 2048)
        self.linear_2 = nn.Linear(2048, 96)
        self.linear_3 = nn.Linear(96, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv_1(x)))
        x = self.pool(F.relu(self.conv_2(x)))
        x = self.pool(F.relu(self.conv_3(x)))
        x = self.pool(F.relu(self.conv_4(x)))
        x = self.pool(F.relu(self.conv_5(x)))

        x = torch.flatten(x, 1)
        x = F.relu(self.linear_1(x))
        x = F.relu(self.linear_2(x))
        x = self.linear_3(x)

        return x
    

if __name__ == "__main__":
    from cotton_dataset import CottonDataset
    from torch.utils.data import DataLoader, ConcatDataset, random_split
    from torchvision import transforms
    from tqdm import trange, tqdm
    
    import matplotlib.pyplot as plt
    from matplotlib_inline.backend_inline import set_matplotlib_formats
    set_matplotlib_formats("pdf")

    import numpy as np
    from sklearn.metrics import mean_squared_error

    # hyper-parameters
    #############################################################################################
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    num_epochs = 2
    batch_size = 16
    lr = 1e-6

    transform = transforms.Compose([
        transforms.Normalize((0), (255)),
        transforms.RandomHorizontalFlip(),
    ])

    # training data
    img_dirs = [ "dataset/images/2020",  "dataset/images/2021"]
    yield_dirs = [ "dataset/yields/2020", "dataset/yields/2021" ]
    defol_days = [ "20200713", "20210727" ]

   # test data 
    test_img_dir = "dataset/images/2022" 
    test_yield_dir = "dataset/yields/2022"
    test_defol_day = "20220719" 
    #############################################################################################


    datasets = []
    for img_dir, yield_dir, defol_day in zip(img_dirs, yield_dirs, defol_days):
        datasets.append(CottonDataset(img_dir=img_dir, yield_dir=yield_dir, defol=defol_day, normalize=True, img_transform=transform))
    datasets = ConcatDataset(datasets)
    
    # split data: 0.8 to 0.2
    seed = torch.Generator().manual_seed(421)
    train_size = int(len(datasets) * 0.8)
    val_size =  int(len(datasets)) - train_size
    train_set, val_set = random_split(datasets, [train_size, val_size], generator=seed)

    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=False, num_workers=4)

    model = ConttonModel()
    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # training
    total_train_loss = []
    total_val_loss = []
    model.to(device)

    # early stopping
    best_loss = float("inf")
    patience = 5

    for epoch in trange(num_epochs):
        # training process
        model.train()
        epoch_train_loss = 0.0

        train_process_bar = tqdm(train_loader)
        for idx, data in enumerate(train_process_bar, 0):
            inputs, target = data
            inputs = inputs.to(device, dtype=torch.float32)
            target = target.to(device, dtype=torch.float32)

            outputs = model(inputs)
            loss = loss_fn(outputs.view_as(target), target)

            epoch_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_process_bar.set_description(f"Epoch {idx + 1}/{epoch}......train loss: {loss:.3f}")
        total_train_loss.append(epoch_train_loss / (idx + 1))

        # validation process
        model.eval()
        epoch_val_loss = 0.0

        val_process_bar= tqdm(val_loader)
        with torch.no_grad():
            for idx, data in enumerate(val_process_bar, 0):
                inputs, target = data
                inputs = inputs.to(device, dtype=torch.float32)
                target = target.to(device, dtype=torch.float32)

                outputs = model(inputs)
                loss = loss_fn(outputs.view_as(target), target)

                epoch_val_loss += loss.item()

                val_process_bar.set_description(f"Epoch {idx + 1}/{epoch}......validation loss: {loss:.3f}")
            total_val_loss.append(epoch_val_loss / (idx + 1))

        # early stopping
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            patience = 5
        else:
            patience -= 1
            if patience == 0:
                break

    # save model
    torch.save(
        { "state_dict": model.state_dict() },
        "model.pt",
    )    

    # save the training and validation curve
    plt.figure(figsize=(10, 5))
    plt.plot(total_train_loss, label="training loss")
    plt.plot(total_val_loss, label="validation loss")
    plt.grid(linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("train_validation.pdf")

    # test
    dataset = CottonDataset(img_dir=test_img_dir, yield_dir=test_yield_dir, defol=test_defol_day, normalize=True, img_transform=transform)
    test_loader = DataLoader(dataset=dataset, shuffle=False, num_workers=4)

    # load model
    model_state_dict = torch.load("model.pt")
    model.load_state_dict(model_state_dict["state_dict"])
    model.eval()
    # test processing
    preds = []
    targets = []
    with torch.no_grad():
        for inputs, target in tqdm(test_loader):
            inputs = inputs.to(device, dtype=torch.float32)
            targets.append(target.item())

            pred = model(inputs)
            preds.append(pred.item())

    preds = np.array(preds).reshape(-1, 1)
    targets = np.array(targets).reshape(-1, 1)

    # save the test result
    plt.figure(figsize=(5, 5))
    plt.scatter(preds, targets, facecolors="none", edgecolors="steelblue", alpha=0.5);
    plt.plot([0, 38], [0, 38], linestyle="--", color="red");
    plt.axis("square");
    plt.grid(linestyle="--", alpha=0.5);
    plt.title(f"mse: {mean_squared_error(targets, preds):.3f}")
    plt.xlabel("predictions")
    plt.ylabel("observations")
    plt.tight_layout()
    plt.savefig("preds_targets.pdf")
    print("!!!done!!!") 



