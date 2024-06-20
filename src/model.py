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
    model = ConttonModel()

    from cotton_dataset import CottonDataset
    from torch.utils.data import DataLoader, ConcatDataset, random_split
    from torchvision.transforms import Normalize
    from tqdm import trange
    
    img_dirs = [ "dataset/images/2020",  "dataset/images/2021"]
    yield_dirs = [ "dataset/yields/2020", "dataset/yields/2021" ]
    defol_days = [ "20200713", "202010727" ]

    datasets = []
    for img_dir, yield_dir, defol_day in zip(img_dirs, yield_dirs, defol_days):
        datasets.append(CottonDataset(img_dir=img_dir, yield_dir=yield_dir, defol=defol_day, normalize=True, img_transform=Normalize((0), (255))))
    datasets = ConcatDataset(datasets)
    
    # data_loader = DataLoader(
        # dataset=datasets, batch_size=16, shuffle=True, num_workers=16
    # )

    # train and validation split
    seed = torch.Generator().manual_seed(412)
    train_size = int(len(datasets) * 0.8)
    val_size = len(datasets) - train_size
    train_set, val_set = random_split(datasets, [train_size, val_size], generator=seed)
    train_loader = DataLoader(
        train_set, batch_size=16, shuffle=True, num_workers=16
    )
    val_loader= DataLoader(
        val_set, batch_size=16, shuffle=False, num_workers=16
    )

    device = torch.device("cuda:1") if torch.cuda.is_available() else torch.device("cpu")
    print(f"using device: {device}")

    

    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-6)
    num_epochs = 2


    total_train_loss = []
    total_val_loss = []

    # train
    model.to(device)
    for epoch in trange(num_epochs):
        model.train()
        epoch_train_loss = 0.0
        for idx, data in enumerate(train_loader, 0):
            inputs, target = data
            inputs = inputs.to(device, dtype=torch.float32)
            target = target.to(device, dtype=torch.float32)

            outputs = model(inputs)
            loss = loss_fn(outputs.view_as(target), target)
            epoch_train_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Epoch {idx + 1}/{num_epochs}...... train_loss: {loss:.3f}")
        total_train_loss.append(epoch_train_loss / (idx + 1)) 

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for idx, data in enumerate(val_loader, 0):
                inputs, target = data
                inputs = inputs.to(device, dtype=torch.float32)
                target = target.to(device, dtype=torch.float32)

                outputs = model(inputs)
                loss = loss_fn(outputs.view_as(target), target)
                epoch_val_loss += loss.item()

                print(f"Epoch {idx+ 1}/{num_epochs} ...... val_loss: {loss:.3f}")
            total_val_loss.append(epoch_val_loss / (idx + 1))
    print("!!!Done!!!")

    # visualize
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(13, 5))
    plt.plot(total_train_loss);
    plt.plot(total_val_loss);
    plt.show();

    # test 
    img_dirs = [ "dataset/images/2022"]
    yield_dirs = [ "dataset/yields/2022" ]
    defol_days = [ "202020719" ]

    test_dataset = CottonDataset(img_dir=img_dir, yield_dir=yield_dir, defol=defol_day, normalize=True, img_transform=Normalize((0), (255)))
    test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    total_test_loss = []
    epoch_test_loss = 0.0
    with torch.no_grad():
        for idx, data in enumerate(val_loader, 0):
            inputs, target = data
            inputs = inputs.to(device, dtype=torch.float32)
            target = target.to(device, dtype=torch.float32)

            outputs = model(inputs)
            loss = loss_fn(outputs.view_as(target), target)
            epoch_test_loss += loss.item()

            print(f"Epoch {idx+ 1}/{num_epochs} ...... val_loss: {loss:.3f}")
    total_test_loss.append(epoch_test_loss / (idx + 1))

    plt.figure(figsize=(13, 5))
    plt.plot(total_test_loss);
    plt.show();