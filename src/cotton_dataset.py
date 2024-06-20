#!/usr/bin/env python3
"""
cotton dataset:
    resolutio: 400 x 400
    DAP(day after planting): 14 (default)
"""
import os
import numpy as np
import pandas as pd
import rasterio

from datetime import datetime as dt

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class CottonDataset(Dataset):
    def __init__(self, img_dir, yield_dir, defol, days=14, normalize=False, img_transform=None, yield_transform=None):
        """
        create UAV imagery based pytorch dataset
        arguments:
            - img_dir: the directory of UAV imagery
            - yield_dir: the directory of yield values for each UAV imagery
            - defol: the defoliation day (format: YearMonthDay e.g. "20200717")
            - days: the numbers of days considered use as input, counting back from defoliation day
            - normalize: normalize the day indexes 
            - img_transform: transfom img data
            - yield_transform: transform yield data
        """
        super(CottonDataset, self).__init__()
        self.img_dir = img_dir
        self.yields = pd.read_csv(os.path.join(yield_dir, "final_yield.csv"))
        self.defol = defol
        self.days = days
        self.normalize = normalize
        self.img_transform = img_transform
        self.yield_transform = yield_transform
        self.resize = transforms.Resize((400, 400))
        self.date_format = "%Y%m%d"

    def __len__(self):
        return len(self.yields)
    
    def __getitem__(self, idx):
        # yield
        final_yield = self.yields.iloc[idx]        
        fid = str(int((final_yield["FID"])))
        value = final_yield["final_yield"]

        # image
        images = []
        
        image_files = sorted([img for img in os.listdir(os.path.join(self.img_dir, fid)) if not img.startswith(".")])
        # find the defoliation day index or the day closest to defoliaton day
        try:
            defol_index = image_files.index(self.defol[-4:])
        except ValueError:
            image_files.append(self.defol[-4:])
            image_files.sort()
            defol_index = image_files.index(self.defol[-4:]) - 1
        # find the dates to use
        image_files = image_files[defol_index - self.days + 1: defol_index + 1]

        if self.normalize:
            # calculate dates indexes, indexes start from 1
            date_indexes = [
                (dt.strptime(self.defol[:4] + image_files[i].split(".")[0], self.date_format) - dt.strptime(self.defol[:4] + image_files[0].split(".")[0], self.date_format)).days + 1 for i in range(len(image_files))
            ]
            date_indexes = np.array(date_indexes)
            normalized_date_indexes = date_indexes / date_indexes[-1]
        else:
            normalized_date_indexes = np.array([1] * len(image_files))

        # load images
        for idx, image in enumerate(image_files):
            with rasterio.open(os.path.join(self.img_dir, fid, image)) as fin:
                image = fin.read()[:3].astype(np.float32)
                image = image * normalized_date_indexes[idx]  # normalized
                image = self.resize(torch.tensor(image))
                images.append(image) 
        img = torch.cat(images)

        if self.img_transform:
            img = self.img_transform(img)
        if self.yield_transform:
            value = self.yield_transform(value)

        return img, value


if __name__ == "__main__":
    img_dir = "dataset/images/2020"
    yield_dir = "dataset/yields/2020"
    assert img_dir, "no such image files path"
    assert yield_dir, "no such yield file path"

    transform = transforms.Compose([transforms.Normalize((0), (255))])
    cotton_dataset_2020 = CottonDataset(img_dir=img_dir, yield_dir=yield_dir, defol="20200713", days= 10, normalize=True, img_transform=transform)
    # print(len(cotton_dataset_2020))


    img_dir = "dataset/images/2021"
    yield_dir = "dataset/yields/2021"
    cotton_dataset_2021 = CottonDataset(img_dir=img_dir, yield_dir=yield_dir, defol="20210727", days=10, normalize=True, img_transform=transform)
    # print(len(cotton_dataset_2021))

    from torch.utils.data import ConcatDataset
    concat_dataset = ConcatDataset([cotton_dataset_2020, cotton_dataset_2021])

    # print(len(concat_dataset))

    cotton_loader = DataLoader(concat_dataset, batch_size=16, shuffle=True)
    image, target = next(iter(cotton_loader))

    print(image.shape)
    # print(target)
    # print(image.max())
    # print(image.min())
    