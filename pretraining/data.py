from os.path import join

import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class RetinaData(Dataset):
    def __init__(self, path_col, label_col, tfms=None, subset=100, target_dtype=np.float32):
        if subset:
            self.df = self.df.sample(subset, random_state=123)
        self.path_col = path_col
        self.label_col = label_col
        self.tfms = tfms
        self.target_dtype = target_dtype

    def __len__(self):
        return len(self.df)

    def _load_item(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        p = self.df.iloc[idx][self.path_col]
        label = self.df.iloc[idx][self.label_col].astype(self.target_dtype)

        img = Image.open(p)
        return img, label

    def __getitem__(self, idx):
        img, label = self._load_item(idx)
        if self.tfms:
            img = self.tfms(img)
        return img, label

    def get_original_img(self, idx):
        img, label = self._load_item(idx)
        print('class label: %s, img shape: %s' % (label, img.size))
        return img, label


class AutoRetinaData(RetinaData):
    def __init__(self, img_dir, labels_path, tfms=None, subset=100):
        ds = DiabeticRetinopathyKaggle(img_dir=img_dir, labels_path=labels_path, tfms=tfms, subset=subset)
        self.df = ds.df
        path_col = ds.path_col
        super().__init__(
                path_col=path_col,
                label_col=None,
                tfms=tfms,
                subset=None,
                target_dtype=np.float32,
                )

    def _load_item(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        p = self.df.iloc[idx][self.path_col]

        img = Image.open(p)
        return img, None

    def __getitem__(self, idx):
        img, _ = self._load_item(idx)
        if self.tfms:
            img = self.tfms(img)
        return img, img


class DiabeticRetinopathyKaggle(RetinaData):
    def __init__(self, img_dir, labels_path, tfms=None, subset=100, target_dtype=np.float32):
        df = pd.read_csv(labels_path)
        df.image = [join(img_dir, p+'.jpeg') for p in df.image]
        self.df = df.drop([1146, ])[:subset]
        super().__init__(
                path_col='image',
                label_col='level',
                tfms=tfms,
                subset=subset,
                target_dtype=target_dtype,
                )


def get_tfms(size=224, interpolation=Image.BILINEAR):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    norm = transforms.Normalize(mean=mean, std=std)
    train = transforms.Compose([
        transforms.RandomRotation(degrees=180, resample=interpolation),
        transforms.Resize(size=size),
        transforms.ColorJitter(brightness=0.5, contrast=0.25, saturation=0.25, hue=0.03),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        norm,
        ])
    valid = transforms.Compose([
        transforms.Resize(size=size),
        transforms.ToTensor(),
        norm,
        ])

    return train, valid


def build_retina_dataset(
        img_dir,
        labels_path,
        ae=False,
        size=224,
        batch_size=32,
        num_workers=12,
        train_pct=0.8,
        subset=None,
        seed=123,
        ):
    train_tfm, valid_tfm = get_tfms(size=size, interpolation=Image.BILINEAR)
    if ae:
        train_ds = AutoRetinaData(img_dir=img_dir, labels_path=labels_path, tfms=train_tfm, subset=subset)
        valid_ds = AutoRetinaData(img_dir=img_dir, labels_path=labels_path, tfms=valid_tfm, subset=subset)
    else:
        train_ds = DiabeticRetinopathyKaggle(img_dir=img_dir, labels_path=labels_path, tfms=train_tfm, subset=subset)
        valid_ds = DiabeticRetinopathyKaggle(img_dir=img_dir, labels_path=labels_path, tfms=valid_tfm, subset=subset)

    m = len(train_ds)
    cut = int(train_pct * m)
    torch.manual_seed(seed)
    inds = torch.randperm(m)

    train_inds = inds[:cut]
    valid_inds = inds[cut:]

    train_sampler = SubsetRandomSampler(train_inds)
    valid_sampler = SubsetRandomSampler(valid_inds)
    train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=num_workers,
            )
    valid_loader = DataLoader(
            valid_ds,
            batch_size=batch_size,
            sampler=valid_sampler,
            num_workers=num_workers,
            )

    return train_loader, valid_loader
