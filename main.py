import argparse
from pathlib import Path

import torch
from torch import nn
from torchvision import transforms

from config import Config
from models.resnet import build_resnet_lstm_model
from pre import Pre
from train import train_model
from utils import download_dataset, unzip_files
from dataset import get_dataloaders


def train_resnet_lstm(p: Pre, c: Config):
    # setup the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # setup the dataset
    DATASET_NAME = "CricketEC"
    CRICKET_EC_URL = "https://drive.google.com/file/d/1b1gKYveWSfAQB3S75Nq3t3MeiUYzDPkt/view?usp=sharing"

    download_dir = Path("zipped_data")
    if not Path(download_dir).exists():
        download_dataset(download_dir, CRICKET_EC_URL)
        unzip_files(download_dir, DATASET_NAME)

    # setup transforms
    train_transform = transforms.Compose(
        [
            transforms.ConvertImageDtype(torch.float32),
            transforms.RandomApply([transforms.RandomRotation(10)], p=0.3),
            transforms.RandomApply(
                [transforms.RandomAffine(0, translate=(0.1, 0.1))], p=0.3
            ),
            transforms.RandomApply([transforms.ColorJitter(0.2, 0.2, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply(
                [transforms.GaussianBlur(kernel_size=(5, 5))], p=0.5
            ),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    # setup dataloaders
    train_dataloader, test_dataloader = get_dataloaders(
        c, train_transform=train_transform, test_transform=test_transform
    )

    # setup model
    model = build_resnet_lstm_model(c).to(device)

    # loss function
    loss_fn = nn.CrossEntropyLoss()

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=c.LR)

    # lr-scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = 5)

    # train
    train_model(
        p=p,
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # model experiment name
    parser.add_argument("-n", "--name", help="name of the model experiment", type=str)
    # path to store model weights
    parser.add_argument(
        "-o", help="directory to store model weights", type=Path, default=Path.cwd()
    )
    args = parser.parse_args()

    # train our model
    c = Config()
    p = Pre(epochs=c.NUM_EPOCHS, exp_name=args.name, weights_dir=args.o)
    train_resnet_lstm(p, c)
