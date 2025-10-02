import argparse
from pathlib import Path

from pre import Pre
from train import train_model

parser = argparse.ArgumentParser()
# model experiment name
parser.add_argument("-n", "--name", help="name of the model experiment", type=str)
# path to store model weights
parser.add_argument(
    "-o", help="directory to store model weights", type=Path, default=Path.cwd()
)
# the number of epochs
parser.add_argument("-e", "--epoch", help="number of epochs", type=int)
args = parser.parse_args()


p = Pre(epochs=args.epoch, exp_name=args.name, weights_dir=args.o)


# provide the dataset


# provide the dataloaders


# setup transforms


# loss function


# optimizer


# lr-scheduler


# model


# train
train_model(
    p=p,
    model=model,
    train_dataloader=train_dataloader,
    test_dataloader=test_dataloader,
    loss_fn=loss_fn,
    optimizer=optimizer,
    scheduler=scheduler,
    device=DEVICE,
)
