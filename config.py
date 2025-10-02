import os
from dataclasses import dataclass


@dataclass
class Config:
    DATASET_NAME = "CricketEC"
    TRAIN_SIZE = 0.8
    NUM_FRAMES = 32
    FRAME_SIZE = (224, 224)
    BATCH_SIZE = 8
    NUM_CLASSES = 14
    TRAIN_SIZE = 0.8
    NUM_WORKERS = min(6, os.cpu_count() or 0)
    LSTM_HIDDEN_DIM = 256
    LSTM_BIDIR = True
    LSTM_NUM_LAYERS = 1
    LR = 1e-3
    NUM_EPOCHS = 20
    WEIGHT_DECAY = 5e-4
    PREFETCH_FACTOR = 2
    LSTM_DROPOUT = 0.5
