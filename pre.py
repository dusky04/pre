from pathlib import Path
from dataclasses import dataclass


# This class takes in all the necessary arguments necessary to use pytorch
@dataclass
class Pre:
    epochs: int
    exp_name: str
    weights_dir: Path
