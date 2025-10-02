from dataclasses import dataclass
from pathlib import Path


# This class takes in all the necessary arguments necessary to use pytorch
@dataclass
class Pre:
    epochs: int
    exp_name: str
    weights_dir: Path
