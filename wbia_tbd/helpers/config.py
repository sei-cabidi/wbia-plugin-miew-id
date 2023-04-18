import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

from dataclasses import asdict

def dataclass_to_dict(dataclass_instance):
    return asdict(dataclass_instance)

class DictClass:
    def __iter__(self):
        yield from dataclass_to_dict(self).items()

@dataclass
class SchedulerParams(DictClass):
    lr_start: float
    lr_max: float
    lr_min: float
    lr_ramp_ep: int
    lr_sus_ep: int
    lr_decay: float

@dataclass
class ModelParams(DictClass):
    model_name: str
    use_fc: bool
    fc_dim: int
    dropout: float
    loss_module: str
    s: float
    margin: float
    ls_eps: float
    theta_zero: float
    pretrained: bool
    n_classes: int

@dataclass
class TestParams():
    batch_size: int = 4
    fliplr: bool = False
    fliplr_view: List = field(default_factory=list)

@dataclass
class Config(DictClass):
    exp_name: str
    project_name: str
    DIM: Tuple[int, int]
    NUM_WORKERS: int
    TRAIN_BATCH_SIZE: int
    VALID_BATCH_SIZE: int
    EPOCHS: int
    SEED: int
    device: str
    model_name: str
    loss_module: str
    comment: str
    use_wandb: bool
    scheduler_params: SchedulerParams
    model_params: ModelParams
    test: TestParams

def get_config(file_path: str) -> Config:
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    config_dict['scheduler_params'] = SchedulerParams(**config_dict['scheduler_params'])
    config_dict['model_params'] = ModelParams(**config_dict['model_params'])
    config_dict['test'] = TestParams()
    config = Config(**config_dict)
    return config