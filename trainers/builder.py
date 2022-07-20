import enum

class TrainerTypes(enum.Enum):
    SILOG = 1
    SILOG_CHAMFER = 2


def get_trainer(config, trainer_type=None):
    if config.model == 'localbins':
        from .localbins_trainer import Trainer
        return Trainer

    # other trainers here ...
    raise ValueError(f"Couldnt find trainer from given info: Model: {config.model}, Type: {trainer_type}")
    

