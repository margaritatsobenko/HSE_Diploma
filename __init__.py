from typing import Dict, Union


def get_base_loader_config(
    batch_size: int, shuffle: bool
) -> Dict[str, Union[int, bool]]:
    return {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "pin_memory": True,
    }
