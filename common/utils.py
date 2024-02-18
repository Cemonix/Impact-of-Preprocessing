import random
from typing import Any, Dict


def get_random_from_min_max_dict(
    min_max_dict: Dict[str, Dict[str, int | float] | Any]
) -> Dict[str, int | float]:
    random_dict = {}
    for key, value in min_max_dict.items():
        if isinstance(value, Dict):
            random_dict[key] = random.uniform(value['min'], value['max'])
        else:
            random_dict[key] = value
    return random_dict
24242