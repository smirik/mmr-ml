import astdys.astdys
from typing import List, Union

# Set catalog type to synthetic for proper elements
astdys.astdys.catalog_type = 'synthetic'


# Fetch data for asteroids
def get_asteroids_data(asteroids: List[int]) -> dict[str, dict[str, Union[float, int]]]:
    return astdys.astdys.search(asteroids)


def get_train_set(train_size: dict[str, int], positives_set: set[int], negatives_set: set[int]) -> List[int]:
    positive_samples = list(positives_set)[: train_size['positive']]
    negative_samples = list(negatives_set)[: train_size['negative']]

    return positive_samples + negative_samples


# Read asteroid numbers from file
def read_file(file_path: str) -> List[int]:
    with open(file_path, 'r') as file:
        return [int(line.strip()) for line in file]
