import astdys.astdys
from typing import List, Union

# Set catalog type to synthetic for proper elements
astdys.astdys.catalog_type = 'synthetic'


# Fetch data for asteroids
def get_asteroids_data(asteroids: List[int]) -> dict[str, dict[str, Union[float, int]]]:
    return astdys.astdys.search(asteroids)


# Read asteroid numbers from file
def read_asteroid_numbers(file_path: str) -> List[int]:
    with open(file_path, 'r') as file:
        return [int(line.strip()) for line in file]
