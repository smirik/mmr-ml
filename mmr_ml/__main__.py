import sys
import importlib.util
from display import display_results
from asteroids import read_file
from simulation import run_simulation
from typing import TYPE_CHECKING


def import_file_variables(file_path):
    spec = importlib.util.spec_from_file_location("module_name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    globals().update(vars(module))


if TYPE_CHECKING:
    TEST_SIZES: list[int]
    TRAIN_SIZES: list[dict[str, int]]
    POSITIVE_OBJECTS_FILE: str
    NEGATIVE_OBJECTS_FILE: str
    TEST_SET_FILE: str
    MODELS: dict[str, object]
    FEATURES_COMBINATIONS: list[list[str]]
    DEBUG: bool

# from experiments.astrobook_initial_data import *

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the file path as an argument.")
        sys.exit(1)

    file_path = sys.argv[1]
    import_file_variables(file_path)

    OUTPUT_FILE = 'results.csv'
    if len(sys.argv) > 2:
        OUTPUT_FILE = sys.argv[2]

positives_set = set(read_file(POSITIVE_OBJECTS_FILE))
negatives_set = set(read_file(NEGATIVE_OBJECTS_FILE))
test_set = set(read_file(TEST_SET_FILE))

results = run_simulation(
    positives_set=positives_set,
    negatives_set=negatives_set,
    test_set=test_set,
    test_sizes=test_set,
    models=MODELS,
    features_combinations=FEATURES_COMBINATIONS,
    test_sizes=TEST_SIZES,
    train_sizes=TRAIN_SIZES,
    debug=DEBUG,
)
display_results(results, save=True, output_file=OUTPUT_FILE)
