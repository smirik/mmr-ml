import sys
import importlib.util
from display import display_results
from asteroids import read_file
from simulation import run_simulation
from typing import TYPE_CHECKING

TEST_LST = False


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
    TEST_LST: bool or list

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide the file path as an argument.")
        sys.exit(1)

    file_path = sys.argv[1]
    import_file_variables(file_path)

    OUTPUT_FILE = 'results.csv'
    if len(sys.argv) > 2:
        OUTPUT_FILE = sys.argv[2]

positives_lst = read_file(POSITIVE_OBJECTS_FILE)
negatives_lst = read_file(NEGATIVE_OBJECTS_FILE)

if TEST_LST:
    test_lst = TEST_LST
else:
    test_lst = read_file(TEST_SET_FILE)

results = run_simulation(
    positives_lst=positives_lst,
    negatives_lst=negatives_lst,
    test_lst=test_lst,
    test_sizes=TEST_SIZES,
    models=MODELS,
    features_combinations=FEATURES_COMBINATIONS,
    train_sizes=TRAIN_SIZES,
    debug=DEBUG,
)
display_results(results, save=True, output_file=OUTPUT_FILE)
