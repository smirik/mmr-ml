from display import display_results
from asteroids import read_asteroid_numbers
from simulation import run_simulation

from experiments.astrobook_initial_data import *

asteroid_numbers_set = set(read_asteroid_numbers(ASTEROID_NUMBERS_FILE))

results = run_simulation(TEST_SIZES, TRAIN_SIZES, asteroid_numbers_set, MODELS, FEATURES_COMBINATIONS, DEBUG)
display_results(results, save=True, output_file=OUTPUT_FILE)
