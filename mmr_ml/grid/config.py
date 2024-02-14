import itertools
from asteroids import read_file, get_train_set, get_asteroids_data

features = ['a', 'e', 'sinI', 'n']  # All possible features
feature_indices = {'a': 0, 'e': 1, 'sinI': 2, 'n': 3}  # Map feature names to their indices
FEATURES_COMBINATIONS = [list(comb) for i in range(1, len(features) + 1) for comb in itertools.combinations(features, i)]

TEST_SIZE = 0.3

positives_lst = read_file('cache/positives.csv')
negatives_lst = read_file('cache/negatives.csv')
test_set = read_file('cache/classifying.csv')
train_size = {'positive': 150, 'negative': 60000}
