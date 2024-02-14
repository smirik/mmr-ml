from sklearn.ensemble import GradientBoostingClassifier
from asteroids import read_file

DEBUG = True

FEATURES_COMBINATIONS = [['a', 'e'], ['a', 'n']]

POSITIVE_OBJECTS_FILE = 'cache/4J-2S-1.csv'
NEGATIVE_OBJECTS_FILE = 'cache/4J-2S-1-negatives.csv'
TEST_SET_FILE = 'cache/4J-2S-1-classifying.csv'

asteroid_numbers = read_file(POSITIVE_OBJECTS_FILE)

train_sizes_lst = []
for i in range(1, 101, 1):
    train_sizes_lst.append({'positive': i, 'negative': asteroid_numbers[i] - i})

TRAIN_SIZES = train_sizes_lst

TEST_SIZES = [50000]

models_lst = {}
models_lst[f"GB1"] = GradientBoostingClassifier(
    n_estimators=100, max_depth=5, random_state=42, learning_rate=0.1, min_samples_split=9, max_features='sqrt', loss='exponential'
)
models_lst[f"GB2"] = GradientBoostingClassifier(
    n_estimators=100, max_depth=5, random_state=42, learning_rate=0.1, min_samples_split=3, max_features='sqrt', loss='exponential'
)

MODELS = models_lst
