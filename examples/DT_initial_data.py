from sklearn.tree import DecisionTreeClassifier
from asteroids import read_file

DEBUG = True

features = ['a', 'e', 'sinI', 'n']
FEATURES_COMBINATIONS = [
    ['a', 'n', 'sinI'],
]

POSITIVE_OBJECTS_FILE = 'cache/positives.csv'
NEGATIVE_OBJECTS_FILE = 'cache/negatives.csv'
TEST_SET_FILE = 'cache/classifying.csv'

asteroid_numbers = read_file(POSITIVE_OBJECTS_FILE)

train_sizes_lst = []
for i in range(1, 101, 1):
    train_sizes_lst.append({'positive': i, 'negative': asteroid_numbers[i] - i})

TRAIN_SIZES = train_sizes_lst

TEST_SIZES = [100000]

CLASS_WEIGHT_DICT = {0: 0.003, 1: 0.997}

models_lst = {}
for min_split in [3]:
    models_lst[f"DT"] = DecisionTreeClassifier(
        max_depth=None,
        min_samples_split=min_split,
        criterion='gini',
        class_weight=None,
        max_features='sqrt',
        random_state=42,
    )

MODELS = models_lst
