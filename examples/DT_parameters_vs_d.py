from sklearn.tree import DecisionTreeClassifier
from asteroids import read_file

DEBUG = True

features = ['a', 'e', 'sinI', 'n']
FEATURES_COMBINATIONS = [
    ['n'],
    ['a', 'n', 'sinI'],
    ['a', 'sinI'],
]

POSITIVE_OBJECTS_FILE = 'cache/positives.csv'
NEGATIVE_OBJECTS_FILE = 'cache/negatives.csv'
TEST_SET_FILE = 'cache/classifying.csv'

asteroid_numbers = read_file(POSITIVE_OBJECTS_FILE)

train_sizes_lst = []
for i in [50]:
    train_sizes_lst.append({'positive': i, 'negative': asteroid_numbers[i] - i})

TRAIN_SIZES = train_sizes_lst

TEST_SIZES = [50000]

CLASS_WEIGHT_DICT = {0: 0.003, 1: 0.997}

models_lst = {}
for d in [None] + list(range(1, 101)):
    for min_split in range(2, 21, 1):
        for max_features in ['sqrt', 'log2', None]:
            models_lst[f"DT, d={d}, s={min_split}, f={max_features}"] = DecisionTreeClassifier(
                max_depth=d,
                min_samples_split=min_split,
                criterion='gini',
                class_weight=None,
                max_features=max_features,
                random_state=42,
            )

MODELS = models_lst
