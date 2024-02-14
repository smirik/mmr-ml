from sklearn.tree import DecisionTreeClassifier
from asteroids import read_file

DEBUG = True

features = ['a', 'e', 'sinI', 'n']
FEATURES_COMBINATIONS = [
    ['a'],
    ['n'],
    ['a', 'n'],
    ['e', 'n'],
    ['a', 'sinI'],
    ['a', 'e'],
    ['sinI', 'n'],
    ['a', 'e', 'n'],
    ['a', 'e', 'sinI', 'n'],
    ['a', 'e', 'sinI'],
    ['a', 'n', 'sinI'],
]

POSITIVE_OBJECTS_FILE = 'cache/4J-2S-1.csv'
NEGATIVE_OBJECTS_FILE = 'cache/4J-2S-1-negatives.csv'
TEST_SET_FILE = 'cache/4J-2S-1-classifying.csv'

asteroid_numbers = read_file(POSITIVE_OBJECTS_FILE)

train_sizes_lst = []
for i in [50]:
    train_sizes_lst.append({'positive': i, 'negative': asteroid_numbers[i] - i})

TRAIN_SIZES = train_sizes_lst

TEST_SIZES = [50000]

CLASS_WEIGHT_DICT = {0: 0.003, 1: 0.997}

models_lst = {}
for d in [5, 10, 20, 50]:
    for min_split in range(2, 21, 4):
        for max_features in [None, 'sqrt', 'log2']:
            for min_leaf in range(1, 21, 4):
                models_lst[f"DT, d={d}, p=1, w=1, s={min_split}, l={min_leaf}, f={max_features}"] = DecisionTreeClassifier(
                    max_depth=d,
                    min_samples_split=min_split,
                    max_features=max_features,
                    min_samples_leaf=min_leaf,
                    criterion='gini',
                    class_weight=CLASS_WEIGHT_DICT,
                )
                models_lst[f"DT, d={d}, p=2, w=1, s={min_split}, l={min_leaf}, f={max_features}"] = DecisionTreeClassifier(
                    max_depth=d,
                    min_samples_split=min_split,
                    max_features=max_features,
                    min_samples_leaf=min_leaf,
                    criterion='entropy',
                    class_weight=CLASS_WEIGHT_DICT,
                )
                models_lst[f"DT, d={d}, p=3, w=1, s={min_split}, l={min_leaf}, f={max_features}"] = DecisionTreeClassifier(
                    max_depth=d,
                    min_samples_split=min_split,
                    max_features=max_features,
                    min_samples_leaf=min_leaf,
                    criterion='log_loss',
                    class_weight=CLASS_WEIGHT_DICT,
                )
                models_lst[f"DT, d={d}, p=1, w=0, s={min_split}, l={min_leaf}, f={max_features}"] = DecisionTreeClassifier(
                    max_depth=d,
                    min_samples_split=min_split,
                    max_features=max_features,
                    min_samples_leaf=min_leaf,
                    criterion='gini',
                    class_weight=None,
                )
                models_lst[f"DT, d={d}, p=2, w=0, s={min_split}, l={min_leaf}, f={max_features}"] = DecisionTreeClassifier(
                    max_depth=d,
                    min_samples_split=min_split,
                    max_features=max_features,
                    min_samples_leaf=min_leaf,
                    criterion='entropy',
                    class_weight=None,
                )
                models_lst[f"DT, d={d}, p=3, w=0, s={min_split}, l={min_leaf}, f={max_features}"] = DecisionTreeClassifier(
                    max_depth=d,
                    min_samples_split=min_split,
                    max_features=max_features,
                    min_samples_leaf=min_leaf,
                    criterion='log_loss',
                    class_weight=None,
                )

MODELS = models_lst
