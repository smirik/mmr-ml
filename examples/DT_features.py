from sklearn.tree import DecisionTreeClassifier
import itertools
from asteroids import read_file

DEBUG = True

features = ['a', 'e', 'sinI', 'n']
FEATURES_COMBINATIONS = [list(comb) for i in range(1, len(features) + 1) for comb in itertools.combinations(features, i)]

POSITIVE_OBJECTS_FILE = 'cache/4J-2S-1.csv'
NEGATIVE_OBJECTS_FILE = 'cache/4J-2S-1-negatives.csv'
TEST_SET_FILE = 'cache/4J-2S-1-classifying.csv'

asteroid_numbers = read_file(POSITIVE_OBJECTS_FILE)

train_sizes_lst = []
for i in [50]:
    train_sizes_lst.append({'positive': i, 'negative': i})
    train_sizes_lst.append({'positive': i, 'negative': asteroid_numbers[i] - i})

TRAIN_SIZES = train_sizes_lst

TEST_SIZES = [50000]

CLASS_WEIGHT_DICT = {0: 0.003, 1: 0.997}

models_lst = {}
for d in range(1, 20, 4):
    models_lst[f"kNN, d={d}"] = DecisionTreeClassifier(max_depth=d, criterion='gini', class_weight=CLASS_WEIGHT_DICT)
    # models_lst[f"kNN, k={k}, p={p}, w=2"] = KNeighborsClassifier(n_neighbors=k, p=p, weights='distance')

MODELS = models_lst
