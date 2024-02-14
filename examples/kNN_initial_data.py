from sklearn.neighbors import KNeighborsClassifier
from asteroids import read_file

DEBUG = True

FEATURES_COMBINATIONS = [['n'], ['a', 'n']]

POSITIVE_OBJECTS_FILE = 'cache/4J-2S-1.csv'
NEGATIVE_OBJECTS_FILE = 'cache/4J-2S-1-negatives.csv'
TEST_SET_FILE = 'cache/4J-2S-1-classifying.csv'

asteroid_numbers = read_file(POSITIVE_OBJECTS_FILE)

train_sizes_lst = []
for i in range(1, 101, 1):
    train_sizes_lst.append({'positive': i, 'negative': asteroid_numbers[i] - i})

TRAIN_SIZES = train_sizes_lst

TEST_SIZES = [50000]

CLASS_WEIGHT_DICT = {0: 0.003, 1: 0.997}

models_lst = {}
for k in [3, 16]:
    for p in [1, 2]:
        models_lst[f"kNN, k={k}, p={p}"] = KNeighborsClassifier(n_neighbors=k, p=p, weights='distance')

MODELS = models_lst
