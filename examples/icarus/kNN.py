from sklearn.neighbors import KNeighborsClassifier
from asteroids import read_file

DEBUG = True

FEATURES_COMBINATIONS = [['a', 'n']]

POSITIVE_OBJECTS_FILE = 'cache/4J-2S-1.csv'
NEGATIVE_OBJECTS_FILE = 'cache/4J-2S-1-negatives.csv'
TEST_SET_FILE = 'cache/4J-2S-1-classifying.csv'

asteroid_numbers = read_file(POSITIVE_OBJECTS_FILE)

train_sizes_lst = []
for i in [100]:
    train_sizes_lst.append({'positive': i, 'negative': asteroid_numbers[i] - i})

TRAIN_SIZES = train_sizes_lst

TEST_SIZES = [100000]

models_lst = {}
for k in [6]:
    models_lst[f"kNN, k={k}"] = KNeighborsClassifier(n_neighbors=k, p=1, weights='distance', n_jobs=-1)

MODELS = models_lst
