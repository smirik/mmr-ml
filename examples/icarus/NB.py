from sklearn.naive_bayes import GaussianNB
from asteroids import read_file

DEBUG = True

FEATURES_COMBINATIONS = [['n']]

POSITIVE_OBJECTS_FILE = 'cache/4J-2S-1.csv'
NEGATIVE_OBJECTS_FILE = 'cache/4J-2S-1-negatives.csv'
TEST_SET_FILE = 'cache/4J-2S-1-classifying.csv'

asteroid_numbers = read_file(POSITIVE_OBJECTS_FILE)

train_sizes_lst = []
for i in [3, 100]:
    train_sizes_lst.append({'positive': i, 'negative': asteroid_numbers[i] - i})

TRAIN_SIZES = train_sizes_lst

TEST_SIZES = [100000]

models_lst = {}
models_lst[f"NB"] = GaussianNB()
MODELS = models_lst
