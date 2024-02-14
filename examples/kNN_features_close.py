from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
import itertools
from mmr_ml.asteroids import read_file

DEBUG = True

features = ['a', 'e', 'sinI', 'n']
FEATURES_COMBINATIONS = [list(comb) for i in range(1, len(features) + 1) for comb in itertools.combinations(features, i)]

POSITIVE_OBJECTS_FILE = 'cache/4J-2S-1.csv'
NEGATIVE_OBJECTS_FILE = 'cache/4J-2S-1-close-negatives.csv'
TEST_SET_FILE = 'cache/4J-2S-1-classifying.csv'

asteroid_numbers = read_file(POSITIVE_OBJECTS_FILE)

train_sizes_lst = []
for i in [50]:
    train_sizes_lst.append({'positive': i, 'negative': 450})

TRAIN_SIZES = train_sizes_lst

TEST_SIZES = [10000]
tmp = read_file('cache/4J-2S-1_close_asteroids.csv')
TEST_LST = tmp[496:]

CLASS_WEIGHT_DICT = {0: 0.003, 1: 0.997}

models_lst = {}
for k in [3, 5, 11]:
    for p in range(1, 3):
        models_lst[f"kNN, k={k}, p={p}, w=1"] = KNeighborsClassifier(n_neighbors=k, p=p)
        models_lst[f"kNN, k={k}, p={p}, w=2"] = KNeighborsClassifier(n_neighbors=k, p=p, weights='distance')

MODELS = models_lst
