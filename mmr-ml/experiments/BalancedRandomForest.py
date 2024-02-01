from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
import itertools


DEBUG = True
TRAIN_SIZES = [17000, 41000]
MAX_ASTEROID_NUMBER = 640000
ASTEROID_NUMBERS_FILE = 'cache/4J-2S-1.csv'
OUTPUT_FILE = 'BRF.csv'
features = ['a', 'e', 'sinI', 'n']
FEATURES_COMBINATIONS = [list(comb) for i in range(1, len(features) + 1) for comb in itertools.combinations(features, i)]
TEST_SIZES = [50000]
CLASS_WEIGHT_DICT = {0: 0.005, 1: 0.995}

balanced_random_forest_models = {}
for n in range(1, 101, 10):
    for d in range(1, 20, 5):
        balanced_random_forest_models[f"BRF, n={n}, d={d}"] = BalancedRandomForestClassifier(n_estimators=n, max_depth=d)

MODELS = balanced_random_forest_models
