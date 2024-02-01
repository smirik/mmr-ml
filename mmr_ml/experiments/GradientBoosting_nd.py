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
POSITIVE_OBJECTS_FILE = 'cache/4J-2S-1.csv'
OUTPUT_FILE = 'GB_ND2.csv'
features = ['a', 'e', 'sinI', 'n']
# FEATURES_COMBINATIONS = [list(comb) for i in range(1, len(features) + 1) for comb in itertools.combinations(features, i)]
FEATURES_COMBINATIONS = [['e', 'n']]
TEST_SIZES = [50000]
CLASS_WEIGHT_DICT = {0: 0.005, 1: 0.995}

gradient_boosting_models = {}
for n in range(1, 101, 1):
    # for d in range(1, 21, 1):
    for d in [1, 2, 3, 4, 5, 6]:
        gradient_boosting_models[f"GBoost, n={n}, d={d}"] = GradientBoostingClassifier(n_estimators=n, max_depth=d)

MODELS = gradient_boosting_models