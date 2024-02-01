import itertools
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import BalancedRandomForestClassifier


DEBUG = True
TRAIN_SIZES = [17000]
MAX_ASTEROID_NUMBER = 640000
ASTEROID_NUMBERS_FILE = 'cache/4J-2S-1.csv'
OUTPUT_FILE = 'astrobook_features.csv'
features = ['a', 'e', 'sinI', 'n']
TEST_SIZES = [50000]
CLASS_WEIGHT_DICT = {0: 0.005, 1: 0.995}

FEATURES_COMBINATIONS = [list(comb) for i in range(1, len(features) + 1) for comb in itertools.combinations(features, i)]

# Models
MODELS = {
    'kNN': KNeighborsClassifier(n_neighbors=16, p=1, weights='distance'),
    'GBoost': GradientBoostingClassifier(n_estimators=10),
    'NB': GaussianNB(),
    'RF': RandomForestClassifier(n_estimators=10, class_weight="balanced"),
}
