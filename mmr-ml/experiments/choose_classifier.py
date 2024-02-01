from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from imblearn.ensemble import BalancedRandomForestClassifier


DEBUG = True
NUM_TRAIN_ASTEROIDS = 17000
MAX_ASTEROID_NUMBER = 640000
POSITIVE_OBJECTS_FILE = 'cache/4J-2S-1.csv'
OUTPUT_FILE = 'choose_classifier.csv'
FEATURES_COMBINATIONS = [
    # ['mag', 'a', 'e', 'sinI', 'n'],
    ['a', 'e', 'sinI', 'n'],
    # ['e', 'sinI', 'n'],
    # ['e', 'n'],
    # ['sinI', 'n'],
    # ['e', 'sinI'],
]
TEST_SIZES = [620515]
CLASS_WEIGHT_DICT = {0: 0.005, 1: 0.995}

# Models
MODELS = {
    'kNN, k=3': KNeighborsClassifier(n_neighbors=3),
    'kNN, k=5': KNeighborsClassifier(n_neighbors=5),
    'kNN, k=10': KNeighborsClassifier(n_neighbors=10),
    'DTree, d=None': DecisionTreeClassifier(max_depth=None),
    'DTree, d=10': DecisionTreeClassifier(max_depth=10),
    'DTree, d=20': DecisionTreeClassifier(max_depth=20),
    'DTree, d=100': DecisionTreeClassifier(max_depth=100),
    'LogR, i=10': LogisticRegression(C=1, max_iter=10, class_weight='balanced', solver='liblinear'),
    'LogR, i=100': LogisticRegression(C=1, max_iter=100, class_weight='balanced', solver='liblinear'),
    'LogR, i=1000': LogisticRegression(C=1, max_iter=1000, class_weight='balanced', solver='liblinear'),
    'GBoost, n=10': GradientBoostingClassifier(n_estimators=10),
    'GBoost, n=100': GradientBoostingClassifier(n_estimators=100),
    'GBoost, n=200': GradientBoostingClassifier(n_estimators=200),
    'RF, n=10': RandomForestClassifier(n_estimators=10, class_weight="balanced"),
    'RF, n=100': RandomForestClassifier(n_estimators=100, class_weight="balanced"),
    'RF, n=200': RandomForestClassifier(n_estimators=200, class_weight="balanced"),
    'SVM, rbf': SVC(kernel='sigmoid', class_weight=CLASS_WEIGHT_DICT),
    'MLP, i=10': MLPClassifier(max_iter=10),
    'MLP, i=100': MLPClassifier(max_iter=100),
    'MLP, i=1000': MLPClassifier(max_iter=1000),
    'NB': GaussianNB(),
    'Ada, n=10': AdaBoostClassifier(n_estimators=10, algorithm='SAMME'),
    'Ada, n=100': AdaBoostClassifier(n_estimators=100, algorithm='SAMME'),
    'Ada, n=200': AdaBoostClassifier(n_estimators=200, algorithm='SAMME'),
    'BRF, n=10': BalancedRandomForestClassifier(n_estimators=10),
    'BRF, n=100': BalancedRandomForestClassifier(n_estimators=100),
    'BRF, n=200': BalancedRandomForestClassifier(n_estimators=200),
}
