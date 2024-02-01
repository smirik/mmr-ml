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
OUTPUT_FILE = 'choose_classifier_astrobook.csv'
features = ['mag', 'a', 'e', 'sinI', 'n']
TEST_SIZES = [50000]
CLASS_WEIGHT_DICT = {0: 0.005, 1: 0.995}

FEATURES_COMBINATIONS = [['a', 'n']]
FEATURES_COMBINATIONS = [
    ['a', 'e', 'sinI', 'n'],
    # ['e', 'n'],
    ['n'],
]

# Models
MODELS = {
    'kNN': KNeighborsClassifier(n_neighbors=16, p=1, weights='distance'),
    'Decision Tree': DecisionTreeClassifier(max_depth=None),
    'Logistic Regression': LogisticRegression(C=1, max_iter=100, class_weight='balanced', solver='liblinear'),
    'GBoost': GradientBoostingClassifier(n_estimators=10),
    'RandomForest, n=10': RandomForestClassifier(n_estimators=10, class_weight="balanced"),
    'NaiveBayes': GaussianNB(),
}
