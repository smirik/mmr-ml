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
OUTPUT_FILE = 'DT.csv'
features = ['a', 'e', 'sinI', 'n']
FEATURES_COMBINATIONS = [list(comb) for i in range(1, len(features) + 1) for comb in itertools.combinations(features, i)]
TEST_SIZES = [50000]
CLASS_WEIGHT_DICT = {0: 0.005, 1: 0.995}

decision_tree_models = {}
for d in range(1, 101, 10):
    decision_tree_models[f"DT, d={d} E"] = DecisionTreeClassifier(max_depth=d, criterion='entropy')
    decision_tree_models[f"DT, d={d} G"] = DecisionTreeClassifier(max_depth=d, criterion='gini')
    decision_tree_models[f"DT, d={d} L"] = DecisionTreeClassifier(max_depth=d, criterion='log_loss')

MODELS = decision_tree_models
