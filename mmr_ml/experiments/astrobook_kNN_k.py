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
TRAIN_SIZES = [22150]
MAX_ASTEROID_NUMBER = 640000
POSITIVE_OBJECTS_FILE = 'cache/4J-2S-1.csv'
OUTPUT_FILE = 'astrobook_kNN_k.csv'
features = ['mag', 'a', 'e', 'sinI', 'n']
FEATURES_COMBINATIONS = [['a', 'n']]
TEST_SIZES = [50000]
CLASS_WEIGHT_DICT = {0: 0.005, 1: 0.995}

knn_models = {}
for k in range(1, 100, 1):
    knn_models[f"k={k}, minkoswki"] = KNeighborsClassifier(n_neighbors=k, p=1, weights='distance')
    knn_models[f"k={k}, euclidian"] = KNeighborsClassifier(n_neighbors=k, p=2, weights='distance')
    knn_models[f"k={k}, cosine"] = KNeighborsClassifier(n_neighbors=k, metric='cosine', weights='distance')

MODELS = {
    # 'kNN, k=3': KNeighborsClassifier(n_neighbors=3),
    # 'kNN, k=5': KNeighborsClassifier(n_neighbors=5),
    # 'kNN, k=5, D': KNeighborsClassifier(n_neighbors=5, weights='distance'),
    # 'kNN, k=5, p=3': KNeighborsClassifier(n_neighbors=5, p=3),
    # 'kNN, k=5, p=1': KNeighborsClassifier(n_neighbors=5, p=1),
    # 'kNN, k=10': KNeighborsClassifier(n_neighbors=10),
    # 'kNN, k=3, p=1, D': KNeighborsClassifier(n_neighbors=3, p=1, weights='distance'),
    # 'DTree, d=None': DecisionTreeClassifier(max_depth=None),
    # 'DTree, d=10': DecisionTreeClassifier(max_depth=10),
    # 'DTree, E, d=10': DecisionTreeClassifier(max_depth=10, criterion='entropy'),
    # 'DTree, d=20': DecisionTreeClassifier(max_depth=20),
    # 'DTree, d=100': DecisionTreeClassifier(max_depth=100),
    # 'LogReg, i=100': LogisticRegression(C=1, max_iter=100, class_weight='balanced', solver='liblinear'),
    # 'LogReg, i=1000': LogisticRegression(C=1, max_iter=1000, class_weight='balanced', solver='liblinear'),
    # 'LogReg, i=1000c': LogisticRegression(C=1, max_iter=1000, class_weight='balanced', solver='newton-cholesky'),
    # 'GBoost, n=10': GradientBoostingClassifier(n_estimators=10),
    # 'GBoost, n=100': GradientBoostingClassifier(n_estimators=100),
    # 'GBoost, n=200': GradientBoostingClassifier(n_estimators=200),
    # 'RandomForest, n=5': RandomForestClassifier(n_estimators=5, class_weight="balanced"),
    # 'RandomForest, n=10': RandomForestClassifier(n_estimators=10, class_weight="balanced"),
    # 'RandomForest, n=100': RandomForestClassifier(n_estimators=100, class_weight="balanced"),
    # 'RandomForest, n=200': RandomForestClassifier(n_estimators=200, class_weight="balanced"),
    # 'SVM, rbf': SVC(kernel='sigmoid', class_weight=CLASS_WEIGHT_DICT),
    # 'MLP, n=10': MLPClassifier(max_iter=10),
    # 'MLP, n=100': MLPClassifier(max_iter=100),
    # 'MLP, n=1000': MLPClassifier(max_iter=1000),
    # 'NaiveBayes': GaussianNB(),
    # 'BernouliNB': BernoulliNB(),
    # 'Ada, n=10': AdaBoostClassifier(n_estimators=10, algorithm='SAMME'),
    # 'Ada, n=100': AdaBoostClassifier(n_estimators=100, algorithm='SAMME'),
    # 'Ada, n=200': AdaBoostClassifier(n_estimators=200, algorithm='SAMME'),
    # 'B. Random Forest, n=10': BalancedRandomForestClassifier(n_estimators=10),
    # 'B. Random Forest, n=100': BalancedRandomForestClassifier(n_estimators=100),
    # 'B. Random Forest, n=200': BalancedRandomForestClassifier(n_estimators=200),
}

MODELS = knn_models
