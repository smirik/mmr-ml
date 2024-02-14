from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
import numpy as np


def get_classifiers():
    return CLASSIFIERS


CLASSIFIERS = {
    'DecisionTree': (
        DecisionTreeClassifier(random_state=42),
        {
            'criterion': ['gini'],
            'max_depth': range(1, 101, 1),
            'min_samples_split': range(2, 31, 1),
            # 'min_samples_leaf': [1, 2, 5],
        },
    ),
    # 'DecisionTree': (
    #     DecisionTreeClassifier(random_state=42),
    #     {
    #         'criterion': ['gini', 'entropy'],
    #         'max_depth': [None, 5, 10, 15, 20],
    #         'min_samples_split': [2, 5, 10],
    #         # 'min_samples_leaf': [1, 2, 5],
    #     },
    # ),
    'kNN': (
        KNeighborsClassifier(),
        {
            'n_neighbors': range(1, 101, 1),
            'weights': ['uniform', 'distance'],
            # 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'p': [1, 2],
        },
    ),
    'LogisticRegression': (
        LogisticRegression(max_iter=1000, random_state=42),
        {
            'C': np.logspace(-4, 4, 5),
            'solver': ['liblinear', 'lbfgs'],
        },
    ),
    'RandomForest': (
        RandomForestClassifier(random_state=42, n_jobs=-1),
        {
            'n_estimators': range(1, 101, 1),
            'max_depth': [None, 3, 7],
            'min_samples_split': [3],
            'max_features': ['sqrt'],
            'class_weight': ['balanced', None],
            # 'min_samples_leaf': [1, 2, 5],
        },
    ),
    'GradientBoosting': (
        # GradientBoostingClassifier(random_state=42),
        # {
        #     'n_estimators': [55, 95],
        #     'learning_rate': [0.1],
        #     'max_depth': [None, 6],
        #     'min_samples_split': range(2, 101, 1),
        #     'max_features': ['sqrt'],
        #     'loss': ['exponential'],
        # },
        GradientBoostingClassifier(random_state=42),
        {
            'n_estimators': range(1, 101, 1),
            'learning_rate': [0.10],
            'max_depth': [5, None],
            'min_samples_split': [3],
            'max_features': ['sqrt'],
            'loss': ['exponential'],
        },
        #         0,1,"a, e",1.000,0.956,0.970,0.962,0.1,exponential,5,sqrt,9,95
        # 1,2,"a, sinI",1.000,0.962,0.962,0.962,0.1,exponential,5,sqrt,3,55
    ),
    'AdaBoost': (
        AdaBoostClassifier(random_state=42),
        {
            'algorithm': ['SAMME'],
            'n_estimators': range(0, 2000, 20),
            'learning_rate': [0.2],
        },
    ),
    'BalancedRandomForest': (
        BalancedRandomForestClassifier(random_state=42),
        {
            'n_estimators': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            'max_depth': [None, 3],
            'min_samples_split': [2, 3, 4, 5],
            'max_features': ['sqrt'],
            'class_weight': ['balanced'],
        },
    ),
    'NaiveBayes': (
        GaussianNB(),
        {},
    ),
}
