import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
import os

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from itertools import combinations

from asteroids import read_file, get_train_set, get_asteroids_data
from prepare_and_train import prepare_data_and_labels
from grid.classifiers import get_classifiers
from grid.prettyprint import pretty_print

from grid.config import *


os.environ['PYTHONWARNINGS'] = 'ignore'

train_asteroids = get_train_set(train_size, positives_lst, negatives_lst)
train_data_dict = get_asteroids_data(train_asteroids)
prepared_train_data, filtered_train_asteroids = prepare_data_and_labels(train_data_dict, train_asteroids, features)
train_labels = [1 if num in positives_lst else 0 for num in filtered_train_asteroids]

X_train, X_val, y_train, y_val = train_test_split(prepared_train_data, train_labels, test_size=TEST_SIZE, random_state=42)

classifiers = get_classifiers()
selected_classifiers = ['BalancedRandomForest']

all_results = []
# FEATURES_COMBINATIONS = [['a', 'e', 'n'], ['a', 'e', 'n', 'sinI'], ['a', 'e'], ['a', 'n', 'sinI'], ['a', 'sinI']]
# FEATURES_COMBINATIONS = [['a', 'e', 'n', 'sinI']]
for classifier_name in selected_classifiers:
    classifier, parameters = classifiers[classifier_name]
    for feature in FEATURES_COMBINATIONS:
        # for L in range(1, len(features) + 1):
        # print(f"Trying combinations of {L} features")
        # for subset in combinations(features, L):
        print(f"Trying subset: {feature}")
        subset_indices = [feature_indices[feature] for feature in feature]
        X_train_subset = np.array(X_train)[:, subset_indices]

        pipeline = Pipeline([('scaler', StandardScaler()), ('classifier', classifier)])
        updated_parameters = {f'classifier__{key}': value for key, value in parameters.items()}

        grid_search = GridSearchCV(
            pipeline,
            updated_parameters,
            cv=5,
            scoring={'f1': 'f1', 'recall': 'recall', 'precision': 'precision', 'accuracy': 'accuracy'},
            refit='f1',
            verbose=1,
            n_jobs=-1,
        )
        grid_search.fit(X_train_subset, y_train)

        for params, accuracy, precision, recall_score, f1_score in zip(
            grid_search.cv_results_['params'],
            grid_search.cv_results_['mean_test_accuracy'],
            grid_search.cv_results_['mean_test_precision'],
            grid_search.cv_results_['mean_test_recall'],
            grid_search.cv_results_['mean_test_f1'],
        ):
            all_results.append((accuracy, precision, recall_score, f1_score, params, feature))

all_results.sort(key=lambda x: x[3], reverse=True)
rows, table = pretty_print(all_results, N=20)
print(table)

df = pd.DataFrame(rows, columns=table.field_names)
df.to_csv('BRF_grid_sm.csv')
