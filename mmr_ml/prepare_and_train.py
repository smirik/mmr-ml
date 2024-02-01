from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


def prepare_data_and_labels(data_dict, asteroids, features):
    prepared_data = []
    filtered_asteroids = []
    for asteroid in asteroids:
        asteroid_key = str(asteroid)
        if asteroid_key in data_dict:
            asteroid_data = data_dict[asteroid_key]
            feature_values = [asteroid_data.get(feature) for feature in features if feature in asteroid_data]
            if len(feature_values) == len(features):
                prepared_data.append(feature_values)
                filtered_asteroids.append(asteroid)
    return prepared_data, filtered_asteroids


def train_model(model, train_data, train_labels):
    model.fit(train_data, train_labels)
    return model


# Evaluate the model
def evaluate_model(model, test_data, test_labels):
    predictions = model.predict(test_data)
    tn, fp, fn, tp = confusion_matrix(test_labels, predictions, labels=[0, 1]).ravel()
    return {
        'true_positive': tp,
        'true_negative': tn,
        'false_positive': fp,
        'false_negative': fn,
        'accuracy': round(accuracy_score(test_labels, predictions), 3),
        'precision': round(precision_score(test_labels, predictions, zero_division=0), 3),
        'recall': round(recall_score(test_labels, predictions, zero_division=0), 3),
        'f1_score': round(f1_score(test_labels, predictions, zero_division=0), 3),
    }
