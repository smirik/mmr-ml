import random
from prepare_and_train import prepare_data_and_labels, train_model, evaluate_model
from asteroids import get_asteroids_data


def run_simulation(
    test_sizes: list[int], train_sizes: list[int], asteroid_numbers_set, models, features_combinations: list[str], debug=False
):
    results = []
    for test_size in test_sizes:
        for train_size in train_sizes:
            if debug:
                print(f"Going for test_size = {test_size}, train_size = {train_size}")
            train_asteroids = range(1, train_size + 1)
            train_data_dict = get_asteroids_data(train_asteroids)

            for model_name, model in models.items():
                for features in features_combinations:
                    if debug:
                        print(f"Going for model = {model_name}, features = {features}")
                    prepared_train_data, filtered_train_asteroids = prepare_data_and_labels(train_data_dict, train_asteroids, features)
                    train_labels = [1 if num in asteroid_numbers_set else 0 for num in filtered_train_asteroids]

                    trained_model = train_model(model, prepared_train_data, train_labels)

                    test_asteroids = range(train_size + 1, train_size + test_size + 1)
                    test_data_dict = get_asteroids_data(test_asteroids)
                    prepared_test_data, filtered_test_asteroids = prepare_data_and_labels(test_data_dict, test_asteroids, features)
                    test_labels = [1 if num in asteroid_numbers_set else 0 for num in filtered_test_asteroids]
                    metrics = evaluate_model(trained_model, prepared_test_data, test_labels)

                    results.append(
                        [
                            len(filtered_train_asteroids),
                            sum(train_labels),
                            len(filtered_train_asteroids) - sum(train_labels),
                            len(filtered_test_asteroids),
                            sum(test_labels),
                            len(filtered_test_asteroids) - sum(test_labels),
                            model_name,
                            ', '.join(features),
                            metrics['true_positive'],
                            metrics['false_positive'],
                            metrics['true_negative'],
                            metrics['false_negative'],
                            metrics['accuracy'],
                            metrics['precision'],
                            metrics['recall'],
                            metrics['f1_score'],
                        ]
                    )
    return results
