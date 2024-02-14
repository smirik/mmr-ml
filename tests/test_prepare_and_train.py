from mmr_ml.prepare_and_train import prepare_data_and_labels


def test_prepare_data_and_labels():
    data_dict = {
        '1': {'feature1': 1, 'feature2': 2},
        '2': {'feature1': 3, 'feature2': 4},
        '3': {'feature1': 5, 'feature2': 6},
        '4': {'feature1': 7, 'feature2': 8},
        '5': {'feature1': 9, 'feature2': 10},
    }
    asteroids = [1, 2, 3, 4, 5]

    features = ['feature1', 'feature2']
    prepared_data, filtered_asteroids = prepare_data_and_labels(data_dict, asteroids, features)
    assert prepared_data == [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]
    assert filtered_asteroids == [1, 2, 3, 4, 5]

    asteroids = [1, 2, 3]
    features = ['feature1']
    prepared_data, filtered_asteroids = prepare_data_and_labels(data_dict, asteroids, features)
    assert prepared_data == [[1], [3], [5]]
    assert filtered_asteroids == [1, 2, 3]
