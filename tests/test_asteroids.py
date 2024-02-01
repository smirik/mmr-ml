from mmr_ml.asteroids import get_train_set


def test_get_train_set():
    train_size = {'positive': 3, 'negative': 2}
    positives_set = {1, 2, 3, 4, 5}
    negatives_set = {6, 7, 8, 9, 10}

    expected_result = [1, 2, 3, 6, 7]
    assert get_train_set(train_size, positives_set, negatives_set) == expected_result
