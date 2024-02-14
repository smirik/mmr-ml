from mmr_ml.asteroids import get_train_set


def test_get_train_set():
    positives_lst = [1, 2, 3, 4, 5]
    negatives_lst = [6, 7, 8, 9, 10]

    train_size = {'positive': 3, 'negative': 2}
    assert get_train_set(train_size, positives_lst, negatives_lst) == [1, 2, 3, 6, 7]

    train_size = {'positive': 1, 'negative': 3}
    assert get_train_set(train_size, positives_lst, negatives_lst) == [1, 6, 7, 8]

    train_size = {'positive': 10, 'negative': 3}
    assert get_train_set(train_size, positives_lst, negatives_lst) == [1, 2, 3, 4, 5, 6, 7, 8]
