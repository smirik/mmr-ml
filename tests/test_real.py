from mmr_ml.asteroids import get_asteroids_data


def test_get_asteroids_data():
    lst = [
        400000,
        400001,
        400002,
    ]

    data = get_asteroids_data(lst)
    print(data)
    assert len(data) == len(lst)
