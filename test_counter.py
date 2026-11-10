from counter import VehicleCounter

def test_counting():
    c = VehicleCounter(line=((0, 100), (200, 100)))
    # легковая едет вниз, грузовик вверх, автобус не доезжает до линии
    for y in range(50, 160, 10):
        c.update([(1, 2, 100, y, 0.9), (2, 7, 50, 200 - y, 0.8), (3, 5, 150, y / 2, 0.9)])
    assert c.total("car") == 1 and c.total("truck") == 1 and c.total("bus") == 0
    assert c.counts["car"][0] + c.counts["truck"][0] == 1  # разные направления
    # повторное пересечение тем же ID не считается
    for y in range(160, 40, -10):
        c.update([(1, 2, 100, y, 0.9)])
    assert c.total("car") == 1

def test_class_vote():
    c = VehicleCounter(line=((0, 100), (200, 100)))
    seq = [7, 7, 2, 7, 7, 7, 7]  # одно «мигание» в car
    for i, y in enumerate(range(50, 120, 10)):
        c.update([(5, seq[i], 100, y, 0.8)])
    assert c.total("truck") == 1 and c.total("car") == 0

def test_outside_segment():
    c = VehicleCounter(line=((0, 100), (200, 100)))
    for y in range(50, 160, 10):
        c.update([(9, 2, 500, y, 0.9)])  # проезжает мимо отрезка
    assert c.total() == 0

if __name__ == "__main__":
    test_counting(); test_class_vote(); test_outside_segment(); print("OK")
