import sys
import os  # noqa
sys.path.insert(0, ".")  # noqa

from utils.shapley import get_value
from datasets.shapley import COMB1, COMB2, COMB3, COMB4, COMB5, COMB6, COMB7, COMB8, COMB9, COMB10, COMB11, COMB12, COMB13, COMB14, COMB15, COMB16
from tests.config import WORKING_DIR

module = __import__(f"{WORKING_DIR}.cooperative_game", fromlist=[
    'get_shapley', 'get_shapley_by_order', 'check_symmetry', 'check_dummy', 'check_additivity', 'check_efficiency'])


def test_get_shapley():
    P = [1, 2, 3, 4]
    def v(S): return get_value(S, COMB10)
    value, S_all = module.get_shapley(P, P[1], v)

    assert value == 20.75
    assert set(()) in S_all
    assert set((1, 4)) in S_all


def test_get_shapley_by_order():
    P = [1, 2, 3, 4]
    def v(S): return get_value(S, COMB10)

    value, S_all = module.get_shapley_by_order(P, P[3], v)

    assert value == 31.25
    assert S_all[-9] == set((2, 3))
    assert S_all[1] == set((1, 2))


def test_get_shapley_by_order_approx():
    P = [1, 2, 3, 4]
    def v(S): return get_value(S, COMB10)

    value, S_all = module.get_shapley_by_order(P, P[3], v, M=24)

    assert value == 31.25
    assert S_all[-9] == set((2, 3))
    assert S_all[1] == set((1, 2))

    value, S_all = module.get_shapley_by_order(P, P[3], v, M=10)

    assert value == 44.7
    assert S_all[0] == set((1, 2, 3))
    assert S_all[-1] == set((2, 3))


def test_check_symmetry():
    P = [1, 2, 3]

    def v(S): return get_value(S, COMB2)
    assert not module.check_symmetry(P, 2, 3, v)

    def v(S): return get_value(S, COMB1)
    assert not module.check_symmetry(P, 2, 3, v)

    def v(S): return get_value(S, COMB8)
    assert not module.check_symmetry(P, 2, 3, v)

    def v(S): return get_value(S, COMB16)
    assert module.check_symmetry(P, 2, 3, v)


def test_check_dummy():
    S = [1, 3]

    def v(S): return get_value(S, COMB3)
    assert module.check_dummy(S, 2, v)

    def v(S): return get_value(S, COMB4)
    assert not module.check_dummy(S, 2, v)


def test_check_additivity():
    S = [1, 3]

    def v(S): return get_value(S, COMB4)
    def v1(S): return get_value(S, COMB5)
    def v2(S): return get_value(S, COMB6)
    assert not module.check_additivity(S, 2, v, v1, v2)

    def v(S): return get_value(S, COMB7)
    def v1(S): return get_value(S, COMB8)
    def v2(S): return get_value(S, COMB9)
    assert not module.check_additivity(S, 2, v, v1, v2)

    def v(S): return get_value(S, COMB13)
    def v1(S): return get_value(S, COMB14)
    def v2(S): return get_value(S, COMB15)
    assert module.check_additivity(S, 2, v, v1, v2)


def test_check_efficiency():
    P = [1, 2, 3]

    def v(S): return get_value(S, COMB1)
    assert module.check_efficiency(P, v)

    def v(S): return get_value(S, COMB11)
    assert not module.check_efficiency(P, v)

    def v(S): return get_value(S, COMB10)
    assert module.check_efficiency(P, v)

    def v(S): return get_value(S, COMB12)
    assert not module.check_efficiency(P, v)


if __name__ == "__main__":
    # test_get_shapley()
    # test_get_shapley_by_order()
    # test_get_shapley_by_order_approx()
    test_check_symmetry()
    # test_check_dummy()
    # test_check_additivity()
    # test_check_efficiency()
