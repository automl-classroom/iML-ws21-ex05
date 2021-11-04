import sys
import os  # noqa
sys.path.insert(0, ".")  # noqa

from utils.shapley import get_value
from datasets.shapley import COMB1, COMB2, COMB3, COMB7, COMB8, COMB9
import itertools
import numpy as np
from math import factorial


def get_shapley(P, j, v):
    """
    Returns the shapley value based on the original implementation.

    Parameters:
        P (list): All players.
        j (int): Selected player. j is in the list of P.
        v (func): Function to compute the value given a set/list.
            Elements from set/list must be in P.

    Returns:
        value (float): Rounded Shapley value to two decimal places.
        S_all (list of sets): All possible combinations.
    """

    return None, None


def get_shapley_by_order(P, j, v, M=None):
    """
    Returns the shapley value based on order permutations.

    Parameters:
        P (list): All players.
        j (int): Selected player. j is in the list of P.
        v (func): Function to compute the value given a set/list.
            Elements from set/list must be in P.
        M (int): Number of used permutations. Optional.

    Returns:
        value (float): Rounded Shapley value to two decimal places.
        S_all (list of sets): One set for each permutation order. Set should only contain the players
            before j is added.
    """

    return None, None


def check_symmetry(P, j, k, v):
    """
    Checks if players j and k contributes the same.

    Parameters:
        P (list): All players.
        j (int): Selected player. j is in the list of P.
        k (int): Selected player. k is in the list of P.
        v (func): Function to compute the value given a set/list.
            Elements from set/list must be in P.

    Returns:
        bool (bool): True if player j and k have the same payout.
    """

    return None


def check_dummy(S, j, v):
    """
    Checks if player j in S has a contribution.

    Parameters:
        S (list): All players.
        j (int): Selected player. j is in the list of P.
        v (func): Function to compute the value given a set/list.
            Elements from set/list must be in P.

    Returns:
        bool (bool): True if player j has no contribution.
    """

    return None


def check_additivity(P, j, v, v1, v2):
    """
    Checks if game v can be derived from v1 and v2.

    Parameters:
        S (list): All players.
        j (int): Selected player. j is in the list of P.
        v (func): Function to compute the value given a set/list.
            Elements from set/list must be in P.

    Returns:
        bool (bool): True if marginal contributions are additive.
    """

    return None


def check_efficiency(P, v):
    """
    Checks if player contributions add up to the total payout of the game.

    Parameters:
        P (list): All players.
        v (func): Function to compute the value given a set/list.
            Elements from set/list must be in P.

    Returns:
        bool (bool): True if contributions add up to the total payout.
    """

    return None


if __name__ == "__main__":

    S = [1, 3]
    P = [1, 2, 3]
    def v(S): return get_value(S, COMB1)

    print("Given are the following sets:")
    print(COMB1)

    print("Shapley by original implementation:")
    for j in P:
        shapley = get_shapley(P, j, v)
        print(f"--- P{j}: {shapley[0]}")

    print("Shapley by orders")
    for j in P:
        shapley = get_shapley_by_order(P, j, v)
        print(f"--- P{j}: {shapley[0]}")

    print("Check symmetry ...")
    def v(S): return get_value(S, COMB2)
    print("---", check_symmetry(P, 2, 3, v))

    print("Check dummy ...")
    def v(S): return get_value(S, COMB3)
    print("---", check_dummy(S, 2, v))

    print("Check additivity ...")
    def v(S): return get_value(S, COMB7)
    def v1(S): return get_value(S, COMB8)
    def v2(S): return get_value(S, COMB9)
    print("---", check_additivity(S, 2, v, v1, v2))

    print("Check efficiency ...")

    def v(S): return get_value(S, COMB1)
    print("---", check_efficiency(P, v))
