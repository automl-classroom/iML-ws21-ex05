import sys
import os  # noqa
sys.path.insert(0, ".")  # noqa

from utils.dataset import Dataset
from classifiers.mlp import MLP
import numpy as np


def merge(a, b, j):
    """
    Merges elements a and b s.t. two new arrays are created.
    Elements before position j will be from a, whereas elements after position j will be from b.

    Parameters:
        a (np.ndarray): Numpy array with shape (?,)
        b (np.ndarray): Numpy array with shape (?,)
        j (int): Index of the feature.

    Returns:
        plus_j (np.ndarray): Element from a at the position j. Same shape as a or b.
        minus_j (np.ndarray): Element from b at the position j. Same shape as a or b.
    """

    return None, None


def get_shapley_by_estimation(model, X, x, j, M=None, seed=0):
    """
    Calculates the shapley value by estimation.

    Parameters:
        X (np.ndarray): Data from which z is sampled from.
        x: Instance of interest.
        j (int): Feature index. Starts from 0.
        M (int): Number of sampling iterations.
        seed (int): Seed for numpy.

    Returns:
        value (float): Shapley value.
    """

    np.random.seed(seed)

    return None


if __name__ == "__main__":
    dataset = Dataset(
        "wheat_seeds",
        [2, 5, 6],
        [7],
        normalize=True,
        categorical=True)

    (X_train, y_train), (X_test, y_test) = dataset.get_data()

    input_units = X_train.shape[1]
    output_units = len(dataset.get_classes())
    features = dataset.get_input_labels()

    model = MLP(3, input_units*2, input_units, output_units, lr=0.01)
    filename = "weights.pth"
    if not os.path.isfile(filename):
        model.fit(X_train, y_train, num_epochs=50, batch_size=16)
        model.save(filename)
    else:
        model.load(filename)

    a = np.array([0, 1, 2, 3, 4, 5])
    b = np.array([6, 7, 8, 9, 10, 11])
    j = 2

    print("Run `merge` ...")
    plus, minus = merge(a, b, j)

    print(f"--- a: {a}")
    print(f"--- b: {b}")
    print(f"--- j: {j}")
    print(f"--- plus: {plus}")
    print(f"--- minus: {minus}")

    print("\nRun `get_shapley_by_estimation` ...")
    x = np.array([0.5, 0.5, 0.5])

    for j in range(3):
        value = get_shapley_by_estimation(model, X_train, x, j, M=10)
        print(f"--- {features[j]}: {value}")
