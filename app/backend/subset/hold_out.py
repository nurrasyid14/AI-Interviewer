import numpy as np

def HoldOut(X, test_size=0.2):
    n_samples = len(X)
    test_size = int(n_samples * test_size)
    indices = np.random.permutation(n_samples)
    test_indices = indices[:test_size].tolist()
    train_indices = indices[test_size:].tolist()
    return train_indices, test_indices
__all__ = ["HoldOut"]