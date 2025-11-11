import numpy as np

def Bootstrap(X, n_splits=5, max_splits=None):
    n_samples = len(X)
    for i in range(n_splits):
        train_indices = np.random.choice(n_samples, size=n_samples, replace=True).tolist()
        test_indices = list(set(range(n_samples)) - set(train_indices))
        yield train_indices, test_indices
        if max_splits is not None and i + 1 >= max_splits:
            break
__all__ = ["Bootstrap"]