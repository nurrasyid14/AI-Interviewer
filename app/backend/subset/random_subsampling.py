import numpy as np

def RandomSubsampling(X, n_splits=5, test_size=0.2, max_splits=None):
    n_samples = len(X)
    test_size = int(n_samples * test_size)
    for i in range(n_splits):
        indices = np.random.permutation(n_samples)
        test_indices = indices[:test_size].tolist()
        train_indices = indices[test_size:].tolist()
        yield train_indices, test_indices
        if max_splits is not None and i + 1 >= max_splits:
            break
__all__ = ["RandomSubsampling"]