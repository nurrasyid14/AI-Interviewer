import numpy as np

def KFold(X, n_splits=5, max_splits=None):
    fold_size = len(X) // n_splits
    for i in range(n_splits):
        start = i * fold_size
        end = start + fold_size if i != n_splits - 1 else len(X)
        test_indices = list(range(start, end))
        train_indices = list(range(0, start)) + list(range(end, len(X)))
        yield train_indices, test_indices
        if max_splits is not None and i + 1 >= max_splits:
            break
__all__ = ["KFold"]