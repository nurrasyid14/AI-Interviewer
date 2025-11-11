import numpy as np
from collections import defaultdict

def StratKFoldSplit(X, y, n_splits=5, max_splits=None):
    label_indices = defaultdict(list)
    for idx, label in enumerate(y):
        label_indices[label].append(idx)

    folds = [[] for _ in range(n_splits)]
    for label, indices in label_indices.items():
        np.random.shuffle(indices)
        fold_size = len(indices) // n_splits
        for i in range(n_splits):
            start = i * fold_size
            end = start + fold_size if i != n_splits - 1 else len(indices)
            folds[i].extend(indices[start:end])

    for i in range(n_splits):
        test_indices = folds[i]
        train_indices = [idx for j in range(n_splits) if j != i for idx in folds[j]]
        yield train_indices, test_indices
        if max_splits is not None and i + 1 >= max_splits:
            break
__all__ = ["StratKFoldSplit"]