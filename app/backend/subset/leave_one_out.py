def LeaveOneOut(X, max_splits=None):
    n_samples = len(X)
    for i in range(n_samples):
        test_indices = [i]
        train_indices = list(range(0, i)) + list(range(i + 1, n_samples))
        yield train_indices, test_indices
        if max_splits is not None and i + 1 >= max_splits:
            break
__all__ = ["LeaveOneOut"]