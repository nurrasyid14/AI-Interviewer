from sklearn.model_selection import LeavePOut as SKLeavePOut

def LeavePOut(X, p=2, max_splits=None):
    lpo = SKLeavePOut(p)
    for i, (train_idx, test_idx) in enumerate(lpo.split(X)):
        yield train_idx.tolist(), test_idx.tolist()
        if max_splits is not None and i + 1 >= max_splits:
            break
__all__ = ["LeavePOut"]