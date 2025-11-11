# ensemble/svm.py
from sklearn.svm import SVC
import joblib

class SVMModel:
    """Support Vector Machine wrapper supporting probability outputs."""

    def __init__(self, kernel='rbf', C=1.0, gamma='scale', probability=True, random_state=42):
        self.model = SVC(kernel=kernel, C=C, gamma=gamma, probability=probability, random_state=random_state)

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def save(self, path: str):
        joblib.dump(self.model, path)

    def load(self, path: str):
        self.model = joblib.load(path)
