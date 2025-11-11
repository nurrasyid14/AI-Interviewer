# ensemble/neural_network.py
from sklearn.neural_network import MLPClassifier
import joblib

class NeuralNetworkModel:
    """Simple MLPClassifier wrapper."""

    def __init__(self, hidden_layer_sizes=(64, 32), activation='relu', max_iter=300, random_state=42):
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            max_iter=max_iter,
            random_state=random_state
        )

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
