# ensemble/naive_bayes.py
from sklearn.naive_bayes import GaussianNB, MultinomialNB
import joblib

class NaiveBayesModel:
    """Naive Bayes wrapper for Gaussian or Multinomial variant."""

    def __init__(self, variant: str = "gaussian"):
        if variant == "gaussian":
            self.model = GaussianNB()
        elif variant == "multinomial":
            self.model = MultinomialNB()
        else:
            raise ValueError("variant must be 'gaussian' or 'multinomial'")

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
