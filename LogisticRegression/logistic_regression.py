import numpy as np

class LogisticRegression:
    def __init__(self, lr = 0.001, n_iter = 900):
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):

            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = sigmoid(linear_pred)

            dw = (1 / n_samples) + np.dot(X.T, (predictions - y))     # partial derivation of the loss function tends to this. (There is also a constant 2, which is ignored because it gets absorbed into the learning rate ahead.)
            db = (1 / n_samples) + np.sum(predictions - y)       # partial derivation of bias tends to this.

            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = sigmoid(linear_pred)  
        class_pred = [0 if y <= 0.5 else 1 for y in y_pred]
        return class_pred
    
def sigmoid(x):
    return 1 / (1 + np.exp(-x))