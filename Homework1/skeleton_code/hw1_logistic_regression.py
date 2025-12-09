# Deep Learning Homework 1 - Question 2: Logistic Regression

import numpy as np
import pickle

class SoftmaxRegressionL2:
    def __init__(self, learning_rate=0.0001, l2_penalty=0.00001, epochs=10, n_classes=26, n_features=784):
        self.lr = learning_rate
        self.l2 = l2_penalty
        self.epochs = epochs
        self.n_classes = n_classes
        self.n_features = n_features
        limit = np.sqrt(6 / (n_features + n_classes))
        self.W = np.random.uniform(-limit, limit, (n_classes, n_features))
        self.b = np.zeros(n_classes)
        self.loss_history = []
        self.train_acc_history = []
        self.val_acc_history = []

    def save(self, path):
        """
        Save perceptron to the provided path
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        """
        Load perceptron from the provided path
        """
        with open(path, "rb") as f:
            return pickle.load(f)

    def softmax(self, z):
        # Shift z for numerical stability
        e_z = np.exp(z)
        return e_z / np.sum(e_z)


    def train_epoch(self, X, y):
        n_samples = X.shape[0]
        # Shuffle data for SGD
        indices = np.arange(n_samples)
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        epoch_loss = 0
        
        # STOCHASTIC GRADIENT DESCENT (Batch Size = 1)
        for i in range(n_samples):
            x_i = X_shuffled[i]      # (784,)
            y_idx = y_shuffled[i]    # scalar index
            
            # 1. Forward Pass
            z = np.dot(self.W, x_i) + self.b
            probs = self.softmax(z)
            
            # 2. Compute Loss (Cross Entropy + L2 Penalty)
            # CE Loss
            ce_loss = -np.log(probs[y_idx] + 1e-9)
            # L2 Loss (Sum of squared weights)
            l2_loss = (self.l2 / 2) * np.sum(self.W**2)
            
            epoch_loss += (ce_loss + l2_loss)
            
            # 3. Backward Pass
            # Create one-hot vector for true label
            y_one_hot = np.zeros(self.n_classes)
            y_one_hot[y_idx] = 1
            
            # Gradient of Cross Entropy
            error = probs - y_one_hot         # (26,)
            dW_data = np.outer(error, x_i)    # (26, 784)
            db_data = error                   # (26,)
            
            # Add L2 Gradient Penalty (Standard gradient update for L2)
            # Note: We usually do NOT regularize bias
            dW_total = dW_data + (self.l2 * self.W)
            
            # 4. Update Weights
            self.W -= self.lr * dW_total
            self.b -= self.lr * db_data
        
        # Average loss for reporting
        avg_loss = epoch_loss / n_samples
        self.loss_history.append(avg_loss)


    def fit(self, X, y):
        for _ in range(self.epochs):
            self.train_epoch(X, y)

    def predict(self, X):
        z = np.dot(X, self.W.T) + self.b
        return np.argmax(z, axis=1)
