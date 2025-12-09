import numpy as np
import pickle

class MLP:
    def __init__(self, n_samples, input_size=784, hidden_size=100, output_size=26, learning_rate=0.001, epochs=20):
        self.n_samples = n_samples
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate
        self.epochs = epochs
        
        # Initialize Weights and Biases
        # Requirement: Weights ~ N(0.1, 0.1^2), Biases = 0
        
        self.W1 = np.random.normal(loc=0.1, scale=0.1, size=(hidden_size, input_size))
        self.b1 = np.zeros(hidden_size)
        
        self.W2 = np.random.normal(loc=0.1, scale=0.1, size=(output_size, hidden_size))
        self.b2 = np.zeros(output_size)
        
        self.loss_history = []

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

    def relu(self, z):
        return np.maximum(0, z)

    def softmax(self, z):
        exp_z = np.exp(z)
        return exp_z / np.sum(exp_z)

    def train_epoch(self, X, y):
        # Shuffle for SGD
        indices = np.arange(self.n_samples)
        np.random.shuffle(indices)
        X_shuffled = X[indices]
        y_shuffled = y[indices]
        
        epoch_loss = 0
        
        for i in range(self.n_samples):
            x = X_shuffled[i]      # Shape (784,)
            target = y_shuffled[i] # Scalar label
            
            # --- FORWARD PASS ---
            
            # Layer 1: Input -> Hidden
            z1 = np.dot(self.W1, x) + self.b1
            h1 = self.relu(z1)
            
            # Layer 2: Hidden -> Output
            z2 = np.dot(self.W2, h1) + self.b2
            probs = self.softmax(z2)
            
            # Track Loss (Cross Entropy)
            loss = -np.log(probs[target] + 1e-9)
            epoch_loss += loss
            
            # --- BACKPROPAGATION ---
            
            # 1. Gradient at Output (delta2)
            y_one_hot = np.zeros(self.output_size)
            y_one_hot[target] = 1
            delta2 = probs - y_one_hot  # Shape (26,)
            
            # 2. Gradient for W2, b2
            dW2 = np.outer(delta2, h1)
            db2 = delta2
            
            # 3. Propagate error to Hidden Layer (delta1)
            error_hidden = np.dot(self.W2.T, delta2)
            
            # Multiply by ReLU derivative
            # derivative is 1 where z1 > 0, else 0
            relu_derivative = (z1 > 0).astype(float)
            delta1 = error_hidden * relu_derivative # Shape (100,)
            
            # 4. Gradient for W1, b1
            dW1 = np.outer(delta1, x)
            db1 = delta1
            
            self.W2 -= self.lr * dW2
            self.b2 -= self.lr * db2
            self.W1 -= self.lr * dW1
            self.b1 -= self.lr * db1
            
        avg_loss = epoch_loss / self.n_samples
        self.loss_history.append(avg_loss)
        return avg_loss

    def fit(self, X, y):
        for epoch in range(self.epochs):
            avg_loss = self.train_epoch(X, y)
            print(f"Epoch {epoch+1}/{self.epochs} - Loss: {avg_loss:.4f}")

    def predict(self, X):
        # Layer 1
        z1 = np.dot(X, self.W1.T) + self.b1
        h1 = self.relu(z1)
        
        # Layer 2
        z2 = np.dot(h1, self.W2.T) + self.b2
        
        return np.argmax(z2, axis=1)
