import numpy as np

class SingleLayerPerceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=20):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = np.zeros(input_size)
        self.bias = 0.0

    def activation(self, z):
        return 1 if z >= 0 else 0

    def predict(self, x):
        z = np.dot(x, self.weights) + self.bias
        return self.activation(z)

    def fit(self, X, y):
        for epoch in range(self.epochs):
            total_errors = 0
            for xi, target in zip(X, y):
                prediction = self.predict(xi)
                error = target - prediction
                
                if error != 0:
                    update = self.lr * error
                    self.weights += update * xi
                    self.bias += update
                    total_errors += 1
            
            if total_errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                break

if __name__ == "__main__":
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([0, 0, 0, 1])

    slp = SingleLayerPerceptron(input_size=2, learning_rate=0.1, epochs=20)
    slp.fit(X, y)

    print(f"Learned Weights: {slp.weights}")
    print(f"Learned Bias: {slp.bias}\n")

    print("Inference Results:")
    for xi in X:
        pred = slp.predict(xi)
        print(f"Input: {xi} -> Predicted: {pred}")