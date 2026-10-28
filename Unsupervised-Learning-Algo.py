import numpy as np

print("KOHONEN SELF ORGANIZING MAP (SOM)")
print("-----------------------------------")

X = np.array([
    [0.0, 0.0, 1.0, 1.0],
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0]
], dtype=float)

weights = np.array([
    [0.2, 0.4, 0.6, 0.8],
    [0.9, 0.7, 0.5, 0.3]
], dtype=float)

learning_rate = 0.5
epochs = 1

n = X.shape[0]
units = weights.shape[0]

def euclidean_distance(x, w):
    return np.sqrt(np.sum((x - w) ** 2))

for epoch in range(epochs):
    print("\nEpoch", epoch + 1)
    for i in range(n):
        x = X[i]
        
        distances = []
        for j in range(units):
            distance = euclidean_distance(x, weights[j])
            distances.append(distance)
            
        winner = np.argmin(distances)
        print("Input", i + 1, "-> Unit", winner + 1)
        
        weights[winner] = weights[winner] + learning_rate * (x - weights[winner])

print("\nTraining Completed")
print("\nFinal Weight Vectors")
print("--------------------------------")
for i in range(units):
    print("Unit", i + 1, ":", np.round(weights[i], 4))

print("\nFinal Customer/Input Clustering")
print("--------------------------------")
for i in range(n):
    x = X[i]
    distances = []
    for j in range(units):
        distance = euclidean_distance(x, weights[j])
        distances.append(distance)
    winner = np.argmin(distances)
    formatted_x = x.astype(int) if np.all(X[i] == X[i].astype(int)) else x
    print("Input", i + 1, formatted_x, "-> Unit", winner + 1)