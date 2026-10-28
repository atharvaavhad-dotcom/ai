import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))

    def train(self, patterns):
        num_patterns = len(patterns)
        for p in patterns:
            self.weights += np.outer(p, p)
        np.fill_diagonal(self.weights, 0)
        self.weights /= num_patterns

    def recall(self, pattern, max_iter=20):
        state = np.copy(pattern)
        for _ in range(max_iter):
            prev_state = np.copy(state)
            indices = np.random.permutation(self.size)
            for i in indices:
                activation = np.dot(self.weights[i], state)
                state[i] = 1 if activation >= 0 else -1
            if np.array_equal(state, prev_state):
                break
        return state

if __name__ == "__main__":
    p1 = np.array([1, 1, -1, -1])
    p2 = np.array([-1, 1, -1, 1])
    patterns = [p1, p2]

    net = HopfieldNetwork(size=4)
    net.train(patterns)

    noisy_input = np.array([1, -1, -1, -1])
    retrieved = net.recall(noisy_input)

    print("Stored Pattern 1:", p1)
    print("Stored Pattern 2:", p2)
    print("Noisy Input:     ", noisy_input)
    print("Retrieved Pattern:", retrieved)