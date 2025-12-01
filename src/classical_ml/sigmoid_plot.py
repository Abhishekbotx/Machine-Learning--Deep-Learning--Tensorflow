import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
# Sigmoid function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Generate values for x-axis
x = np.linspace(-10, 10, 200)
y = sigmoid(x)

# Plot
plt.plot(x, y)
plt.title("Sigmoid Function")
plt.xlabel("x")
plt.ylabel("σ(x)")
plt.grid(True)
plt.show()
