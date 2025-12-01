import numpy as np
import matplotlib.pyplot as plt

def loss(x): return (x - 3)**2
def gradient(x): return 2 * (x - 3)

x = 0
lr = 0.1
history = [x]

for i in range(40):
    grad = gradient(x)
    x = x - lr * grad
    history.append(x)

print("Final x:", x)


# Example: 2
# Simple gradient descent visualization

# Loss function: y = (x - 3)^2
def loss(x):
    return (x - 3) ** 2

# Derivative
def gradient(x):
    return 2 * (x - 3)

# Gradient descent
x = 0  # Starting point
learning_rate = 0.1
history = [x]

for i in range(50):
    grad = gradient(x)
    x = x - learning_rate * grad
    history.append(x)

# Plot
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
x_vals = np.linspace(-1, 6, 100)
plt.plot(x_vals, loss(x_vals))
plt.scatter(history, [loss(h) for h in history], c='red', s=10)
plt.title("Gradient Descent Path")

plt.subplot(1, 2, 2)
plt.plot(history)
plt.title("X value over iterations")
plt.xlabel("Iteration")
plt.show()
