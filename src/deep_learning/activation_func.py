import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)

sigmoid = 1 / (1 + np.exp(-x))
relu = np.maximum(0, x)
tanh = np.tanh(x)

plt.figure(figsize=(15, 4))

plt.subplot(1, 2, 1)
plt.plot(x, sigmoid)
plt.title("Sigmoid")

plt.subplot(1, 2, 2)
plt.plot(x, relu)
plt.title("ReLU")

plt.show()
