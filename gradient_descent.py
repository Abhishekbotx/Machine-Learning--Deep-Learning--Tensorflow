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


