import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Dataset
marks = np.array([20, 25, 35, 42, 50, 55, 60, 72, 85, 90]).reshape(-1, 1)
labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# Train model
model = LogisticRegression()
model.fit(marks, labels)

# User input
score = float(input("Enter student marks: "))

# Predict probability
prob = model.predict_proba([[score]])[0][1]  # probability of PASS

print(f"\nPass Probability = {prob*100:.2f}%")

# Decision boundary
boundary = -(model.intercept_[0] / model.coef_[0][0])
print(f"Decision boundary: {boundary:.2f}")

# Plot logistic curve
x_test = np.linspace(0, 100, 300).reshape(-1, 1)
probs = model.predict_proba(x_test)[:, 1]
#   predict_proba() → gives two probabilities (fail & pass) [0.99, 0.01],   
#   [:, 1] --> Probability of class 1 (pass)


plt.plot(x_test, probs, color="red", label="Pass Probability Curve")
plt.axvline(boundary, color="green", linestyle="--", label="Boundary")
plt.scatter([score], [prob], color="blue", label="Your Score")
plt.legend()
plt.xlabel("Marks")
plt.ylabel("Probability of Passing")
plt.title("Pass Probability Predictor")
plt.grid(True)
plt.show(block=False)

