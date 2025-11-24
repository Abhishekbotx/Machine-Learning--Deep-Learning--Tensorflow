import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Normalize
x_train = x_train / 255.0
x_test = x_test / 255.0

#Why dividing by 255? Because MNIST images are grayscale images, and each pixel value is between:
#0 → black    255 → white
# We divide by 255 to scale pixel values into a smaller range (0–1), which helps the neural network train faster, converge smoother, and avoid exploding gradients.

print(f"Training data shape: {x_train.shape}")
print(f"Test data shape: {x_test.shape}")

# Visualize samples
plt.figure(figsize=(10, 4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_train[i], cmap='gray') #28 x 28 matrix 
    # imshow() = display an image
    # x_train[i] = the i-th handwritten digit from the MNIST dataset (it will be random)

    plt.title(f"Label: {y_train[i]}") # prints the actual digit value (0–9) from the dataset
    plt.axis('off') #off is used to remove the axes, ticks, borders, so image looks clean
plt.show()


