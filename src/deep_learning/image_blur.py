import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Loading an image and preprocessing
image = Image.open("./assets/cat.jpg").convert("L")   # Convert to grayscale
image = image.resize((256, 256))                  # Resize for consistency
image = np.array(image, dtype=np.float32) / 255.0 # Normalize to [0,1]

# Adding batch and channel dimensions → shape: (1, 256, 256, 1)
image = np.expand_dims(image, axis=(0, -1))

# Defining 3x3 Blurring Filter (Averaging Kernel) 
blur_filter = np.array([
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9],
    [1/9, 1/9, 1/9]
], dtype=np.float32)

# Reshaping to TensorFlow format: (kernel_height, kernel_width, in_channels, out_channels)
blur_filter = blur_filter.reshape(3, 3, 1, 1)

# Applying Convolution for Blurring
blurred_tensor = tf.nn.conv2d(image, blur_filter, strides=[1, 1, 1, 1], padding="SAME")

# Converting to NumPy for visualization
blurred_image = blurred_tensor.numpy().squeeze()

# Plotting original and blurred images
plt.figure(figsize=(8, 4))

plt.subplot(1, 2, 1)
plt.imshow(image.squeeze(), cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(blurred_image, cmap="gray")
plt.title("Blurred Image")
plt.axis("off")

plt.show()
