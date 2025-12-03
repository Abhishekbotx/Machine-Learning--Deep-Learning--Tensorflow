import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

# Normalizing and reshape
x_train = x_train.astype("float32") / 255.
x_test = x_test.astype("float32") / 255.

x_train = x_train.reshape((len(x_train), 28, 28, 1))
x_test = x_test.reshape((len(x_test), 28, 28, 1))

# Encoder
encoder = models.Sequential([
    layers.InputLayer(input_shape=(28, 28, 1)),
    layers.Conv2D(32, (3, 3), activation='relu', strides=2, padding='same'),
    # stride is similar to max pooling but it is done in convolutional layer itself
    # stride of 2 means the filter moves 2 pixels at a time
    # 32 filters(kernels) of size 3x3 , each filter learns  different features ,eg: edges, corners, lines, textures,etc
    # Output shape: (14, 14, 32)
    # kernal size 3x3 means the filter is a 3x3 matrix that slides over the image to detect features
    
    layers.Conv2D(64, (3, 3), activation='relu', strides=2, padding='same'),
    layers.Flatten(),
    layers.Dense(64, activation='relu')
])

# Decoder
decoder = models.Sequential([
    layers.InputLayer(input_shape=(64,)),
    layers.Dense(7 * 7 * 64, activation='relu'),
    layers.Reshape((7, 7, 64)),
    layers.Conv2DTranspose(64, (3, 3), activation='relu', strides=2, padding='same'),    
    layers.Conv2DTranspose(32, (3, 3), activation='relu', strides=2, padding='same'),
    layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')
])

# Autoencoder 
autoencoder = models.Sequential([encoder, decoder])
autoencoder(tf.random.normal((1, 28, 28, 1)))
autoencoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train
autoencoder.fit(
    x_train, x_train,
    epochs=5,
    batch_size=128,
    validation_data=(x_test, x_test)
)

autoencoder.save('models/mnist_autoencoder.h5')

autoencoder = tf.keras.models.load_model('models/mnist_autoencoder.h5')

# Inference
decoded_images = autoencoder.predict(x_test)

# Visualization of original and reconstructed images
n = 10
plt.figure(figsize=(20, 4))

for i in range(n):
    
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28), cmap="gray")
    plt.title("Original")
    plt.axis("off")

    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_images[i].reshape(28, 28), cmap="gray")
    plt.title("Reconstructed")
    plt.axis("off")

plt.show()
