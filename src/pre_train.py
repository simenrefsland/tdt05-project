import tensorflow as tf
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

class AutoEncoder():
    def __init__(self):
        self.encoder = self.create_encoder()
        self.decoder = self.create_decoder()
        self.autoencoder = self.create_autoencoder()
        
    def create_encoder(self):
        input_encoder = tf.keras.layers.Input(shape=(32, 32, 3))
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding="same")(input_encoder)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same")(x)
        x = tf.keras.layers.MaxPooling2D((2, 2), padding="same")(x)
        return tf.keras.Model(input_encoder, x, name="encoder")
    
    def create_decoder(self):
        input_decoder = tf.keras.layers.Input(shape=(8, 8, 64))
        
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding="same")(input_decoder)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding="same")(x)
        x = tf.keras.layers.UpSampling2D((2, 2))(x)
        decode_output = tf.keras.layers.Conv2D(3, (3, 3), activation='sigmoid', padding="same")(x)
        return tf.keras.Model(input_decoder, decode_output, name="decoder")
        
    def create_autoencoder(self):
        autoencoder_input = tf.keras.layers.Input(shape=(32, 32, 3))
        encoded_img = self.encoder(autoencoder_input)
        decoded_img = self.decoder(encoded_img)
        return tf.keras.Model(autoencoder_input, decoded_img, name='autoencoder')
        

def augment_training_data(data):
    """Augment training data using albumentations library

    Args:
        data: the data to augment

    Returns:
        Augmented data
    """
    transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=20, p=0.5),
        A.CoarseDropout(max_holes=2, min_holes=1, max_height=15, max_width=15, p=1),
    ])
    
    augmented_data = []
    for image in data:
        augmented_data.append(transform(image=image)['image'])
    
    augmented_data = np.array(augmented_data)
    
    return augmented_data

def plot_augmented_data(model, data, augmented_data):
    """Simple function to plot original data, augmented data and the reconstructed data

    Args:
        model: the model to use for reconstruction
        data: original data
        augmented_data: the augmented data
    """
    predictions = model.predict(augmented_data)
    
    _, axes = plt.subplots(3, 3, figsize=(10,10))
    
    for i in range(3):
        axes[i, 0].imshow(data[i])
        axes[i, 1].imshow(augmented_data[i])
        axes[i, 2].imshow(predictions[i])
    
    plt.show()
        

def pre_train_model(visualize=False):
    """Pre-trains the model using autoencoder

    Args:
        visualize (bool, optional): Whether to show the comparison of original, 
        augmented and reconstructed data. Defaults to False.

    Returns:
        AutoEncoder: The autoencoder model
    """
    # Load CIFAR10 data
    (x_train, _), (x_val, _) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.
    x_val = x_val.astype('float32') / 255.
    
    x_train_augmented = augment_training_data(x_train)
    x_val_augmented = augment_training_data(x_val)
    
    autoencoder = AutoEncoder()
    autoencoder.autoencoder.compile(optimizer='adam', loss='mse')
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)

    autoencoder.autoencoder.fit(x_train_augmented, x_train, epochs=10, batch_size=64, shuffle=True, validation_data=(x_val_augmented, x_val), callbacks=[callback])

    # Save encoder
    if visualize:
        plot_augmented_data(autoencoder.autoencoder, x_val[:3], x_val_augmented[:3])
    autoencoder.encoder.save('models/encoder.h5')
    
    return autoencoder
    
        