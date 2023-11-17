import tensorflow as tf
import numpy as np
# Print confusion matrix in matplotlib
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def create_non_pre_trained_model():
    input_shape = (32, 32, 3)
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def create_non_pre_trained_model2():
    input_shape = (32, 32, 3)
    inputs = tf.keras.layers.Input(shape=input_shape)
    x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = tf.keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(10, activation='softmax')(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def create_model_from_pre_trained(model):
    x = tf.keras.layers.Flatten()(model.output)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(10, activation='softmax')(x)
       
    return tf.keras.Model(inputs=model.input, outputs=x)

def get_subset_data(data, data_y, fraction=0.01):
    """Gets a random subset of the data to train on, simulating the effect of having
    few labeled data samples.

    Args:
        data: the original data fully labeled
        data_y: the original labels
        fraction (float, optional): fraction of original dataset. Defaults to 0.01.

    Returns:
        _type_: _description_
    """
    indices = np.random.choice(len(data), int(len(data)*fraction), replace=False)
    return data[indices], data_y[indices]

def fine_tune(pre_trained_model, fraction=0.01):
    """_summary_

    Args:
        pre_trained_model (_type_): _description_
        fraction (float, optional): _description_. Defaults to 0.01.
    """
    RANDOM_SEED = 123
        
    # Build fine-tuning model
    fine_tuning_model = create_model_from_pre_trained(pre_trained_model)    
    fine_tuning_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    non_pretrained_model = create_non_pre_trained_model()
    non_pretrained_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    non_pretrained_model2 = create_non_pre_trained_model2()
    non_pretrained_model2.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Load cifar10 data
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
    
    # Get random subset of training data
    np.random.seed(RANDOM_SEED)
    x_train, y_train = get_subset_data(x_train, y_train, fraction=fraction)
    
    non_pretrained_model.fit(x_train, y_train, epochs=10, batch_size=32)
    non_pretrained_model2.fit(x_train, y_train, epochs=10, batch_size=32)
    fine_tuning_model.fit(x_train, y_train, epochs=10, batch_size=32)
    
    evaluation_non_pre_trained = non_pretrained_model.evaluate(x_test, y_test)
    evaluation_non_pre_trained2 = non_pretrained_model2.evaluate(x_test, y_test)
    evaluation_pre_trained = fine_tuning_model.evaluate(x_test, y_test)
    
    print(f"Evaluation non pre-trained model: {evaluation_non_pre_trained[1]}")
    print(f"Evaluation non pre-trained model dropout: {evaluation_non_pre_trained2[1]}")
    print(f"Evaluation pre-trained model: {evaluation_pre_trained[1]}")
    
    # Confusion matrix that shows difference in accuracy between pre-trained and non pre-trained model on test data for different classes
    y_pred_classes_pretrained = np.argmax(fine_tuning_model.predict(x_test), axis=1)
    y_true = np.squeeze(y_test)
    
    y_pred_classes = np.argmax(non_pretrained_model2.predict(x_test), axis=1)
    
    cm_pretrained = confusion_matrix(y_true, y_pred_classes_pretrained)
    cm_non_pretrained = confusion_matrix(y_true, y_pred_classes)

    # Create a heatmap
    label_dict = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    # Create a figure with 1 row and 2 columns for subplots
    fig, ax = plt.subplots(1, 2, figsize=(20, 8)) # Adjust the figsize as needed

    # Plot the first heatmap
    sns.heatmap(cm_pretrained, annot=True, cmap='coolwarm', fmt='.2f', 
                xticklabels=label_dict, yticklabels=label_dict, ax=ax[0])
    ax[0].set_title('Pretrained Model')
    ax[0].set_xlabel('Predicted Label')
    ax[0].set_ylabel('True Label')

    # Plot the second heatmap
    sns.heatmap(cm_non_pretrained, annot=True, cmap='coolwarm', fmt='.2f', 
                xticklabels=label_dict, yticklabels=label_dict, ax=ax[1])
    ax[1].set_title('Non-Pretrained Model')
    ax[1].set_xlabel('Predicted Label')
    ax[1].set_ylabel('True Label')

    # Display the plot
    plt.tight_layout() # Adjusts the plots to fit into the figure area.
    plt.show()

        