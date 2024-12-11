import matplotlib.pyplot as plt
import numpy as np

def compile_model(model):
    model.compile(
        optimizer='adam',
        loss={
            'breed_classification': 'categorical_crossentropy',
            'rgb_output': 'mse'
        },
        loss_weights={
            'breed_classification': 0.5,
            'rgb_output': 1.0
        },
        metrics={
            'breed_classification': 'accuracy',
            'rgb_output': 'mae'
        }
    )

def train_model(model, X_train_gray, y_train_labels, Y_train_rgb, X_test_gray, y_test_labels, Y_test_rgb, epochs=50, batch_size=32):
    return model.fit(
        X_train_gray,
        {'breed_classification': y_train_labels, 'rgb_output': Y_train_rgb},
        validation_data=(
            X_test_gray, 
            {'breed_classification': y_test_labels, 'rgb_output': Y_test_rgb}
        ),
        epochs=epochs,
        batch_size=batch_size
    )

def evaluate_model(model, X_test_gray, y_test_labels, Y_test_rgb):
    return model.evaluate(
        X_test_gray,
        {'breed_classification': y_test_labels, 'rgb_output': Y_test_rgb}
    )

def visualize_results(model, X_test_gray, Y_test_rgb, breed_mapping):
    predictions = model.predict(X_test_gray)
    breed_predictions, rgb_predictions = predictions
    reverse_breed_mapping = {v: k for k, v in breed_mapping.items()}

    for i in range(5):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 3, 1)
        plt.imshow(X_test_gray[i, :, :, 0], cmap='gray')  # Use only the first channel for grayscale
        plt.title("Grayscale Input")
        plt.axis('off')

        plt.subplot(1, 3, 2)
        plt.imshow(rgb_predictions[i])
        plt.title(f"Predicted RGB ({reverse_breed_mapping[np.argmax(breed_predictions[i])]})")
        plt.axis('off')

        plt.subplot(1, 3, 3)
        plt.imshow(Y_test_rgb[i])
        plt.title("Ground Truth RGB")
        plt.axis('off')

        plt.show()

def plot_training_history(history):
    # Plot loss for both outputs
    plt.figure(figsize=(12, 6))

    # Breed classification loss
    plt.subplot(2, 2, 1)
    plt.plot(history.history['breed_classification_loss'], label='Train Breed Loss')
    plt.plot(history.history['val_breed_classification_loss'], label='Validation Breed Loss')
    plt.title('Breed Classification Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # RGB output loss
    plt.subplot(2, 2, 2)
    plt.plot(history.history['rgb_output_loss'], label='Train RGB Loss')
    plt.plot(history.history['val_rgb_output_loss'], label='Validation RGB Loss')
    plt.title('RGB Output Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Breed classification accuracy
    plt.subplot(2, 2, 3)
    plt.plot(history.history['breed_classification_accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_breed_classification_accuracy'], label='Validation Accuracy')
    plt.title('Breed Classification Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # RGB output Mean Absolute Error (MAE)
    plt.subplot(2, 2, 4)
    plt.plot(history.history['rgb_output_mae'], label='Train MAE')
    plt.plot(history.history['val_rgb_output_mae'], label='Validation MAE')
    plt.title('RGB Output Mean Absolute Error')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.show()
