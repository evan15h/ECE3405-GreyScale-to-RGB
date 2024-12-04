import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Dense, concatenate
from tensorflow.keras.models import Model

def build_stanford_dogs_model(input_shape=(64, 64, 1), num_breeds=120):
    # Input: Grayscale Image
    input_img = Input(shape=input_shape)

    # Shared Encoder
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    encoded_features = MaxPooling2D((2, 2), padding='same')(x)
    print("Encoded features shape:", encoded_features.shape)

    # Branch 1: Breed Classification
    x_class = Flatten()(encoded_features)
    print("Flattened features shape:", Flatten()(encoded_features).shape)
    breed_prediction = Dense(num_breeds, activation='softmax', name='breed_classification')(x_class)

    # Branch 2: Grayscale-to-RGB Colorization
    breed_embedding = Dense(128, activation='relu')(breed_prediction)
    print("Breed embedding shape:", breed_embedding.shape)
    combined_features = concatenate([Flatten()(encoded_features), breed_embedding])
    print("Combined features shape:", combined_features.shape)

    # Adjust the target shape for reshaping
    adjusted_features = Dense(16 * 16 * 64, activation='relu')(combined_features)
    reshaped_features = tf.keras.layers.Reshape((16, 16, 64))(adjusted_features)


    x_color = UpSampling2D((2, 2))(reshaped_features)
    x_color = Conv2D(64, (3, 3), activation='relu', padding='same')(x_color)
    x_color = UpSampling2D((2, 2))(x_color)
    output_img = Conv2D(3, (3, 3), activation='sigmoid', padding='same', name='rgb_output')(x_color)

    # Combine the model
    model = Model(inputs=input_img, outputs=[breed_prediction, output_img])

    return model

