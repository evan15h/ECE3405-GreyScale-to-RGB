from model.load_data import load_stanford_dogs_data, split_data
from model.build_model import build_stanford_dogs_model
from model.train_evaluate import compile_model, train_model, evaluate_model, visualize_results, plot_training_history
from model.breed_classifier import train_breed_classifier

# Load and preprocess data 
# DATA_DIR = "data/Images"
DATA_DIR = "/Users/paigerust/Desktop/ECE3405/Final/Preprocessed"
MASK_DIR = "/Users/paigerust/Desktop/ECE3405/Final/Masks"
# 'n02099601-golden_retriever'
# selected_breeds = ['Keeshond', 'Afghan Hound', 'Golden Retriever', 'Standard Poodle']
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 16
EPOCHS = 20

X_gray, Y_rgb, y_labels, breed_mapping = load_stanford_dogs_data(DATA_DIR, MASK_DIR, IMAGE_SIZE)
X_train_gray, X_test_gray, Y_train_rgb, Y_test_rgb, y_train_labels, y_test_labels = split_data(X_gray, Y_rgb, y_labels)

# Build and compile model
model = build_stanford_dogs_model(input_shape=(128, 128, 2), num_breeds=len(breed_mapping))
compile_model(model)

# Train model
history = train_model(model, X_train_gray, y_train_labels, Y_train_rgb, X_test_gray, y_test_labels, Y_test_rgb, epochs=EPOCHS)

# Plot training history
plot_training_history(history)

# Evaluate model
results = evaluate_model(model, X_test_gray, y_test_labels, Y_test_rgb)
print(f"Test Losses: {results}")

# Visualize results
visualize_results(model, X_test_gray, Y_test_rgb, breed_mapping)