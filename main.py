from model.load_data import load_stanford_dogs_data, split_data
from model.build_model import build_stanford_dogs_model
from model.train_evaluate import compile_model, train_model, evaluate_model, visualize_results

# Load and preprocess data 
# DATA_DIR = "data/Images"
DATA_DIR = "data/Preprocessed"
#'n02088094-Afghan_hound', 'n02099601-golden_retriever'
selected_breeds = ['Keeshond', 'Pomeranian', 'Standard Poodle']
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 25

X_gray, Y_rgb, y_labels, breed_mapping = load_stanford_dogs_data(DATA_DIR, IMAGE_SIZE)
X_train_gray, X_test_gray, Y_train_rgb, Y_test_rgb, y_train_labels, y_test_labels = split_data(X_gray, Y_rgb, y_labels)

# Build and compile model
model = build_stanford_dogs_model(input_shape=(64, 64, 1), num_breeds=len(breed_mapping))
compile_model(model)

# Train model
history = train_model(model, X_train_gray, y_train_labels, Y_train_rgb, X_test_gray, y_test_labels, Y_test_rgb, epochs=EPOCHS)

# Evaluate model
results = evaluate_model(model, X_test_gray, y_test_labels, Y_test_rgb)
print(f"Test Losses: {results}")

# Visualize results
visualize_results(model, X_test_gray, Y_test_rgb, breed_mapping)