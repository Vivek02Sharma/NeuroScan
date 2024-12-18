import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

class BrainTumorClassifier:
    def __init__(self, dataset_path, img_size = 300, batch_size = 32):
        self.dataset_path = dataset_path
        self.img_size = img_size
        self.batch_size = batch_size
        self.model = None
        self.history = None
        self.class_dict = None
        self.class_names = None
        self.train_data = None
        self.validation_data = None
        
        # Create ImageDataGenerators
        self.train_datagen = ImageDataGenerator(rescale = 1/255,
                                                brightness_range = (0.8, 1.2),
                                                validation_split = 0.2)
        self.test_datagen = ImageDataGenerator(rescale = 1/255, validation_split=0.2)

    def load_data(self):
        try: 
             # Load training data
            self.train_data = self.train_datagen.flow_from_directory(self.dataset_path,
                                                                    target_size = (self.img_size, self.img_size),
                                                                    batch_size = self.batch_size,
                                                                    class_mode = 'categorical',
                                                                    color_mode = 'grayscale',
                                                                    subset = 'training')
            # Load validation data
            self.validation_data = self.test_datagen.flow_from_directory(self.dataset_path,
                                                                        target_size = (self.img_size, self.img_size),
                                                                        batch_size = self.batch_size,
                                                                        class_mode = 'categorical',
                                                                        color_mode = 'grayscale',
                                                                        subset = 'validation')
            
            # Map class indices to class names
            self.class_dict = self.train_data.class_indices
            self.class_names = {v: k for k, v in self.class_dict.items()}
            
        except FileNotFoundError as e:
            print(f"Error: Dataset path '{self.dataset_path}' not found. {e}")
        except Exception as e:
            print(f"An error occurred while loading data: {e}")

    def visualize_sample_images(self):
        try:
            images, labels = next(self.validation_data)
            plt.figure(figsize = (10, 10))
            for i in range(16):
                plt.subplot(4, 4, i + 1)
                plt.axis('off')
                label_index = np.argmax(labels[i])
                class_name = self.class_names[label_index]
                plt.title(class_name)
                plt.imshow(images[i])
            plt.show()
        except Exception as e:
            print(f"An error occurred while visualizing sample images: {e}")

    def create_model(self):
        try:
            self.model = Sequential([
                Input(shape = (self.img_size, self.img_size, 1)),
                Conv2D(32, kernel_size = (3, 3), strides = 1, padding = 'same', activation = 'relu'),
                MaxPooling2D(pool_size = (2, 2), strides = 2, padding = 'valid'),

                Conv2D(64, kernel_size = (3, 3), strides = 1, padding = 'same', activation = 'relu'),
                MaxPooling2D(pool_size = (2, 2), strides = 2, padding = 'valid'),

                Conv2D(128, kernel_size = (3, 3), strides = 1, padding = 'same', activation = 'relu'),
                MaxPooling2D(pool_size = (2, 2), strides = 2, padding = 'valid'),

                Flatten(),
                Dense(128, activation = 'relu'),
                Dropout(0.5),

                Dense(4, activation = 'softmax')
            ])
            self.model.summary()
            self.model.compile(optimizer = Adam(learning_rate = 0.001), loss = 'categorical_crossentropy', metrics = ['accuracy'])
        except Exception as e:
            print(f"An error occurred while creating the model: {e}")

    def train_model(self, epochs = 10):
        try:
            self.history = self.model.fit(self.train_data,
                                          epochs = epochs,
                                          validation_data = self.validation_data,
                                          batch_size = self.batch_size)
        except Exception as e:
            print(f"An error occurred during model training: {e}")

    def save_model(self, model_filename = 'brain-tumor-model.h5'):
        try:
            if self.model:
                self.model.save(model_filename)
                print(f"Model saved to {model_filename}")
            else:
                print("Error: Model is not created yet.")
        except Exception as e:
            print(f"An error occurred while saving the model: {e}")

    def plot_training_history(self):
        try:
            if self.history:
                plt.plot(self.history.history['accuracy'], label = 'Training Accuracy')
                plt.plot(self.history.history['val_accuracy'], label = 'Validation Accuracy')
                plt.title('Training and Validation Accuracy')
                plt.xlabel('Epochs')
                plt.ylabel('Accuracy')
                plt.legend()
                plt.show()

                plt.plot(self.history.history['loss'], label = 'Training Loss')
                plt.plot(self.history.history['val_loss'], label = 'Validation Loss')
                plt.title('Training and Validation Loss')
                plt.xlabel('Epochs')
                plt.ylabel('Loss')
                plt.legend()
                plt.show()
            else:
                print("Error: Model has not been trained yet.")
        except Exception as e:
            print(f"An error occurred while plotting training history: {e}")

    def evaluate_model(self):
        try:
            if self.model:
                loss, accuracy = self.model.evaluate(self.validation_data)
                print(f"Test Loss: {loss}")
                print(f"Test Accuracy: {accuracy}")
            else:
                print("Error: Model is not created yet.")
        except Exception as e:
            print(f"An error occurred during model evaluation: {e}")

if __name__ == "__main__":

    dataset_path = './brain-tumor-dataset'

    classifier = BrainTumorClassifier(dataset_path)

    classifier.load_data()
    classifier.visualize_sample_images()
    classifier.create_model()
    classifier.train_model(epochs = 10)
    classifier.save_model('brain-tumor-model.h5')
    classifier.plot_training_history()
    classifier.evaluate_model()
