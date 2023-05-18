import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import numpy as np
from PIL import Image
import os
from binascii import a2b_base64
from services.speech import SpeechResponseEngine

speak = SpeechResponseEngine()


class Classifier:
    def __init__(self, trained_model: str = "trained_model"):
        self._load_model(trained_model)
        self.on_training = False

    async def predict_label(self, data_url: str):

        data_url = data_url.split("base64,")[1];
        binary_data = a2b_base64(data_url)
        with open('temp.png', 'wb') as f:
            f.write(binary_data)
        image = cv2.imread('temp.png')
        cropped = Classifier.image_to_bound(image)
        os.remove("temp.png")
        cv2.imwrite("temp.png", cropped)
        img = Image.open("temp.png")
        processed = self.__preprocess_image(img)
        prediction = self.model.predict(processed)
        label = self.labels[np.argmax(prediction).item()]
        confidence = np.max(prediction).item()
        #os.remove("temp.jpg")
        audio = speak.get_binary_audio(label)
        return 200, {"label": label, "confidence": confidence, "bAudio": audio}, None;

    def _load_model(self, trained_model="trained_model"):
        self.model = tf.keras.models.load_model(trained_model)
        self.labels = []
        with open(os.path.dirname(__file__) + "/../labels", 'r') as file:
            self.labels = [label.strip() for label in file]

    def __preprocess_image(self, image):
        image = image.convert('RGB')  # Convert to RGB format
        image = image.resize((224, 224))  # Resize to the same size used during training
        image = np.array(image)  # Convert to numpy array
        image = image / 255.0  # Normalize pixel values
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        return image

    def retrain_model(self):
        self.on_training = True
        Classifier.make_and_export_model()
        self._load_model()
        self.on_training = False
        return 200

    @staticmethod
    def make_and_export_model(path_to_training_dataset: str = "dataset", export_folder_name: str = "trained_model"):
        # Define hyper parameters
        batch_size = 32
        epochs = 10
        image_size = (224, 224)  # Set the desired image size

        # Create ImageDataGenerator to perform data augmentation
        datagen = ImageDataGenerator(rescale=1. / 255,
                                     validation_split=0.2)  # You can customize data augmentation parameters here

        # Load training data
        train_generator = datagen.flow_from_directory(
            path_to_training_dataset,  # Path to the dataset folder
            target_size=image_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',  # Use 80% of the data for training
        )

        # Load validation data
        val_generator = datagen.flow_from_directory(
            path_to_training_dataset,  # Path to the dataset folder
            target_size=image_size,
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',  # Use 20% of the data for validation
        )

        # Define the model architecture
        model = Sequential([
            Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
            MaxPooling2D((2, 2)),
            Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Conv2D(128, (3, 3), activation='relu'),
            MaxPooling2D((2, 2)),
            Flatten(),
            Dense(128, activation='relu'),
            Dense(train_generator.num_classes, activation='softmax')
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(
            train_generator,
            epochs=epochs,
            validation_data=val_generator
        )

        labels = train_generator.class_indices

        with open(os.path.dirname(__file__) + "/../labels", "w") as file:
            for label in labels:
                file.write(f'{label}\n')

        # Save the trained model
        model.save(export_folder_name)

    @staticmethod
    def add_to_model(label: str, dataset: [str]):
        os.mkdir(os.path.dirname(__file__) + f"/../dataset/{label}")
        for index, item in enumerate(dataset):
            item = item.split("base64,")[1];
            binary_data = a2b_base64(item)
            path = os.path.dirname(__file__) + f"/../dataset/{label}/{index}.png"
            with open(path, 'wb') as f:
                f.write(binary_data)
                image = cv2.imread(path)
                cropped = Classifier.image_to_bound(image)
                os.remove(path)
                cv2.imwrite(path, cropped)

        return 200

    @staticmethod
    def image_to_bound(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        lower_red = (0, 50, 50)
        upper_red = (10, 255, 255)
        lower_green = (40, 50, 50)
        upper_green = (80, 255, 255)

        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        red_mask = cv2.inRange(hsv_image, lower_red, upper_red)
        green_mask = cv2.inRange(hsv_image, lower_green, upper_green)
        combined_mask = cv2.bitwise_or(red_mask, green_mask)

        contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            x, y, w, h = 0, 0, 0, 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)

            return image[y:y + h, x:x + w]

        return image


if __name__ == "__main__":
    Classifier.make_and_export_model()
