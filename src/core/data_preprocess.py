import os
# Suppress TensorFlow warnings and info logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0 = all logs, 1 = info, 2 = warnings, 3 = errors

# Optional: Disable GPU completely if it's causing runtime errors
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tensorflow as tf
import keras
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from keras.applications.resnet50 import preprocess_input

class DataPreprocessor:
    """
    Class for preprocessing brain tumor image data, including:
    - Label encoding
    - Train/validation/test splitting
    - Oversampling
    - Data augmentation
    """

    def __init__(self, img_size=(224, 224), batch_size=32, random_state=42):
        self.img_size = img_size
        self.batch_size = batch_size
        self.random_state = random_state
        self.label_encoder = LabelEncoder()

    def encode_labels(self, df):
        """Encode categorical labels to numerical values"""
        df['category_encoded'] = self.label_encoder.fit_transform(df['label'])

        print("DataFrame after encode labels:")
        print(df.head(3))
        print(f"encoded: {self.label_encoder.classes_} -> {self.label_encoder.transform(self.label_encoder.classes_)}")
        
        return df

    def split_data(self, df):
        """Split data into train, validation and test sets"""
        # First split: 80% train, 20% temp (val + test)
        X_train_original, X_temp, y_train_original, y_temp = train_test_split(
            df[['image_path']],
            df['category_encoded'],
            train_size=0.8,
            shuffle=True,
            random_state=self.random_state,
            stratify=df['category_encoded']
        )
        # Second split: from temp -> 50% for validation and 50% for test
        X_valid, X_test, y_valid, y_test = train_test_split(
            X_temp,
            y_temp,
            test_size=0.5,
            shuffle=True,
            random_state=self.random_state,
            stratify=y_temp
        )
        return X_train_original, y_train_original, X_valid, y_valid, X_test, y_test

    def oversample_training_data(self, X_train, y_train):
        """Apply oversampling to balance the training dataset"""
        ros = RandomOverSampler(random_state=self.random_state)
        resampled = ros.fit_resample(X_train, y_train)
        X_train_resampled, y_train_resampled = resampled[0], resampled[1]
        return X_train_resampled, y_train_resampled

    def create_data_augmentation(self):
        """Create data augmentation layer using Keras preprocessing layers"""
        return keras.Sequential([
            keras.layers.RandomRotation(0.2),
            keras.layers.RandomTranslation(0.2, 0.2),
            keras.layers.RandomZoom(0.2),
            keras.layers.RandomFlip("horizontal")
        ])

    def setup_data_flows(self, train_df, valid_df, test_df):
        """Setup tf.data.Dataset pipelines for training, validation and testing"""
        AUTOTUNE = tf.data.AUTOTUNE # Automatically tune the number of parallel calls
        augmentation = self.create_data_augmentation()  # ← Include augmentation

        def create_dataset(df, shuffle=True, augment=False):
            paths = df['image_path'].values
            labels = df['category_encoded'].astype(int).values
            ds = tf.data.Dataset.from_tensor_slices((paths, labels))

            def load_and_preprocess_image(path, label):
                image = tf.io.read_file(path)
                image = tf.image.decode_jpeg(image, channels=3)  # type: ignore
                image = tf.image.resize(image, self.img_size)
                image = preprocess_input(image) # handles scaling & mean subtraction
                if augment:
                    image = augmentation(image)  # Apply only if requested
                return image, label

            ds = ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
            if shuffle:
                ds = ds.shuffle(buffer_size=len(df))
            ds = ds.batch(self.batch_size).prefetch(AUTOTUNE) # prefetch for performance optimization
            return ds

        train_ds = create_dataset(train_df, shuffle=True, augment=True)
        valid_ds = create_dataset(valid_df, shuffle=False, augment=False)
        test_ds = create_dataset(test_df, shuffle=False, augment=False)

        return train_ds, valid_ds, test_ds


    def process_data(self, df):
        """Main method to run the complete preprocessing pipeline"""

        df = self.encode_labels(df)

        X_train, y_train, X_valid, y_valid, X_test, y_test = self.split_data(df)

        X_train_resampled, y_train_resampled = self.oversample_training_data(X_train, y_train)

        train_df = pd.DataFrame(X_train_resampled, columns=['image_path'])
        train_df['category_encoded'] = y_train_resampled.astype(str)

        valid_df = pd.DataFrame(X_valid, columns=['image_path'])
        valid_df['category_encoded'] = y_valid.astype(str)

        test_df = pd.DataFrame(X_test, columns=['image_path'])
        test_df['category_encoded'] = y_test.astype(str)

        train_ds, valid_ds, test_ds = self.setup_data_flows(train_df, valid_df, test_df)

        print("Data preprocessing completed successfully.")
        print(f"Train dataset : {train_ds}")

        return train_ds, valid_ds, test_ds, self.label_encoder
