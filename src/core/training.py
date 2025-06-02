import tensorflow as tf
from keras.applications.resnet50 import ResNet50
from keras import layers, Model, callbacks, regularizers
from keras.optimizers import Adam
from keras.metrics import AUC
import matplotlib.pyplot as plt

class ModelTrainer:
    """
    Class for building and training ResNet50-based models for binary classification.
    """
    
    def __init__(self, input_shape=(224, 224, 3)):
        """
        Initialize the trainer with model configuration.
        
        :param input_shape: Shape of input images (height, width, channels)
        """
        self.input_shape = input_shape
        self.model = None
        self.history = None
        self.history_fine = None
    
    def build_resnet_model(self):
        """
        Build ResNet50-based model for binary classification.
        
        :return: Compiled Keras model
        """
        # Load pre-trained ResNet50 (without top classification layers)
        base_model = ResNet50(
            include_top=False, # Do not include the top dense layers
            weights='imagenet', # Use ImageNet weights
            input_shape=self.input_shape
        )
        
        # Freeze base model layers initially
        base_model.trainable = False # weights will not be updated during initial training

        # Print summary of the base model
        base_model.summary()
        print("\nBase model (ResNet50) loaded and trainable False. Summary:")

        # Build complete model with custom top layers
        inputs = layers.Input(shape=self.input_shape)
        x = base_model(inputs, training=False)  # training=False ????
        x = layers.GlobalAveragePooling2D()(x) # to reduce spatial dimensions
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(1, activation='sigmoid', # binary classification
                              kernel_regularizer=regularizers.l2(0.01))(x) # L2 regularization to prevent overfitting   
        
        self.model = Model(inputs, outputs)
        return self.model
    
    def compile_model(self, learning_rate=1e-3):
        """
        Compile the model with optimizer, loss, and metrics.
        
        :param learning_rate: Learning rate for the optimizer
        """
        if self.model is None:
            raise ValueError("Model must be built before compilation")
        
        self.model.compile(
            optimizer=Adam(learning_rate=learning_rate), #type: ignore
            loss='binary_crossentropy', # Binary crossentropy for binary classification
            metrics=['accuracy', AUC(name='auc')] # AUC metric for evaluation
        )
    
    def get_callbacks(self, model_save_path='best_resnet_model.h5'):
        """
        Get list of callbacks for training.
        
        :param model_save_path: Path to save the best model
        :return: List of callbacks
        """
        return [
            # Stop training when AUC stops improving
            callbacks.EarlyStopping( 
                monitor='val_auc',
                patience=5, # epochs to wait before stopping
                mode='max', # We want to maximize AUC
                restore_best_weights=True,
                verbose=1
            ),
            # Reduce learning rate when validation loss plateaus
            callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1, # Reduce LR by this factor
                patience=3, # epochs to wait before reducing LR
                verbose=1
            ),
            # Save the best model based on validation AUC
            callbacks.ModelCheckpoint(
                model_save_path,
                monitor='val_auc',
                save_best_only=True,
                mode='max', # We want to maximize AUC
                verbose=1
            )
        ]
    
    def train_initial_phase(self, train_ds, valid_ds, initial_epochs=10, 
                           model_save_path='best_resnet_model.h5'):
        """
        Train only the new layers (Phase 1).
        
        :param train_ds: Training data set (tf.data.Dataset)
        :param valid_ds: Validation data set (tf.data.Dataset)
        :param initial_epochs: Number of epochs for initial training
        :param model_save_path: Path to save the best model
        :return: Training history
        """
        if self.model is None:
            raise ValueError("Model must be built and compiled before training")
        
        print("\nTraining new layers (Phase 1)...")
        callbacks_list = self.get_callbacks(model_save_path)
        
        self.history = self.model.fit(
            train_ds,
            validation_data=valid_ds,
            epochs=initial_epochs,
            callbacks=callbacks_list,
            verbose="auto"
        )
        return self.history
    
    def train_fine_tune(self, train_ds, valid_ds, fine_tune_epochs=10,
                  model_save_path='best_resnet_model.h5'):
        """
        Fine-tune the entire model (Phase 2).
        
        :param train_ds: Training data set (tf.data.Dataset)
        :param valid_ds: Validation data set (tf.data.Dataset)
        :param fine_tune_epochs: Number of epochs for fine-tuning
        :param model_save_path: Path to save the best model
        :return: Fine-tuning history
        """
        if self.history is None:
            raise ValueError("Initial training must be completed before fine-tuning")
        
        if self.model is None:
            raise ValueError("Base model not found. Please build the model first.")
        
        print("\nFine-tuning entire model (Phase 2)...")
        
        # Unfreeze the base model
        base_model_layer = self.model.layers[1] # Access the ResNet50 base model layer
        print(f"Base model layer: {base_model_layer}")

        base_model_layer.trainable = True # Make the base model trainable   
        
        # Recompile with lower learning rate
        # This is essential to not destroy the pre-trained weights
        self.model.compile(
            optimizer=Adam(learning_rate=1e-5), #type: ignore
            loss='binary_crossentropy',
            metrics=['accuracy', AUC(name='auc')]
        )
        
        callbacks_list = self.get_callbacks(model_save_path)
        total_epochs = len(self.history.epoch) + fine_tune_epochs
        
        self.history_fine = self.model.fit(
            train_ds,
            validation_data=valid_ds,
            initial_epoch=self.history.epoch[-1] + 1,
            epochs=total_epochs,
            callbacks=callbacks_list,
            verbose="auto"
        )
        return self.history_fine
    
    def train_complete(self, train_ds, valid_ds, initial_epochs=10, 
                      fine_tune_epochs=10, model_save_path='best_resnet_model.h5'):
        """
        Complete training pipeline: initial training + fine-tuning.
        
        :param train_ds: Training data set (tf.data.Dataset)
        :param valid_ds: Validation data set (tf.data.Dataset)
        :param initial_epochs: Number of epochs for initial training
        :param fine_tune_epochs: Number of epochs for fine-tuning
        :param model_save_path: Path to save the best model
        :return: Tuple of (initial_history, fine_tune_history)
        """
        # Phase 1: Train new layers
        history1 = self.train_initial_phase(train_ds, valid_ds, initial_epochs, model_save_path)
        
        # Phase 2: Fine-tune entire model
        history2 = self.train_fine_tune(train_ds, valid_ds, fine_tune_epochs, model_save_path)
        
        return history1, history2
    
