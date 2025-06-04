from src.core.load_data import LoadData
from src.core.preprocess import DataPreprocessor
from src.core.training import ModelTrainer
from pathlib import Path
import os

def main():
    """
    Main function to orchestrate the brain tumor analysis pipeline.
    Loads and processes brain tumor image data for analysis.
    """
    try:
        # Define paths and categories
        base_path = os.path.join(Path(__file__).parents[1], "dataset")
        categories = ["Healthy", "Tumor"]
        
        print("=== Brain Tumor Detection Pipeline ===")
        
        # Step 1: Load data
        print("\n1. Loading data...")
        data_loader = LoadData(base_path=base_path, categories=categories)
        data_df = data_loader.load_images()
        
        # Print basic dataset information
        print(f"Total images: {len(data_df)}")
        print(f"Categories distribution:\n{data_df['label'].value_counts()}")
        
        # Step 2: Preprocess data
        print("\n2. Preprocessing data...")
        preprocessor = DataPreprocessor(img_size=(224, 224), batch_size=12)
        train_ds, valid_ds, test_ds, label_encoder = preprocessor.process_data(data_df)
        
        print(f"Training batches: {len(train_ds)}")
        print(f"Validation batches: {len(valid_ds)}")
        print(f"Test batches: {len(test_ds)}")
        print(f"Label encoding: {dict(enumerate(label_encoder.classes_))}")
        
        # Step 3: Build and train model
        print("\n3. Building and training model...")
        trainer = ModelTrainer(input_shape=(224, 224, 3))
        
        # Build model
        model = trainer.build_resnet_model()
        print(f"Model built with {model.count_params():,} parameters")
        
        # Compile model
        trainer.compile_model(learning_rate=1e-3)
        print("Model compiled successfully")
        
        # Train model (complete pipeline)
        print("\nStarting training...")
        history1, history2 = trainer.train_complete(
            train_ds=train_ds,
            valid_ds=valid_ds,
            initial_epochs=10,
            fine_tune_epochs=10,
            model_save_path='models/best_brain_tumor_model.h5'
        )
        
        print("\nTraining completed!")
        
        # Step 4: Plot training results
        print("\n4. Plotting training history...")
        trainer.plot_training_history()
        
        # Step 5: Evaluate on test set (optional)
        print("\n5. Evaluating on test set...")
        test_results = trainer.model.evaluate(test_ds, verbose=1)
        print(f"Test Loss: {test_results[0]:.4f}")
        print(f"Test Accuracy: {test_results[1]:.4f}")
        print(f"Test AUC: {test_results[2]:.4f}")
        
        print("\n=== Pipeline completed successfully! ===")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())