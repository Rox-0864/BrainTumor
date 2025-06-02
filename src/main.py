from src.core.load_data import LoadData
from src.core.data_preprocess import DataPreprocessor
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
        
        # Initialize data loader
        data_loader = LoadData(base_path=base_path, categories=categories)
        
        # Load and process data
        data_df = data_loader.load_images()
        
        # Print basic dataset information
        print(f"Total images: {len(data_df)}")
        print(f"Categories distribution:\n{data_df['label'].value_counts()}")
        
        # Processing pipeline
        # - Preprocess images

        preprocessor = DataPreprocessor(img_size=(224, 224), batch_size=32)
        train_gen, valid_gen, test_gen, label_encoder = preprocessor.process_data(data_df)
        print(train_gen)
        print(valid_gen)
        print(test_gen)
        print(label_encoder)






        # - Train model
        # - Evaluate results
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())