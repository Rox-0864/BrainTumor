import os
import pandas as pd

class LoadData:
    """
    Class to load image data from a structured directory.
    This class searches for images in categorized subdirectories
    and creates a DataFrame with the image paths and their corresponding labels.
    """
    
    def __init__(self, base_path, categories):
        """
        :param base_path: Path to the root directory where category folders are located.
        :param categories: List of subfolder names representing the classes.
        """
        self.base_path = base_path
        self.categories = categories
        self.image_paths = []
        self.labels = []
    
    def load_images(self):
        """
        Loads image paths and their labels from the structured directory.
        
        :return: DataFrame with image paths and labels.
        """
        for category in self.categories:
            category_path = os.path.join(self.base_path, category)
            if os.path.isdir(category_path):
                for image_name in os.listdir(category_path):
                    image_path = os.path.join(category_path, image_name)
                    self.image_paths.append(image_path)
                    self.labels.append(category)
            else:
                print(f"Warning: Directory for category '{category}' was not found in '{category_path}'")
        
        df = pd.DataFrame({"image_path": self.image_paths, "label": self.labels})
        return df

