# %% Imports
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import cv2

# Librerías de Scikit-learn para preprocesamiento y métricas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc

# Imbalanced-learn para manejar desbalance de clases
from imblearn.over_sampling import RandomOverSampler

# Componentes de TensorFlow y Keras
import tensorflow as tf


from tensorflow.keras import layers, Model, callbacks
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50 # El modelo base que usaremos
from tensorflow.keras.applications.resnet import preprocess_input # Función de preprocesamiento específica para ResNet
from tensorflow.keras.metrics import AUC # Métrica de Área Bajo la Curva ROC

# Configuración para la visualización y logs de TensorFlow
sns.set_style("whitegrid") # Estilo para los gráficos de Seaborn
tf.get_logger().setLevel('ERROR') # Suprime mensajes informativos de TensorFlow, mostrando solo errores

# %% Carga de datos
base_path = "dataset"  # Directorio raíz donde se encuentran las carpetas de categorías
categories = ["Healthy", "Tumor"] # Nombres de las subcarpetas y nuestras clases

image_paths = []  # Lista para almacenar las rutas a cada imagen
labels = []       # Lista para almacenar la etiqueta de cada imagen

# Iterar sobre cada categoría definida
for category in categories:
    category_path = os.path.join(base_path, category) # Construir la ruta a la carpeta de la categoría
    if os.path.isdir(category_path):
        # Iterar sobre cada archivo de imagen dentro de la carpeta de la categoría
        for image_name in os.listdir(category_path):
            image_path = os.path.join(category_path, image_name) # Ruta completa a la imagen
            image_paths.append(image_path) # Añadir la ruta de la imagen a la lista
            labels.append(category)        # Añadir la etiqueta (nombre de la categoría) a la lista
    else:
        print(f"Advertencia: El directorio para la categoría '{category}' no fue encontrado en '{category_path}'")

# Crear un DataFrame de Pandas para almacenar las rutas de las imágenes y sus etiquetas
df = pd.DataFrame({"image_path": image_paths, "label": labels})

# Mostrar las primeras filas del DataFrame y la distribución de clases
print("DataFrame inicial con rutas de imágenes y etiquetas:")
print(df.head())
print("\nDistribución de clases inicial:")
print(df['label'].value_counts())


# %% Preprocesamiento
# Codificación de etiquetas
label_encoder = LabelEncoder()
# Se crea una nueva columna 'category_encoded' con las etiquetas numéricas (ej. 0 para Healthy, 1 para Tumor)
df['category_encoded'] = label_encoder.fit_transform(df['label'])

print("DataFrame después de la codificación de etiquetas:")
print(df.head())
print(f"Clases codificadas: {label_encoder.classes_} -> {label_encoder.transform(label_encoder.classes_)}")

# Primero, dividir en entrenamiento (80%) y un conjunto temporal (20% para validación + prueba)
X_train_original, X_temp, y_train_original, y_temp = train_test_split(
    df[['image_path']],  # Características (rutas de imagen como DataFrame)
    df['category_encoded'], # Etiquetas numéricas
    train_size=0.8,         # 80% para entrenamiento
    shuffle=True,           # Mezclar los datos antes de dividir
    random_state=42,        # Para reproducibilidad
    stratify=df['category_encoded'] # Asegurar proporciones de clase similares en la división
)

# Luego, dividir el conjunto temporal en validación (50% de temp -> 10% del total)
X_valid, X_test, y_valid, y_test = train_test_split(
    X_temp,                 # DataFrame con rutas de imagen del conjunto temporal
    y_temp,                 # Etiquetas del conjunto temporal
    test_size=0.5,          # 50% de X_temp para el conjunto de prueba (el resto para validación)
    shuffle=True,
    random_state=42,
    stratify=y_temp         # Estratificar sobre las etiquetas del conjunto temporal
)
#Sobremuestro
ros = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_original, y_train_original)

# train_df utilizará los datos de entrenamiento sobremuestreados.
train_df = pd.DataFrame(X_train_resampled, columns=['image_path'])
train_df['category_encoded'] = y_train_resampled.astype(str)

# valid_df y test_df utilizan los datos originales de validación y prueba, sin sobremuestreo.
valid_df = pd.DataFrame(X_valid, columns=['image_path'])
valid_df['category_encoded'] = y_valid.astype(str)

test_df = pd.DataFrame(X_test, columns=['image_path'])
test_df['category_encoded'] = y_test.astype(str)

# %% Generadores de datos para ResNet
batch_size = 32
img_size = (224, 224)  # ResNet requiere 224x224

# Data augmentation para entrenamiento
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,  # Preprocesamiento específico de ResNet
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

import tensorflow as tf

print("Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))