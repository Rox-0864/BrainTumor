# Clasificación de Imágenes Médicas con ResNet50 🧠🔬

Este repositorio documenta un proyecto para la clasificación de imágenes médicas, distinguiendo entre imágenes "Healthy" y con "Tumor". Se utiliza un modelo de Deep Learning basado en la arquitectura ResNet50, implementado con TensorFlow y Keras. El flujo de trabajo sigue una estructura clara desde la preparación de los datos hasta la evaluación del modelo.

Las imagenes utilizadas para el entrenamiento se encuentra en el siguiente repositorio de Kaggle:
https://www.kaggle.com/code/nirmalgaud/brain-tumor-classification-with-fibonaccinet/input

Si encuentran problemas con la descarga, pueden buscar las imagenes en mi Drive:
https://drive.google.com/drive/folders/1eygT604e48FLQJbU-l2nujjBblmhAoja

Pueden visualizar como funciona nuestro clasificador en la siguiente página de Streamlit:<br>
https://braintumordetection-1.streamlit.app/

## 🚀 Instrucciones para el equipo

### 1. Clona el repositorio

```bash
git clone https://github.com/tu_usuario/Brain_Tumor3.git
cd Brain_Tumor3
```

### 2. Crea y activa un entorno virtual

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

### 3. Instala las dependencias

```bash
pip install -r requirements.txt
```

---

## ☁️ Subir el dataset a Hugging Face (solo Rosela)

1. Descomprime el archivo `.zip` del dataset en `Brain_Tumor3/data/` con la siguiente estructura:

```
data/
├── glioma/
├── meningioma/
```

2. Ejecuta el script:

```bash
python scripts/upload_to_huggingface.py
```

Esto lo subirá al repositorio de Hugging Face: `Rosela/brain_tumor_dataset`

---

## 📥 Descargar el dataset desde cualquier entorno

Todos los integrantes del equipo pueden usar el siguiente script:

```bash
python scripts/download_dataset.py
```

Este descargará y reconstruirá la estructura del dataset localmente en `data/`.

---
 
## 🚀 Estructura del Proyecto por Bloques

El proyecto se organiza en cuatro bloques funcionales principales:

### Bloque 1: **Configuración Inicial y Carga de Datos** 📂
Esta fase establece el entorno básico y carga el conjunto de datos de imágenes.
- Se preparan las herramientas necesarias, siendo **TensorFlow** el framework principal para el modelado.
- Las imágenes se cargan desde un sistema de archivos estructurado, donde las rutas y sus correspondientes etiquetas (ej. "Healthy", "Tumor") se organizan en un formato manejable, típicamente un DataFrame de Pandas.

### Bloque 2: **Preprocesamiento y Generadores de Imágenes** 🖼️➡️🔢
Antes del entrenamiento, los datos de imágenes requieren una preparación significativa:
- **Codificación de Etiquetas**: Las etiquetas textuales de las clases se convierten a un formato numérico.
- **Manejo de Desbalance**: Se aplica sobremuestreo (ej. `RandomOverSampler`) **exclusivamente al conjunto de entrenamiento** para asegurar un aprendizaje equilibrado.
- **División de Datos**: El dataset se divide en conjuntos de entrenamiento, validación y prueba.
- **Generadores de Datos (`ImageDataGenerator`)**: Se configuran generadores para alimentar eficientemente al modelo:
    - Para el entrenamiento, se aplica **aumentación de datos** (rotaciones, zoom, etc.) y el preprocesamiento específico de ResNet.
    - Para validación y prueba, solo se aplica el preprocesamiento de ResNet, sin aumentación, para una evaluación objetiva.

### Bloque 3: **Construcción y Entrenamiento del Modelo ResNet50** 🧠🔧
Aquí se define la arquitectura del modelo de Transfer Learning y se lleva a cabo el proceso de entrenamiento:
- **Modelo Base ResNet50**: Se carga la arquitectura ResNet50 pre-entrenada, sin su capa clasificadora original.
- **Capas Personalizadas**: Se añaden capas superiores (ej. GlobalAveragePooling, Dropout, Dense con activación sigmoide) para adaptar el modelo a la tarea de clasificación binaria.
- **Entrenamiento en Dos Fases**:
    1.  **Entrenamiento del Clasificador**: Inicialmente, solo se entrenan las capas personalizadas nuevas, manteniendo congelado el modelo base ResNet50.
    2.  **Fine-Tuning**: Posteriormente, se descongela el modelo base ResNet50 (o parte de él) y se continúa el entrenamiento de todo el modelo con una tasa de aprendizaje más baja para un ajuste fino.
- **Callbacks**: Se utilizan `EarlyStopping`, `ModelCheckpoint` y `ReduceLROnPlateau` para gestionar el entrenamiento, guardar el mejor modelo y ajustar la tasa de aprendizaje dinámicamente.

### Bloque 4: **Evaluación del Modelo y Visualización de Resultados** 📊📈
Finalmente, se evalúa el rendimiento del modelo entrenado utilizando el conjunto de prueba:
- Se calculan métricas clave como Pérdida, Exactitud y AUC.
- Se generan visualizaciones para interpretar el rendimiento:
    - **Matriz de Confusión**.
    - **Reporte de Clasificación** (precisión, recall, F1-score).
    - **Curva ROC**.
- Se grafica el **historial de entrenamiento** (pérdida y exactitud a lo largo de las épocas) para analizar el proceso de aprendizaje.

---

## 🛠️ Cómo Usar Este Repositorio

1.  **Clona el repositorio**:
    ```bash
    git clone https://github.com/BootcampXperience/ML_Brain_Tumor_Detection.git
    cd ML_Brain_Tumor_Detection
    ```

2.  **Configura tu Entorno**:
    *   Asegúrate de tener Python y las librerías necesarias instaladas (principalmente TensorFlow, Keras, Scikit-learn, Pandas, Imbalanced-learn, Matplotlib). Se recomienda usar un entorno virtual.

3.  **Prepara tus Datos**:
    *   Organiza tus imágenes en una carpeta `images` (o según se especifique en el script) con subcarpetas por categoría (ej. `images/Healthy/`, `images/Tumor/`).

4.  **Ejecuta el Script Principal**:
    *   Revisa y ajusta las configuraciones en el script de Python si es necesario.
    *   Ejecuta el script:
        ```bash
        python3 Brain_Tumor_ResNet.py
        ```

5.  **Revisa los Resultados**:
    *   El script guardará el mejor modelo y mostrará/guardará gráficos de evaluación y métricas.

---

## 🔬 Tecnologías Clave

*   **Python**
*   **TensorFlow / Keras**
*   **ResNet50 (Transfer Learning)**
*   **Pandas, NumPy, Scikit-learn, Imbalanced-learn, Matplotlib**

## 🤝 Buenas prácticas para colaboración

- Usa ramas por funcionalidad: `git checkout -b nombre_rama`
- Haz commits descriptivos: `git commit -m "Agregué función para evaluación"`
- Sincroniza con `git pull origin main` antes de hacer `push`
- Documenta tu código y actualiza el README si haces cambios importantes

---

## 📬 Contacto

Para dudas o colaboración, escribe en el canal del equipo o crea un Issue en el repo.
