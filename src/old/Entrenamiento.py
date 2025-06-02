from keras.applications.resnet50 import ResNet50

# %% Construcción del modelo ResNet50
def build_resnet_model(input_shape=(224, 224, 3)):
    # Cargar ResNet50 preentrenada (sin las capas superiores de clasificación de ImageNet)
    base_model = ResNet50(
        include_top=False,      # No incluir la capa densa final de ResNet50
        weights='imagenet',     # Usar pesos pre-entrenados en ImageNet
        input_shape=input_shape # Definir la forma de entrada de las imágenes
    )

    # Congelar las capas del modelo base inicialmente.
    # Sus pesos no se actualizarán durante la primera fase de entrenamiento.
    base_model.trainable = False

    # Construir el modelo completo añadiendo nuestras propias capas encima de ResNet50
    inputs = layers.Input(shape=input_shape) # Capa de entrada
    # Pasar las entradas a través del modelo base.
    x = base_model(inputs, training=False)
    # Reducir la dimensionalidad espacial a un vector por cada mapa de características.
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)  # Capa de Dropout para regularización
    # Capa densa final para la clasificación binaria, con activación sigmoide.
    # Se añade regularización L2 al kernel para prevenir el sobreajuste.
    outputs = layers.Dense(1, activation='sigmoid',
                          kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

    # Crear el modelo final especificando las entradas y salidas.
    model = Model(inputs, outputs)
    return model

# Instanciar el modelo
model = build_resnet_model()

# Compilación del modelo
model.compile(
    optimizer=Adam(learning_rate=1e-3), # Optimizador Adam con una tasa de aprendizaje inicial
    loss='binary_crossentropy',         # Función de pérdida para clasificación binaria
    metrics=['accuracy', AUC(name='auc')] # Métricas a monitorear
)
# Callbacks para el entrenamiento
callbacks_list = [
    callbacks.EarlyStopping(
        monitor='val_auc',        # Métrica a monitorear (AUC en el conjunto de validación)
        patience=5,               # Número de épocas a esperar sin mejora antes de detener
        mode='max',               # Indica que buscamos maximizar la métrica (AUC)
        restore_best_weights=True,# Restaura los pesos del modelo de la mejor época al finalizar
        verbose=1                 # Muestra mensajes cuando el callback se activa
    ),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss',       # Métrica a monitorear (pérdida en el conjunto de validación)
        factor=0.1,               # Factor por el cual se reduce la tasa de aprendizaje (new_lr = lr * factor)
        patience=3,               # Número de épocas a esperar sin mejora antes de reducir LR
        verbose=1
    ),
    callbacks.ModelCheckpoint(
        'best_resnet_model.h5',   # Nombre del archivo para guardar el mejor modelo
        monitor='val_auc',        # Métrica que determina si el modelo es "mejor"
        save_best_only=True,      # Guarda solo el modelo si la métrica monitoreada ha mejorado
        mode='max',               # El objetivo es maximizar val_auc
        verbose=1
    )
]
# Fase 1: Entrenar solo las capas nuevas
print("\nEntrenando capas nuevas...")
initial_epochs = 10 # Número de épocas para la primera fase de entrenamiento
history = model.fit(
    train_gen,
    validation_data=valid_gen,
    epochs=initial_epochs,
    callbacks=callbacks_list, # Utilizar la lista de callbacks definida
    verbose=1                 # Mostrar barra de progreso e información por época
)

# Fase 2: Fine-tuning de todo el modelo
print("\nFine-tuning de todo el modelo...")
# Acceder a la capa base ResNet50 dentro del modelo 'model'
base_model_layer_from_model = model.layers[1] # Obtenemos la referencia a la capa ResNet50
base_model_layer_from_model.trainable = True   # Hacemos que la capa ResNet50 sea entrenable

# Recompilar el modelo con una tasa de aprendizaje mucho más baja para el fine-tuning
# Esto es esencial para no destruir los pesos pre-entrenados de ResNet50.
model.compile(
    optimizer=Adam(learning_rate=1e-5), # Tasa de aprendizaje significativamente menor
    loss='binary_crossentropy',
    metrics=['accuracy', AUC(name='auc')]
)

fine_tune_epochs = 10 # Número de épocas adicionales para el fine-tuning
total_epochs = initial_epochs + fine_tune_epochs # Número total de épocas de entrenamiento

# Continuar el entrenamiento (fine-tuning)
history_fine = model.fit(
    train_gen,
    validation_data=valid_gen,
    initial_epoch=history.epoch[-1]+1, # Comenzar el conteo de épocas desde el final de la fase anterior
    epochs=total_epochs,            # Entrenar hasta alcanzar el número total de épocas
    callbacks=callbacks_list,
    verbose=1
)