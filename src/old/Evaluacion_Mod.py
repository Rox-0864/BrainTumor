# %% Evaluación del modelo
def evaluate_model(model, test_gen):
    # Reiniciar el generador de prueba para asegurar que empieza desde el principio
    test_gen.reset()
    # Evaluar el modelo en el conjunto de prueba
    loss, accuracy, auc_score = model.evaluate(test_gen, verbose=0)

    print(f"\nResultados en conjunto de prueba:")
    print(f"Pérdida (Loss): {loss:.4f}")
    print(f"Exactitud (Accuracy): {accuracy:.4f}")
    print(f"AUC: {auc_score:.4f}")

    # Obtener las probabilidades predichas por el modelo para el conjunto de prueba
    y_pred_probs = model.predict(test_gen)
    # Convertir las probabilidades a predicciones de clase (0 o 1) usando un umbral de 0.5
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    # Obtener las etiquetas verdaderas del generador de prueba
    y_true = test_gen.classes

    # Obtener los nombres originales de las clases usando el label_encoder ajustado previamente
    class_names = list(label_encoder.inverse_transform([0, 1]))

    # --- Matriz de Confusión ---
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Matriz de Confusión - ResNet50')
    plt.ylabel('Etiqueta Verdadera')
    plt.xlabel('Etiqueta Predicha')
    plt.show()

    # --- Reporte de Clasificación ---
    print("\nReporte de Clasificación:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    # --- Curva ROC ---
    # Calcular la tasa de falsos positivos (fpr) y la tasa de verdaderos positivos (tpr)
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs) # Usar y_pred_probs para la curva ROC
    # Calcular el Área Bajo la Curva ROC
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'Curva ROC (área = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--') # Línea de no discriminación
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC - ResNet50')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

# Llamar a la función de evaluación con el modelo entrenado y el generador de prueba
evaluate_model(model, test_gen)

# %% Visualización del historial de entrenamiento
def plot_combined_history(initial_hist, fine_tune_hist):
    # Extraer métricas de la fase inicial
    acc = initial_hist.history['accuracy']
    val_acc = initial_hist.history['val_accuracy']
    loss = initial_hist.history['loss']
    val_loss = initial_hist.history['val_loss']

    # Añadir métricas de la fase de fine-tuning
    acc += fine_tune_hist.history['accuracy']
    val_acc += fine_tune_hist.history['val_accuracy']
    loss += fine_tune_hist.history['loss']
    val_loss += fine_tune_hist.history['val_loss']

    plt.figure(figsize=(12, 6))

    # Subgráfico para la Exactitud
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Exactitud de Entrenamiento')
    plt.plot(val_acc, label='Exactitud de Validación')
    # Línea vertical para marcar el inicio del fine-tuning
    # Asumiendo que initial_epochs fue definido como en el script original
    plt.plot([initial_epochs-1, initial_epochs-1],
             plt.ylim(), label='Inicio Fine-Tuning', linestyle='--')
    plt.title('Exactitud durante Entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Exactitud')
    plt.legend()
    plt.grid(True)

    # Subgráfico para la Pérdida
    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Pérdida de Entrenamiento')
    plt.plot(val_loss, label='Pérdida de Validación')
    # Línea vertical para marcar el inicio del fine-tuning
    plt.plot([initial_epochs-1, initial_epochs-1],
             plt.ylim(), label='Inicio Fine-Tuning', linestyle='--')
    plt.title('Pérdida durante Entrenamiento')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    plt.grid(True)

    plt.tight_layout() # Ajustar el layout para evitar superposiciones
    plt.show()

# Llamar a la función para graficar el historial combinado
plot_combined_history(history, history_fine)