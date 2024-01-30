import cv2
import numpy as np
import os
from django.http import JsonResponse


def consultarKNN(request, panel_numero):
    ruta_derecho = os.path.abspath('knn_derecho.xml')
    ruta_izquierdo = os.path.abspath('knn_izquierdo.xml')
    try:
        knn_derecho = cv2.ml.KNearest_load(ruta_derecho)
    except Exception as e:
        print("Error al cargar el modelo derecho:", e)

    try:
        knn_izquierdo = cv2.ml.KNearest_load(ruta_izquierdo)
    except Exception as e:
        print("Error al cargar el modelo izquierdo:", e)
    # Construir la ruta de las imágenes basada en el panel_numero
    ruta_ojo_derecho = os.path.join('entrenamiento', 'fotos', 'fotosusuarios', 'fotosnormalizadas', f'panel{panel_numero}', f'ojoder_panel{panel_numero}_foto.jpg')
    ruta_ojo_izquierdo = os.path.join('entrenamiento', 'fotos', 'fotosusuarios', 'fotosnormalizadas', f'panel{panel_numero}', f'ojoizq_panel{panel_numero}_foto.jpg')

    # Cargar las imágenes
    ojo_derecho = cv2.imread(ruta_ojo_derecho)
    ojo_izquierdo = cv2.imread(ruta_ojo_izquierdo)

    # Verificar que las imágenes se hayan cargado correctamente
    if ojo_derecho is None or ojo_izquierdo is None:
        return JsonResponse({'error': 'No se pudieron cargar las imágenes'}, status=500)
    pred_izquierdo = preparar_y_predicir(knn_izquierdo, ojo_izquierdo)
    pred_derecho = preparar_y_predicir(knn_derecho, ojo_derecho)

    # Convertir ndarray a lista o valor simple
    pred_izquierdo = pred_izquierdo.tolist() if isinstance(pred_izquierdo, np.ndarray) else pred_izquierdo
    pred_derecho = pred_derecho.tolist() if isinstance(pred_derecho, np.ndarray) else pred_derecho

    resultado = {"izquierdo": pred_izquierdo, "derecho": pred_derecho}
    return JsonResponse(resultado)

def preparar_y_predicir(knn_modelo, imagen_ojo):
    # Verificar que la imagen no sea None
    if imagen_ojo is None:
        raise ValueError("La imagen de entrada es None")

    gris = cv2.cvtColor(imagen_ojo, cv2.COLOR_BGR2GRAY)
    ecualizado = cv2.equalizeHist(gris)
    redimensionado = cv2.resize(ecualizado, (50, 50))  # Asegurarse de que este tamaño coincida con el del entrenamiento
    vector_1d = np.reshape(redimensionado, (1, -1)).astype(np.float32)

    # Guardar la imagen procesada con una extensión de archivo válida
    cv2.imwrite('../Proyecto_ojos/entrenamiento/fotos/fotohistograma.jpg', vector_1d)

    # Asegurarse de que las dimensiones coincidan con las del entrenamiento
    if vector_1d.shape[1] != 2500:
        raise ValueError(f"Dimensiones incorrectas: {vector_1d.shape}")

    _, resultado, _, _ = knn_modelo.findNearest(vector_1d, k=3)
    print(resultado)
    return resultado
