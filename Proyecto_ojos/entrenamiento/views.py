from django.shortcuts import render
import os
from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import dlib
import numpy as np
from django.shortcuts import render
from django.http import JsonResponse
import time
import cv2
from datetime import datetime


# Directorio base para almacenar las imágenes


# Inicializa un arreglo para rastrear el estado de los botones
pulsado = [0, 0, 0, 0, 0, 0, 0, 0, 0]

# Directorio donde se almacenarán las imágenes capturadas


# Modelo de detección de caras
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_68.dat')

def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

directorio_base = 'entrenamiento/'
# Método de vista para clasificar ojos
@csrf_exempt
def clasificar(request):
    global directorio_base

    # Obtener el número de panel desde la solicitud
    numero_seleccionado = request.POST['panel_numero']

    # Directorio de imágenes capturadas desde el primer código
    directorio_imagenes = f'entrenamiento/fotos/panel{numero_seleccionado}/'

    # Directorio donde se guardarán los ojos normalizados
    directorio_ojos = f'entrenamiento/fotos/fotosnormalizadas/panel{numero_seleccionado}/'

    try:
        # Verificar si el directorio de ojos normalizados existe, si no, créalo
        if not os.path.exists(directorio_ojos):
            os.makedirs(directorio_ojos)

        # Listar archivos en el directorio de imágenes capturadas
        archivos = os.listdir(directorio_imagenes)
        for archivo in archivos:
            if archivo.endswith('.jpg'):
                img_path = os.path.join(directorio_imagenes, archivo)

                # Leer la imagen desde el archivo
                img = cv2.imread(img_path)

                # Verificar si la imagen se cargó correctamente
                if img is not None:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                    # Detectar caras en la imagen
                    rects = detector(gray)
                    if len(rects) > 0:
                        for rect in rects:
                            # Obtener landmarks faciales
                            shape = predictor(gray, rect)

                            # Verificar que se detectaron caras
                            if len(shape.parts()) >= 68:
                                # Recortar y guardar los ojos izquierdo y derecho
                             if img is not None:
                                for i, (ojo_x, ojo_y) in enumerate([(36, 37), (42, 43)]):
                                    x1, y1 = shape.part(ojo_x).x, shape.part(ojo_x).y
                                    x2, y2 = shape.part(ojo_y).x, shape.part(ojo_y).y

                                    # Verifica que las coordenadas sean válidas
                                    if x1 >= 0 and y1 >= 0 and x2 >= 0 and y2 >= 0:
                                        # Realiza el recorte
                                        ojo_recortado = img[y1:y2, x1:x2]

                                        # Verifica que el recorte no sea una matriz NumPy vacía (size > 0)
                                        
                                        ruta_ojo = f'{directorio_ojos}ojo{i}_panel{numero_seleccionado}_{datetime.today().strftime("%Y%m%d%H%M%S")}.jpg'
                                        cv2.imwrite(ruta_ojo, ojo_recortado)
                                      
                                    else:
                                        # Manejar el caso de coordenadas negativas
                                        print(f"Coordenadas negativas para el ojo {i}")
                             else:
                                    # Manejar el caso en el que 'img' no es una imagen válida
                                    print("La imagen no es válida")
                            else:
                                # Manejar el caso en el que no se detectaron caras en la imagen
                                print("No se detectaron caras en la imagen")
                    else:
                        # Manejar el caso en el que no se detectaron caras en la imagen
                        print("No se detectaron caras en la imagen")
                else:
                    # Manejar el caso en el que la imagen no se cargó correctamente
                    print(f"No se pudo cargar la imagen: {img_path}")
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'message': 'Ojos clasificados y guardados correctamente'})


def entrenador_views(request):
    return render(request, 'entrenador.html')

def crea_directorios():
    try:
        os.stat(directorio_base)
    except:
        os.mkdir(directorio_base)

    dirojos = ['ojoder', 'ojoizq']
    for dir in dirojos:
        try:
            os.stat(directorio_base + dir + os.sep)
        except:
            os.mkdir(directorio_base + dir + os.sep)
        for etiq in range(9):
            try:
                os.stat(directorio_base + dir + os.sep + dir + str(etiq))
            except:
                os.mkdir(directorio_base + dir + os.sep + dir + str(etiq))
def crear_directorios_view(request):
    crea_directorios()
    return HttpResponse("Directorios creados con éxito")


@csrf_exempt
def guardar_imagenes(request):
    if request.method == 'POST':
        try:
            # Accede al parámetro "panel_numero" desde el cuerpo de la solicitud POST
            panel_numero = request.POST.get('panel_numero')

            # Asegúrate de que la carpeta "fotos" existe en la aplicación "entrenamiento"
            fotos_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fotos')

            # Crea una subcarpeta dentro de "fotos" con el nombre "panel+numero"
            panel_dir = os.path.join(fotos_dir, f'panel{panel_numero}')
            if not os.path.exists(panel_dir):
                os.makedirs(panel_dir)

            # Guarda las imágenes en la subcarpeta correspondiente
            for key, file in request.FILES.items():
                with open(os.path.join(panel_dir, file.name), 'wb') as destination:
                    for chunk in file.chunks():
                        destination.write(chunk)

            return JsonResponse({'message': f'Imágenes guardadas correctamente en la carpeta panel{panel_numero}'})
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

    return JsonResponse({'error': 'Método no permitido'}, status=405)





def shape_to_np(shape, dtype="int"):
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords

def ajustar_gamma(imagen, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(imagen, table)

def entrena(request):
    global directorio_base

    # Obtener el número de panel desde la solicitud
    numero_seleccionado = request.POST['panel_numero']

    # Directorio de imágenes
    directorio_imagenes = f'{directorio_base}fotos/panel{numero_seleccionado}/'

    # Inicializar el detector de caras de Dlib
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # Lista para almacenar las imágenes procesadas
    imagenes_capturadas = []

    try:
        # Listar archivos en el directorio de imágenes
        archivos = os.listdir(directorio_imagenes)
        for archivo in archivos:
            if archivo.endswith('.jpg'):
                img_path = os.path.join(directorio_imagenes, archivo)

                # Leer la imagen desde el archivo
                img = cv2.imread(img_path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                # Detectar caras en la imagen
                rects = detector(gray)
                if len(rects) > 0:
                    for rect in rects:
                        # Obtener landmarks faciales
                        shape = predictor(gray, rect)

                        # Procesa la imagen o extrae información de interés
                        # Aquí puedes realizar tareas como recortar las caras, extraer características, etc.

                        # Por ejemplo, recortar y guardar la cara detectada
                        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
                        cara_recortada = img[y:y+h, x:x+w]
                        ruta_cara_recortada = f'{directorio_base}caras/panel{numero_seleccionado}/cara_{datetime.today().strftime("%Y%m%d%H%M%S")}.jpg'
                        cv2.imwrite(ruta_cara_recortada, cara_recortada)

                        # Agregar la ruta de la cara recortada a la lista de imágenes procesadas
                        imagenes_capturadas.append(ruta_cara_recortada)
    
    except Exception as e:
        print("Error:", str(e))
    
    return JsonResponse({'imagenes_capturadas': imagenes_capturadas})

@csrf_exempt
def tomar_fotos(request):
    if request.method == 'POST':
        try:
            # Simula la captura de imágenes desde la cámara
            # Agrega aquí tu lógica de captura de imágenes y detección de caras
            # Ejemplo: Creación de una lista de rutas de imágenes simuladas
            imagenes_capturadas = []
            for i in range(3):
                etiqueta = f'panel{request.POST["panel_numero"]}'
                img_path = f'{directorio_base}fotos/{etiqueta}/{etiqueta}_{i}_{datetime.today().strftime("%Y%m%d%H%M%S")}.jpg'
                imagenes_capturadas.append(img_path)
            
            # Simula la detección de caras
            deteccion_exitosa = True  # Cambia esto según el resultado de la detección

            if deteccion_exitosa:
                mensaje = 'Se detectó una cara en todas las imágenes capturadas.'
            else:
                mensaje = 'Error en la detección de caras.'

            resultado = {'deteccion_exitosa': deteccion_exitosa, 'mensaje': mensaje}

        except Exception as e:
            resultado = {'deteccion_exitosa': False, 'mensaje': str(e)}

        return JsonResponse(resultado)
def entrenador(request):
    return render(request, 'entrenador.html')
