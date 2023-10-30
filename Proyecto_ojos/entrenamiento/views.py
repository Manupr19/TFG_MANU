from django.shortcuts import render
import os
from django.http import HttpResponse
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import dlib
import numpy as np
from django.shortcuts import render
import time
import tkinter as tk
import cv2
from PIL import Image, ImageTk
from datetime import datetime

# Inicializa un arreglo para rastrear el estado de los botones
pulsado = [0, 0, 0, 0, 0, 0, 0, 0, 0]
# Modelo de detección de caras
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_68.dat')
vecinos=3

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
                         # Ajusto los puntos de inicio y fin para una región mucho más grande
                                x1_left, y1_left = shape.part(36).x - 20, shape.part(37).y - 20
                                x2_left, y2_left = shape.part(39).x + 20, shape.part(41).y + 20
                                x1_right, y1_right = shape.part(42).x - 20, shape.part(43).y - 20
                                x2_right, y2_right = shape.part(45).x + 20, shape.part(47).y + 20
                                # Verifica que las coordenadas sean válidas
                                if x1_left >= 0 and y1_left >= 0 and x2_left >= 0 and y2_left >= 0:
                                    # Realiza el recorte y guarda el ojo izquierdo
                                    ojo_recortado1 = img[y1_left:y2_left, x1_left:x2_left]
                                    ruta_ojo_izq = f'{directorio_ojos}ojoizq_{archivo}'
                                    cv2.imwrite(ruta_ojo_izq, ojo_recortado1)

                                if x1_right >= 0 and y1_right >= 0 and x2_right >= 0 and y2_right >= 0:
                                    # Realiza el recorte y guarda el ojo derecho
                                    ojo_recortado2 = img[y1_right:y2_right, x1_right:x2_right]
                                    ruta_ojo_der = f'{directorio_ojos}ojoder_{archivo}'
                                    cv2.imwrite(ruta_ojo_der, ojo_recortado2)
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


def arrayTOimgtk(lectura):
    b, g, r = cv2.split(lectura)
    img = cv2.merge((r, g, b))
    im = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=im)
    return imgtk

root = tk.Tk()

ALTO = root.winfo_screenheight() - 50
ANCHO = root.winfo_screenwidth()
ANCHO = 1900  # Lo pongo a mano por mis dos monitores
root.config(width=ANCHO, height=ALTO)
continua = 1
root.title("Interfaz de Entrenamiento")
# Creamos los marcos de ojos
marconegroa = np.zeros((40, 70, 3), dtype=np.uint8)
marconegrotk = arrayTOimgtk(marconegroa)

marcos_ojos = []
for i in range(9):
    marcos_ojos.append([tk.Label(root, image=marconegrotk), tk.Label(root, image=marconegrotk)])


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

def entrenar_knn_para_todos_los_ojos(directorio_base, ojo, vecinos):
    # Lista de etiquetas para las clases de ojos (por ejemplo, 'ojoizq0' hasta 'ojoizq2')
    clases = [f'{ojo}{i}' for i in range(3)]  # Tres imágenes por ojo

    etiqueta = 0

    training_data = []  # Almacenará los datos de entrenamiento (imágenes ecualizadas)
    training_labels = []  # Almacenará las etiquetas correspondientes

    for clase in clases:
        # Ruta al directorio que contiene las imágenes de una clase específica
        input_images_path = os.path.join(directorio_base, ojo, clase)
        files_names = os.listdir(input_images_path)

        for fichero in files_names:
            # Ruta de la imagen actual
            fichpath = os.path.join(input_images_path, fichero)
            print(fichpath)

            # Leer la imagen
            img = cv2.imread(fichpath)

            # Ecualizar el histograma de la imagen en escala de grises
            img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            img_eq = cv2.equalizeHist(img_gray)

            # Redimensionar la imagen a un arreglo 1D
            img_eq_1d = np.reshape(img_eq, (1, -1))

            # Agregar la imagen ecualizada y su etiqueta al conjunto de datos de entrenamiento
            training_data.append(img_eq_1d)
            training_labels.append(etiqueta)

        etiqueta += 1

    # Convertir los datos y etiquetas en arreglos NumPy
    training_data = np.array(training_data, dtype=np.float32)
    training_labels = np.array(training_labels, dtype=np.float32)

    print(training_data.shape)
    
    # Crear el clasificador KNN
    knn = cv2.ml.KNearest_create()

    # Entrenar el clasificador KNN con los datos de entrenamiento
    knn.train(training_data, cv2.ml.ROW_SAMPLE, training_labels)

    return knn


