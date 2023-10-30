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
import tkinter as tk
import cv2
from PIL import Image, ImageTk
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
import os
import cv2
from django.http import JsonResponse
from datetime import datetime

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

def entrena():
    global izquierda_sup
    global derecha_sup
    global derecha_inf
    global izquierda_inf
    global centro

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_68.dat')

    # left = [36, 37, 38, 39, 40, 41]
    # right = [42, 43, 44, 45, 46, 47]

    cap = cv2.VideoCapture(0)
    ret, img = cap.read()
    imgtk = arrayTOimgtk(cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2))))
    marco_camara = tk.Label(root, image=imgtk)
    marco_camara.place(x=ANCHO / 2 - int(np.shape(img)[1] * 0.6) / 2, y=140)

    try:
        while continua == 1:
            ret, img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            rects = detector(gray, 1)
            if len(rects) == 0:
                # No HAY CARAS
                cv2.circle(img, (11, 11), 10, (0, 0, 255), -1)
                root.configure(bg='grey')
            else:
                cv2.circle(img, (11, 11), 10, (0, 255, 0), -1)
                root.configure(bg='lightgrey')
                for rect in rects:
                    shape = predictor(gray, rect)
                    shape = shape_to_np(shape)
                    # Estraigo coordenadas ojo izquierdo
                    xmin = shape[37][0]
                    xmax = shape[40][0]
                    ymax = max(shape[38][1], shape[39][1], shape[41][1], shape[42][1])
                    ymin = min(shape[38][1], shape[39][1], shape[41][1], shape[42][1])

                    # Creamos imagenes del ojo izquierdo
                    ojoizq = img[ymin - 10:ymax + 10, xmin - 20:xmax + 20]
                    ojoizq = cv2.resize(ojoizq, (70, 40), interpolation=cv2.INTER_AREA)

                    # Estraigo coordenadas ojo derecho
                    xmin = shape[42][0]
                    xmax = shape[45][0]
                    ymax = max(shape[43][1], shape[44][1], shape[46][1], shape[47][1])
                    ymin = min(shape[43][1], shape[44][1], shape[46][1], shape[47][1])

                    # Creamos imagenes del ojo derecho
                    ojoder = img[ymin - 10:ymax + 10, xmin - 20:xmax + 20]
                    ojoder = cv2.resize(ojoder, (70, 40), interpolation=cv2.INTER_AREA)

                    # Normalizamos imagen
                    norm_img = np.zeros((70, 40))
                    ojoizqnorm = cv2.normalize(ojoizq, norm_img, 0, 255, cv2.NORM_MINMAX)
                    ojodernorm = cv2.normalize(ojoder, norm_img, 0, 255, cv2.NORM_MINMAX)

                    # Visualizamos los ojos
                    ojoizqnormtk = arrayTOimgtk(ojoizqnorm)
                    ojodernormtk = arrayTOimgtk(ojodernorm)
                    for k in range(9):
                        marcos_ojos[k][0].config(image=ojoizqnormtk)
                        marcos_ojos[k][1].config(image=ojodernormtk)

                    # Guardamos la seleccion
                    for k in range(9):
                        if pulsado[k] == 1:
                            cv2.imwrite(
                                directorio_base + 'ojoder' + os.sep + 'ojoder' + str(k) + os.sep + 'ojoder' + str(k)
                                + '_' + datetime.today().strftime('%Y%m%d%H%M%S') + '.jpg', ojodernorm)
                            cv2.imwrite(
                                directorio_base + 'ojoizq' + os.sep + 'ojoizq' + str(k) + os.sep + 'ojoizq' + str(k)
                                + '_' + datetime.today().strftime('%Y%m%d%H%M%S') + '.jpg', ojoizqnorm)
                            pulsado[k] = 0

            imgtk = arrayTOimgtk(cv2.resize(img, (int(img.shape[1] * 0.6), int(img.shape[0] * 0.6))))
            marco_camara.config(image=imgtk)
        print("Fin del hilo")
        cap.release()
    except Exception as e:
        print("saliendo del hilo por " + str(e))
        cap.release()

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


import os
import cv2
import dlib
import numpy as np
import tkinter as tk
from datetime import datetime

import os
import cv2
import dlib
import numpy as np
import tkinter as tk
from datetime import datetime

@csrf_exempt
def clasificaPedro(request):
    global izquierda_sup
    global derecha_sup
    global derecha_inf
    global izquierda_inf
    global centro

    numero_seleccionado = request.POST['panel_numero']

    # Directorio de imágenes capturadas desde el primer código
    directorio_imagenes = f'entrenamiento/fotos/panel{numero_seleccionado}/'

    # Directorio donde se guardarán los ojos normalizados
    directorio_ojos = f'entrenamiento/fotos/fotosnormalizadas/panel{numero_seleccionado}/'

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_68.dat')

    archivos = os.listdir(directorio_imagenes)

    for archivo in archivos:
        if archivo.endswith('.jpg'):
            # Leer la imagen desde el archivo
            img_path = os.path.join(directorio_imagenes, archivo)
            ima = cv2.imread(img_path)
            img = ima
            imgtk = arrayTOimgtk(cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2))))
            marco_camara = tk.Label(root, image=imgtk)
            marco_camara.place(x=ANCHO / 2 - int(np.shape(img)[1] * 0.6) / 2, y=140)

        try:
            while continua == 1:
                img = ima
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                rects = detector(gray, 1)
                if len(rects) == 0:
                    # No HAY CARAS
                    cv2.circle(img, (11, 11), 10, (0, 0, 255), -1)
                    root.configure(bg='grey')
                else:
                    cv2.circle(img, (11, 11), 10, (0, 255, 0), -1)
                    root.configure(bg='lightgrey')
                    for rect in rects:
                        shape = predictor(gray, rect)
                        shape = shape_to_np(shape)
                        # Resto del código...

                    imgtk = arrayTOimgtk(cv2.resize(img, (int(img.shape[1] * 0.6), int(img.shape[0] * 0.6))))
                    marco_camara.config(image=imgtk)
                    print("Fin del hilo")
                    ima.release()

        except Exception as e:
            ima.release()
    return JsonResponse({'error': str(e)}, status=500)




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
